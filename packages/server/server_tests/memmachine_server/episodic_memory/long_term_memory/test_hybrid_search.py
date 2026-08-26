"""Unit tests for the hybrid search merge logic in `LongTermMemory`.

These tests do NOT require a live Neo4j instance. They patch
`_search_scored_declarative` and `_search_fts` to return canned results and
verify the merge/dedup/fusion behavior of `_search_scored_hybrid` directly:

- append mode respects `append_n`
- append mode deduplicates by episode uid
- RRF mode (True / "rrf") fuses vector + FTS by rank, selects top-k, then
  re-sorts by timestamp
- an invalid `use_fts` value raises ValueError
- score threshold is applied after the merge/fusion
"""

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from memmachine_server.common.episode_store import Episode
from memmachine_server.episodic_memory.long_term_memory import LongTermMemory


def _episode(
    uid: str, content: str = "", *, created_at: datetime | None = None
) -> Episode:
    """Build a minimal Episode for testing."""
    return Episode(
        uid=uid,
        content=content,
        session_key="",
        created_at=created_at or datetime.now(UTC),
        producer_id="",
        producer_role="",
    )


def _make_long_term_memory() -> LongTermMemory:
    """Construct a LongTermMemory without running its full __init__.

    `_search_scored_hybrid` only touches `self._declarative_memory` (asserted
    non-None), so we bypass __init__ and set that attribute directly.
    """
    ltm = LongTermMemory.__new__(LongTermMemory)
    ltm._declarative_memory = object()  # truthy; satisfies the assert
    return ltm


@pytest.mark.asyncio
async def test_append_mode_respects_append_n():
    """Append mode returns at most `append_n` unique FTS results after vector."""
    vector_results = [
        (0.9, _episode("v1", "vector hit one")),
        (0.8, _episode("v2", "vector hit two")),
    ]
    # FTS returns 50 candidates (over-fetch); append_n=5 should slice to 5.
    fts_results = [(float(i), _episode(f"f{i}", f"fts hit {i}")) for i in range(50)]

    ltm = _make_long_term_memory()
    with (
        patch.object(
            ltm,
            "_search_scored_declarative",
            new=AsyncMock(return_value=vector_results),
        ),
        patch.object(
            ltm,
            "_search_fts",
            new=AsyncMock(return_value=fts_results),
        ) as fts_mock,
    ):
        merged = await ltm._search_scored_hybrid(
            "query",
            num_episodes_limit=10,
            expand_context=0,
            score_threshold=None,
            property_filter=None,
            use_fts="append",
            append_n=5,
        )

    # 2 vector results + 5 appended FTS results = 7
    assert len(merged) == 7
    uids = [ep.uid for _, ep in merged]
    assert uids[:2] == ["v1", "v2"]
    # The 5 appended come from the first 5 FTS results (highest score).
    assert uids[2:] == ["f0", "f1", "f2", "f3", "f4"]

    # FTS is called with num_episodes_limit (used for over-fetch sizing), not append_n.
    fts_mock.assert_awaited_once()
    assert fts_mock.call_args.kwargs["num_episodes_limit"] == 10


@pytest.mark.asyncio
async def test_append_mode_dedups_by_uid():
    """FTS results whose uid already exists in vector results are skipped."""
    vector_results = [(0.9, _episode("shared", "in both"))]
    fts_results = [
        (0.7, _episode("shared", "duplicate uid")),
        (0.6, _episode("unique", "only in fts")),
    ]

    ltm = _make_long_term_memory()
    with (
        patch.object(
            ltm,
            "_search_scored_declarative",
            new=AsyncMock(return_value=vector_results),
        ),
        patch.object(
            ltm,
            "_search_fts",
            new=AsyncMock(return_value=fts_results),
        ),
    ):
        merged = await ltm._search_scored_hybrid(
            "query",
            num_episodes_limit=10,
            expand_context=0,
            score_threshold=None,
            property_filter=None,
            use_fts="append",
            append_n=10,
        )

    uids = [ep.uid for _, ep in merged]
    # "shared" appears once (vector), "unique" appended once.
    assert uids == ["shared", "unique"]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_fts", [True, "rrf"])
async def test_rrf_mode_fuses_and_sorts_by_timestamp(use_fts):
    """RRF fuses vector + FTS ranks, selects top-k, then sorts by timestamp.

    Setup: vector ranks v1(1) > v2(2); FTS ranks f1(1) > v1(2) > f2(3).
    v1 appears in both lists → highest RRF score. Final output is re-sorted by
    timestamp, not by RRF score.
    """
    now = datetime.now(UTC)
    # Deliberate timestamps that do NOT match relevance rank, so we can confirm
    # the final ordering is by timestamp, not RRF score.
    v1 = _episode("v1", "vector one", created_at=now + timedelta(seconds=20))
    v2 = _episode("v2", "vector two", created_at=now + timedelta(seconds=10))
    f1 = _episode("f1", "fts one", created_at=now + timedelta(seconds=5))
    f2 = _episode("f2", "fts two", created_at=now + timedelta(seconds=0))

    vector_results = [(0.9, v1), (0.8, v2)]  # ranks 1, 2
    fts_results = [(0.9, f1), (0.7, v1), (0.6, f2)]  # ranks 1, 2, 3

    ltm = _make_long_term_memory()
    with (
        patch.object(
            ltm,
            "_search_scored_declarative",
            new=AsyncMock(return_value=vector_results),
        ),
        patch.object(
            ltm,
            "_search_fts",
            new=AsyncMock(return_value=fts_results),
        ),
    ):
        merged = await ltm._search_scored_hybrid(
            "query",
            num_episodes_limit=10,
            expand_context=0,
            score_threshold=None,
            property_filter=None,
            use_fts=use_fts,
            append_n=10,
        )

    # All 4 distinct uids are selected (limit 10). v1 (in both lists) must be
    # present. Final order must be by timestamp ascending (oldest first).
    uids = [ep.uid for _, ep in merged]
    assert set(uids) == {"v1", "v2", "f1", "f2"}
    assert uids == ["f2", "f1", "v2", "v1"]  # timestamp ascending

    # v1 must have the highest RRF score (appears in both rankings).
    score_by_uid = {ep.uid: score for score, ep in merged}
    assert score_by_uid["v1"] > score_by_uid["f1"]
    assert score_by_uid["v1"] > score_by_uid["v2"]


@pytest.mark.asyncio
async def test_rrf_mode_respects_top_k_limit():
    """RRF returns at most `num_episodes_limit` candidates (top-k selection)."""
    vector_results = [(float(10 - i), _episode(f"v{i}")) for i in range(8)]
    fts_results = [(float(10 - i), _episode(f"f{i}")) for i in range(8)]

    ltm = _make_long_term_memory()
    with (
        patch.object(
            ltm,
            "_search_scored_declarative",
            new=AsyncMock(return_value=vector_results),
        ),
        patch.object(
            ltm,
            "_search_fts",
            new=AsyncMock(return_value=fts_results),
        ),
    ):
        merged = await ltm._search_scored_hybrid(
            "query",
            num_episodes_limit=5,
            expand_context=0,
            score_threshold=None,
            property_filter=None,
            use_fts=True,
            append_n=10,
        )

    assert len(merged) == 5


@pytest.mark.asyncio
async def test_rrf_mode_dedups_across_lists():
    """An episode appearing in both vector and FTS contributes RRF from both."""
    shared = _episode("shared", "in both")
    vector_results = [(0.9, shared)]  # rank 1 -> 1/(60+1)
    fts_results = [(0.8, shared)]  # rank 1 -> 1/(60+1)

    ltm = _make_long_term_memory()
    with (
        patch.object(
            ltm,
            "_search_scored_declarative",
            new=AsyncMock(return_value=vector_results),
        ),
        patch.object(
            ltm,
            "_search_fts",
            new=AsyncMock(return_value=fts_results),
        ),
    ):
        merged = await ltm._search_scored_hybrid(
            "query",
            num_episodes_limit=10,
            expand_context=0,
            score_threshold=None,
            property_filter=None,
            use_fts="rrf",
            append_n=10,
        )

    # Only one entry despite appearing in both lists.
    assert len(merged) == 1
    score, ep = merged[0]
    assert ep.uid == "shared"
    # 1/(60+1) + 1/(60+1) == 2/61
    assert score == pytest.approx(2.0 / 61.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", ["rrf2", "RRF", "unknown", 1, "true"])
async def test_invalid_use_fts_raises_value_error(bad_value):
    """An invalid `use_fts` value must raise ValueError, not fall through."""
    ltm = _make_long_term_memory()
    # Use ExitStack so the two patches and pytest.raises live as three
    # independent enter_context calls under one `with`, instead of nested
    # `with` statements that ruff's SIM117 would try to merge (merging them
    # turns the patches into a tuple, which has no context-manager protocol).
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                ltm,
                "_search_scored_declarative",
                new=AsyncMock(return_value=[]),
            )
        )
        stack.enter_context(
            patch.object(
                ltm,
                "_search_fts",
                new=AsyncMock(return_value=[]),
            )
        )
        with pytest.raises(ValueError, match="Invalid use_fts value"):
            await ltm._search_scored_hybrid(
                "query",
                num_episodes_limit=10,
                expand_context=0,
                score_threshold=None,
                property_filter=None,
                use_fts=bad_value,  # type: ignore[arg-type]
                append_n=10,
            )


@pytest.mark.asyncio
async def test_append_mode_applies_score_threshold():
    """Score threshold filters merged results (drops below threshold)."""
    vector_results = [
        (0.9, _episode("v1")),
        (0.3, _episode("v2")),  # below threshold
    ]
    fts_results = [
        (0.8, _episode("f1")),
        (0.2, _episode("f2")),  # below threshold
    ]

    ltm = _make_long_term_memory()
    with (
        patch.object(
            ltm,
            "_search_scored_declarative",
            new=AsyncMock(return_value=vector_results),
        ),
        patch.object(
            ltm,
            "_search_fts",
            new=AsyncMock(return_value=fts_results),
        ),
    ):
        merged = await ltm._search_scored_hybrid(
            "query",
            num_episodes_limit=10,
            expand_context=0,
            score_threshold=0.5,
            property_filter=None,
            use_fts="append",
            append_n=10,
        )

    uids = [ep.uid for _, ep in merged]
    assert uids == ["v1", "f1"]
