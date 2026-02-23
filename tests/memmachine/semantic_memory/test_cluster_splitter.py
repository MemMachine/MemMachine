"""Tests for the reranker-guided cluster splitting module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from memmachine.common.episode_store import Episode
from memmachine.common.reranker import Reranker
from memmachine.semantic_memory.cluster_manager import (
    ClusterInfo,
    ClusterSplitParams,
    ClusterSplitRecord,
    ClusterState,
)
from memmachine.semantic_memory.cluster_splitter import (
    ContinuitySignals,
    NoOpClusterSplitter,
    RerankerClusterSplitter,
    SplitGate,
    apply_cluster_split,
    segment_cluster_id,
    validate_split_indices,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(uid: str, content: str, minutes_offset: int = 0) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="test-session",
        producer_id="test-producer",
        producer_role="user",
        created_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes_offset),
    )


def _make_signals(
    *,
    adj_sims: list[float] | None = None,
    time_gaps: list[float] | None = None,
) -> ContinuitySignals:
    sims = adj_sims or [0.9]
    gaps = time_gaps or [10.0]
    return ContinuitySignals(
        adjacent_similarities=sims,
        time_gaps_seconds=gaps,
        min_adjacent_similarity=min(sims),
        max_time_gap_seconds=max(gaps),
    )


class StubReranker(Reranker):
    def __init__(
        self,
        scores: list[float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._scores = list(scores or [])
        self._error = error

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        if self._error is not None:
            raise self._error
        if not self._scores:
            return [0.0 for _ in candidates]
        return [float(self._scores.pop(0))]


# ---------------------------------------------------------------------------
# validate_split_indices
# ---------------------------------------------------------------------------


class TestValidateSplitIndices:
    def test_valid_indices(self):
        assert validate_split_indices([3, 5], 10) == [3, 5]

    def test_empty_input(self):
        assert validate_split_indices([], 10) == []

    def test_out_of_range_low(self):
        assert validate_split_indices([0, 3], 10) == [3]

    def test_out_of_range_high(self):
        assert validate_split_indices([3, 10], 10) == [3]

    def test_duplicates_deduped(self):
        assert validate_split_indices([3, 3, 5], 10) == [3, 5]

    def test_unsorted_gets_sorted(self):
        assert validate_split_indices([5, 3], 10) == [3, 5]

    def test_non_integer_returns_empty(self):
        assert validate_split_indices(["a", "b"], 10) == []  # type: ignore[list-item]

    def test_single_message_window(self):
        assert validate_split_indices([1], 2) == [1]

    def test_all_invalid(self):
        assert validate_split_indices([0, 5], 5) == []


# ---------------------------------------------------------------------------
# segment_cluster_id
# ---------------------------------------------------------------------------


class TestSegmentClusterId:
    def test_determinism(self):
        id1 = segment_cluster_id("cluster_0", 1)
        id2 = segment_cluster_id("cluster_0", 1)
        assert id1 == id2

    def test_different_segments(self):
        id1 = segment_cluster_id("cluster_0", 1)
        id2 = segment_cluster_id("cluster_0", 2)
        assert id1 != id2

    def test_different_parents(self):
        id1 = segment_cluster_id("cluster_0", 1)
        id2 = segment_cluster_id("cluster_1", 1)
        assert id1 != id2

    def test_length(self):
        result = segment_cluster_id("cluster_0", 1)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# SplitGate
# ---------------------------------------------------------------------------


class TestSplitGate:
    def test_below_min_size_rejects(self):
        gate = SplitGate(ClusterSplitParams(min_cluster_size=6))
        messages = [_make_episode(f"m{i}", f"msg {i}") for i in range(5)]
        signals = _make_signals(adj_sims=[0.1, 0.1, 0.1, 0.1])
        assert gate.is_candidate(messages, signals) is False

    def test_at_min_size_with_low_similarity_accepts(self):
        gate = SplitGate(
            ClusterSplitParams(min_cluster_size=6, low_similarity_threshold=0.5)
        )
        messages = [_make_episode(f"m{i}", f"msg {i}") for i in range(6)]
        signals = _make_signals(adj_sims=[0.8, 0.8, 0.3, 0.8, 0.8])
        assert gate.is_candidate(messages, signals) is True

    def test_high_cohesion_rejects(self):
        gate = SplitGate(
            ClusterSplitParams(min_cluster_size=6, low_similarity_threshold=0.5)
        )
        messages = [_make_episode(f"m{i}", f"msg {i}") for i in range(6)]
        signals = _make_signals(adj_sims=[0.9, 0.85, 0.88, 0.92, 0.87])
        assert gate.is_candidate(messages, signals) is False

    def test_time_gap_triggers(self):
        gate = SplitGate(
            ClusterSplitParams(
                min_cluster_size=6,
                low_similarity_threshold=0.1,
                time_gap_seconds=3600.0,
            )
        )
        messages = [_make_episode(f"m{i}", f"msg {i}") for i in range(6)]
        signals = _make_signals(
            adj_sims=[0.9, 0.9, 0.9, 0.9, 0.9],
            time_gaps=[100.0, 100.0, 5000.0, 100.0, 100.0],
        )
        assert gate.is_candidate(messages, signals) is True

    def test_zscore_drop_triggers(self):
        gate = SplitGate(
            ClusterSplitParams(
                min_cluster_size=6,
                low_similarity_threshold=0.1,
                cohesion_drop_zscore=2.0,
            )
        )
        messages = [_make_episode(f"m{i}", f"msg {i}") for i in range(6)]
        # Sharp drop at index 2: all others are 0.9, one is 0.1
        signals = _make_signals(adj_sims=[0.9, 0.9, 0.1, 0.9, 0.9])
        assert gate.is_candidate(messages, signals) is True


# ---------------------------------------------------------------------------
# apply_cluster_split
# ---------------------------------------------------------------------------


class TestApplyClusterSplit:
    def test_splits_messages_correctly(self):
        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(6)]
        embeddings = [[float(i), 0.0] for i in range(6)]
        state = ClusterState(
            clusters={"cluster_0": ClusterInfo([0.5, 0.0], 6, messages[0].created_at)},
            event_to_cluster={f"m{i}": "cluster_0" for i in range(6)},
            next_cluster_id=1,
        )

        segments, _segment_ids = apply_cluster_split(
            "cluster_0", messages, embeddings, [3], state
        )

        assert len(segments) == 2
        assert segments[0][0] == "cluster_0"
        assert len(segments[0][1]) == 3
        assert segments[1][0] == segment_cluster_id("cluster_0", 1)
        assert len(segments[1][1]) == 3

    def test_first_segment_retains_original_id(self):
        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(4)]
        embeddings = [[float(i), 0.0] for i in range(4)]
        state = ClusterState()

        segments, _ = apply_cluster_split("cluster_5", messages, embeddings, [2], state)

        assert segments[0][0] == "cluster_5"

    def test_event_to_cluster_updated(self):
        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(4)]
        embeddings = [[float(i), 0.0] for i in range(4)]
        state = ClusterState()

        _, segment_ids = apply_cluster_split(
            "cluster_0", messages, embeddings, [2], state
        )

        assert state.event_to_cluster["m0"] == "cluster_0"
        assert state.event_to_cluster["m1"] == "cluster_0"
        assert state.event_to_cluster["m2"] == segment_ids[1]
        assert state.event_to_cluster["m3"] == segment_ids[1]

    def test_centroids_recomputed(self):
        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(4)]
        embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        state = ClusterState()

        _segments, segment_ids = apply_cluster_split(
            "cluster_0", messages, embeddings, [2], state
        )

        assert state.clusters["cluster_0"].centroid == [1.0, 0.0]
        assert state.clusters[segment_ids[1]].centroid == [0.0, 1.0]


# ---------------------------------------------------------------------------
# NoOpClusterSplitter
# ---------------------------------------------------------------------------


class TestNoOpClusterSplitter:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        splitter = NoOpClusterSplitter()
        messages = [_make_episode("m1", "hello")]
        clusters = [("cluster_0", messages)]
        state = ClusterState()

        result, _result_state = await splitter.maybe_split_clusters(
            cluster_messages=clusters,
            cluster_embeddings={"m1": [1.0, 0.0]},
            state=state,
            reranker=AsyncMock(),
        )

        assert len(result) == 1
        assert result[0] == ("cluster_0", messages)


# ---------------------------------------------------------------------------
# RerankerClusterSplitter
# ---------------------------------------------------------------------------


class TestRerankerClusterSplitter:
    @pytest.mark.asyncio
    async def test_skips_small_cluster(self):
        params = ClusterSplitParams(enabled=True, min_cluster_size=6)
        splitter = RerankerClusterSplitter(params)
        messages = [_make_episode(f"m{i}", f"msg {i}") for i in range(3)]
        clusters = [("cluster_0", messages)]
        embeddings = {f"m{i}": [float(i), 0.0] for i in range(3)}
        state = ClusterState()
        reranker = StubReranker(scores=[0.9, 0.9])

        result, state = await splitter.maybe_split_clusters(
            cluster_messages=clusters,
            cluster_embeddings=embeddings,
            state=state,
            reranker=reranker,
        )

        assert len(result) == 1
        assert result[0][0] == "cluster_0"
        assert "cluster_0" in state.split_records
        assert state.split_records["cluster_0"].segment_ids == []

    @pytest.mark.asyncio
    async def test_splits_when_reranker_scores_low(self):
        params = ClusterSplitParams(
            enabled=True, min_cluster_size=6, low_similarity_threshold=0.8
        )
        splitter = RerankerClusterSplitter(params)

        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(8)]
        # First 4 messages similar, last 4 different
        embeddings = {f"m{i}": [1.0, 0.0] if i < 4 else [0.0, 1.0] for i in range(8)}
        clusters = [("cluster_0", messages)]
        state = ClusterState()
        reranker = StubReranker(scores=[0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9])

        result, state = await splitter.maybe_split_clusters(
            cluster_messages=clusters,
            cluster_embeddings=embeddings,
            state=state,
            reranker=reranker,
        )

        assert len(result) == 2
        assert result[0][0] == "cluster_0"
        assert len(result[0][1]) == 4
        assert len(result[1][1]) == 4
        assert "cluster_0" in state.split_records
        assert len(state.split_records["cluster_0"].segment_ids) == 2

    @pytest.mark.asyncio
    async def test_no_split_when_reranker_scores_high(self):
        params = ClusterSplitParams(
            enabled=True, min_cluster_size=6, low_similarity_threshold=0.9
        )
        splitter = RerankerClusterSplitter(params)

        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(6)]
        embeddings = {f"m{i}": [1.0, 0.0] if i < 3 else [0.0, 1.0] for i in range(6)}
        clusters = [("cluster_0", messages)]
        state = ClusterState()
        reranker = StubReranker(scores=[0.9, 0.9, 0.9, 0.9, 0.9])

        result, state = await splitter.maybe_split_clusters(
            cluster_messages=clusters,
            cluster_embeddings=embeddings,
            state=state,
            reranker=reranker,
        )

        assert len(result) == 1
        assert result[0][0] == "cluster_0"
        assert state.split_records["cluster_0"].segment_ids == []

    @pytest.mark.asyncio
    async def test_skips_already_decided_cluster(self):
        params = ClusterSplitParams(
            enabled=True, min_cluster_size=6, low_similarity_threshold=0.9
        )
        splitter = RerankerClusterSplitter(params)

        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(6)]
        embeddings = {f"m{i}": [1.0, 0.0] for i in range(6)}
        clusters = [("cluster_0", messages)]
        input_hash = splitter._input_hash([f"m{i}" for i in range(6)])
        state = ClusterState(
            split_records={
                "cluster_0": ClusterSplitRecord(
                    original_cluster_id="cluster_0",
                    segment_ids=[],
                    input_hash=input_hash,
                ),
            },
        )

        reranker = AsyncMock()
        reranker.score = AsyncMock()

        result, _ = await splitter.maybe_split_clusters(
            cluster_messages=clusters,
            cluster_embeddings=embeddings,
            state=state,
            reranker=reranker,
        )

        reranker.score.assert_not_awaited()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_recomputes_when_input_changes(self):
        params = ClusterSplitParams(
            enabled=True, min_cluster_size=2, low_similarity_threshold=0.8
        )
        splitter = RerankerClusterSplitter(params)

        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(3)]
        embeddings = {
            "m0": [1.0, 0.0],
            "m1": [0.0, 1.0],
            "m2": [1.0, 0.0],
        }
        clusters = [("cluster_0", messages)]
        state = ClusterState(
            split_records={
                "cluster_0": ClusterSplitRecord(
                    original_cluster_id="cluster_0",
                    segment_ids=[],
                    input_hash="stale",
                )
            },
        )

        reranker = AsyncMock()
        reranker.score = AsyncMock(side_effect=[[0.9], [0.9]])

        result, state = await splitter.maybe_split_clusters(
            cluster_messages=clusters,
            cluster_embeddings=embeddings,
            state=state,
            reranker=reranker,
        )

        assert len(result) == 1
        assert reranker.score.await_count == 2
        assert state.split_records["cluster_0"].input_hash == splitter._input_hash(
            [m.uid for m in messages if m.uid is not None]
        )

    @pytest.mark.asyncio
    async def test_fallback_on_reranker_error(self):
        params = ClusterSplitParams(
            enabled=True, min_cluster_size=6, low_similarity_threshold=0.9
        )
        splitter = RerankerClusterSplitter(params)

        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(6)]
        embeddings = {f"m{i}": [1.0, 0.0] if i < 3 else [0.0, 1.0] for i in range(6)}
        clusters = [("cluster_0", messages)]
        state = ClusterState()
        reranker = StubReranker(error=RuntimeError("Reranker unavailable"))

        result, state = await splitter.maybe_split_clusters(
            cluster_messages=clusters,
            cluster_embeddings=embeddings,
            state=state,
            reranker=reranker,
        )

        assert len(result) == 1
        assert result[0][0] == "cluster_0"
        assert state.split_records["cluster_0"].segment_ids == []

    @pytest.mark.asyncio
    async def test_debug_fail_loudly_reraises(self):
        params = ClusterSplitParams(
            enabled=True,
            min_cluster_size=6,
            low_similarity_threshold=0.8,
            debug_fail_loudly=True,
        )
        splitter = RerankerClusterSplitter(params)

        messages = [_make_episode(f"m{i}", f"msg {i}", i) for i in range(6)]
        # Mix embeddings so adjacent similarity drops below threshold
        embeddings = {f"m{i}": [1.0, 0.0] if i < 3 else [0.0, 1.0] for i in range(6)}
        clusters = [("cluster_0", messages)]
        state = ClusterState()
        reranker = StubReranker(error=RuntimeError("Reranker unavailable"))

        with pytest.raises(RuntimeError, match="Reranker unavailable"):
            await splitter.maybe_split_clusters(
                cluster_messages=clusters,
                cluster_embeddings=embeddings,
                state=state,
                reranker=reranker,
            )
