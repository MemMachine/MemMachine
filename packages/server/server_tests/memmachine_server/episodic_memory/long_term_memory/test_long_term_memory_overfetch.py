"""Unit tests for the temporal overfetch selection policy.

The selection is a pure function of best-first match scores plus knobs; it lives
on ``LongTermMemory`` because it is part of the event-backend search wiring (not
a temporal primitive). These tests exercise it directly, with no backend.
"""

from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
    LongTermMemory,
)


def test_no_temporal_match_collapses_to_top_k_by_cosine():
    # All-zero match scores -> nothing clears the gate -> pure cosine top-k.
    selected = LongTermMemory._select_temporal_overfetch_indices(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        limit=3,
        temporal_fraction=1.0 / 3.0,
        match_threshold=0.0,
    )
    assert selected == [0, 1, 2]


def test_promotes_low_cosine_match_into_k2():
    # k=3 -> k2=round(1)=1, k1=2. Indices 0,1 fill k1 by cosine; the only
    # temporal match (index 4) is promoted over the higher-cosine 2,3. Output
    # stays in cosine (index) order: membership changes, not ordering.
    selected = LongTermMemory._select_temporal_overfetch_indices(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        limit=3,
        temporal_fraction=1.0 / 3.0,
        match_threshold=0.0,
    )
    assert selected == [0, 1, 4]


def test_backfills_with_cosine_when_too_few_match():
    # k2 reserves 1 slot but nothing matches -> backfill the next cosine hit.
    selected = LongTermMemory._select_temporal_overfetch_indices(
        [0.0, 0.0, 0.0, 0.0],
        limit=3,
        temporal_fraction=1.0 / 3.0,
        match_threshold=0.0,
    )
    assert selected == [0, 1, 2]


def test_gate_is_strictly_greater_than_threshold():
    # k=2 -> k2=1, k1=1. With threshold 0.0001, the 0.0001 candidate does NOT
    # clear the strict gate, so k2 backfills with cosine (index 1).
    selected = LongTermMemory._select_temporal_overfetch_indices(
        [0.0, 0.0, 0.0, 0.0001],
        limit=2,
        temporal_fraction=0.5,
        match_threshold=0.0001,
    )
    assert selected == [0, 1]
    # Lowering the threshold below the match lets it clear the gate and promote
    # over the higher-cosine index 1.
    selected = LongTermMemory._select_temporal_overfetch_indices(
        [0.0, 0.0, 0.0, 0.0001],
        limit=2,
        temporal_fraction=0.5,
        match_threshold=0.0,
    )
    assert selected == [0, 3]


def test_fraction_one_fills_entirely_from_matches():
    # temporal_fraction=1 -> k2=k, k1=0: drop the top-cosine non-matches in
    # favour of temporally-matching candidates.
    selected = LongTermMemory._select_temporal_overfetch_indices(
        [0.0, 0.0, 0.5, 0.5],
        limit=2,
        temporal_fraction=1.0,
        match_threshold=0.0,
    )
    assert selected == [2, 3]


def test_limit_exceeding_pool_returns_all():
    selected = LongTermMemory._select_temporal_overfetch_indices(
        [0.0, 0.5],
        limit=5,
        temporal_fraction=1.0 / 3.0,
        match_threshold=0.0,
    )
    assert selected == [0, 1]


def test_empty_pool():
    assert (
        LongTermMemory._select_temporal_overfetch_indices(
            [], limit=3, temporal_fraction=0.5, match_threshold=0.0
        )
        == []
    )
