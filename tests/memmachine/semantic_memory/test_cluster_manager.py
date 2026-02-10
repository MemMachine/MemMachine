from datetime import UTC, datetime, timedelta

import pytest

from memmachine.semantic_memory.cluster_manager import (
    ClusterInfo,
    ClusterManager,
    ClusterParams,
    ClusterState,
)


def test_assign_creates_new_cluster_when_empty() -> None:
    manager = ClusterManager(ClusterParams(similarity_threshold=0.8))
    now = datetime.now(tz=UTC)

    assignment, state = manager.assign(
        event_id="e1",
        embedding=[1.0, 0.0],
        timestamp=now,
        state=ClusterState(),
    )

    assert assignment.created_new is True
    assert assignment.cluster_id == "cluster_0"
    assert state.event_to_cluster["e1"] == "cluster_0"
    assert state.clusters["cluster_0"].count == 1


def test_assign_reuses_cluster_when_similar() -> None:
    manager = ClusterManager(ClusterParams(similarity_threshold=0.8))
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now - timedelta(minutes=1),
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e2",
        embedding=[0.9, 0.1],
        timestamp=now,
        state=state,
    )

    assert assignment.created_new is False
    assert assignment.cluster_id == "cluster_0"
    assert state.clusters["cluster_0"].count == 2
    assert state.event_to_cluster["e2"] == "cluster_0"
    assert state.clusters["cluster_0"].centroid == pytest.approx([0.95, 0.05])


def test_assign_creates_new_cluster_when_similarity_low() -> None:
    manager = ClusterManager(ClusterParams(similarity_threshold=0.9))
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e3",
        embedding=[0.0, 1.0],
        timestamp=now,
        state=state,
    )

    assert assignment.created_new is True
    assert assignment.cluster_id == "cluster_1"
    assert state.clusters["cluster_0"].count == 1
    assert state.clusters["cluster_1"].count == 1


def test_assign_zero_vector_creates_new_cluster() -> None:
    manager = ClusterManager(ClusterParams(similarity_threshold=0.1))
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e_zero",
        embedding=[0.0, 0.0],
        timestamp=now,
        state=state,
    )

    assert assignment.created_new is True
    assert assignment.cluster_id == "cluster_1"
    assert state.clusters["cluster_0"].count == 1


def test_assign_reuses_cluster_on_similarity_threshold_boundary() -> None:
    manager = ClusterManager(ClusterParams(similarity_threshold=1.0))
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e_boundary",
        embedding=[1.0, 0.0],
        timestamp=now,
        state=state,
    )

    assert assignment.created_new is False
    assert assignment.cluster_id == "cluster_0"
    assert state.clusters["cluster_0"].count == 2


def test_time_gap_gate_creates_new_cluster() -> None:
    manager = ClusterManager(
        ClusterParams(similarity_threshold=0.2, max_time_gap=timedelta(days=1))
    )
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now - timedelta(days=2),
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e4",
        embedding=[1.0, 0.0],
        timestamp=now,
        state=state,
    )

    assert assignment.created_new is True
    assert assignment.cluster_id == "cluster_1"
    assert state.clusters["cluster_0"].count == 1


def test_time_gap_zero_reuses_same_timestamp() -> None:
    manager = ClusterManager(
        ClusterParams(similarity_threshold=0.2, max_time_gap=timedelta(0))
    )
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e_same_ts",
        embedding=[1.0, 0.0],
        timestamp=now,
        state=state,
    )

    assert assignment.created_new is False
    assert assignment.cluster_id == "cluster_0"
    assert state.clusters["cluster_0"].count == 2


def test_time_gap_zero_blocks_different_timestamp() -> None:
    manager = ClusterManager(
        ClusterParams(similarity_threshold=0.2, max_time_gap=timedelta(0))
    )
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e_next_ts",
        embedding=[1.0, 0.0],
        timestamp=now + timedelta(seconds=1),
        state=state,
    )

    assert assignment.created_new is True
    assert assignment.cluster_id == "cluster_1"
    assert state.clusters["cluster_0"].count == 1


def test_time_gap_allows_negative_timestamp_within_limit() -> None:
    manager = ClusterManager(
        ClusterParams(similarity_threshold=0.2, max_time_gap=timedelta(minutes=1))
    )
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        next_cluster_id=1,
    )

    assignment, state = manager.assign(
        event_id="e_earlier",
        embedding=[1.0, 0.0],
        timestamp=now - timedelta(seconds=30),
        state=state,
    )

    assert assignment.created_new is False
    assert assignment.cluster_id == "cluster_0"
    assert state.clusters["cluster_0"].count == 2


def test_assign_raises_on_embedding_length_mismatch() -> None:
    manager = ClusterManager(ClusterParams(similarity_threshold=0.5))
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        }
    )

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        manager.assign(
            event_id="e5",
            embedding=[1.0, 0.0, 0.0],
            timestamp=now,
            state=state,
        )
