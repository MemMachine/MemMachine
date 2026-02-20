from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from memmachine.semantic_memory.cluster_manager import (
    ClusterInfo,
    ClusterSplitRecord,
    ClusterState,
)
from memmachine.semantic_memory.cluster_store.cluster_store import ClusterStateStorage
from memmachine.semantic_memory.cluster_store.cluster_store_sqlalchemy import (
    BaseClusterStore,
    ClusterStateStorageSqlAlchemy,
)
from memmachine.semantic_memory.cluster_store.in_memory_cluster_store import (
    InMemoryClusterStateStorage,
)


@pytest_asyncio.fixture
async def sqlite_cluster_state_storage(sqlalchemy_sqlite_engine):
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(BaseClusterStore.metadata.drop_all)
        await conn.run_sync(BaseClusterStore.metadata.create_all)

    storage = ClusterStateStorageSqlAlchemy(sqlalchemy_sqlite_engine)
    await storage.startup()
    yield storage

    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(BaseClusterStore.metadata.drop_all)


@pytest_asyncio.fixture
async def pg_cluster_state_storage(sqlalchemy_pg_engine):
    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseClusterStore.metadata.drop_all)
        await conn.run_sync(BaseClusterStore.metadata.create_all)

    storage = ClusterStateStorageSqlAlchemy(sqlalchemy_pg_engine)
    await storage.startup()
    yield storage

    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseClusterStore.metadata.drop_all)


@pytest_asyncio.fixture
async def in_memory_cluster_state_storage():
    storage = InMemoryClusterStateStorage()
    await storage.startup()
    yield storage
    await storage.delete_all()


@pytest.fixture(
    params=[
        "sqlite_cluster_state_storage",
        pytest.param("pg_cluster_state_storage", marks=pytest.mark.integration),
        "in_memory_cluster_state_storage",
    ]
)
def cluster_state_storage(request):
    return request.getfixturevalue(request.param)


def _sample_state(now: datetime) -> ClusterState:
    return ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=2,
                last_ts=now - timedelta(minutes=1),
            ),
            "cluster_1": ClusterInfo(
                centroid=[0.0, 1.0],
                count=1,
                last_ts=now,
            ),
        },
        event_to_cluster={
            "event-a": "cluster_0",
            "event-b": "cluster_1",
        },
        pending_events={
            "cluster_0": {"event-a": now - timedelta(minutes=2)},
            "cluster_1": {"event-b": now - timedelta(minutes=1)},
        },
        next_cluster_id=2,
    )


@pytest.mark.asyncio
async def test_get_state_returns_none_when_missing(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    loaded = await cluster_state_storage.get_state(set_id="missing")
    assert loaded is None


@pytest.mark.asyncio
async def test_round_trip_state(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    now = datetime.now(tz=UTC)
    state = _sample_state(now)
    await cluster_state_storage.save_state(set_id="set-a", state=state)

    loaded = await cluster_state_storage.get_state(set_id="set-a")

    assert loaded is not None
    assert loaded == state


@pytest.mark.asyncio
async def test_delete_state(cluster_state_storage: ClusterStateStorage) -> None:
    now = datetime.now(tz=UTC)
    await cluster_state_storage.save_state(set_id="set-b", state=_sample_state(now))

    await cluster_state_storage.delete_state(set_id="set-b")

    loaded = await cluster_state_storage.get_state(set_id="set-b")
    assert loaded is None


@pytest.mark.asyncio
async def test_delete_all(cluster_state_storage: ClusterStateStorage) -> None:
    now = datetime.now(tz=UTC)
    await cluster_state_storage.save_state(set_id="set-c", state=_sample_state(now))
    await cluster_state_storage.save_state(set_id="set-d", state=_sample_state(now))

    await cluster_state_storage.delete_all()

    assert await cluster_state_storage.get_state(set_id="set-c") is None
    assert await cluster_state_storage.get_state(set_id="set-d") is None


@pytest.mark.asyncio
async def test_save_overwrites_state(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    now = datetime.now(tz=UTC)
    await cluster_state_storage.save_state(set_id="set-e", state=_sample_state(now))

    new_state = ClusterState(
        clusters={
            "cluster_2": ClusterInfo(
                centroid=[0.5, 0.5],
                count=1,
                last_ts=now + timedelta(minutes=5),
            )
        },
        event_to_cluster={"event-c": "cluster_2"},
        pending_events={"cluster_2": {"event-c": now + timedelta(minutes=4)}},
        next_cluster_id=3,
    )

    await cluster_state_storage.save_state(set_id="set-e", state=new_state)

    loaded = await cluster_state_storage.get_state(set_id="set-e")
    assert loaded == new_state


@pytest.mark.asyncio
async def test_save_reload_and_update_state(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    now = datetime.now(tz=UTC)
    state = _sample_state(now)
    await cluster_state_storage.save_state(set_id="set-f", state=state)

    loaded = await cluster_state_storage.get_state(set_id="set-f")
    assert loaded is not None

    loaded.clusters["cluster_2"] = ClusterInfo(
        centroid=[0.25, 0.75],
        count=1,
        last_ts=now + timedelta(minutes=10),
    )
    loaded.event_to_cluster["event-c"] = "cluster_2"
    loaded.pending_events.setdefault("cluster_2", {})["event-c"] = now + timedelta(
        minutes=9
    )
    loaded.next_cluster_id = 3

    await cluster_state_storage.save_state(set_id="set-f", state=loaded)

    reloaded = await cluster_state_storage.get_state(set_id="set-f")
    assert reloaded == loaded


@pytest.mark.asyncio
async def test_round_trip_state_with_split_records(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=2,
                last_ts=now - timedelta(minutes=1),
            ),
            "cluster_1": ClusterInfo(
                centroid=[0.0, 1.0],
                count=1,
                last_ts=now,
            ),
        },
        event_to_cluster={
            "event-a": "cluster_0",
            "event-b": "cluster_1",
        },
        pending_events={
            "cluster_0": {"event-a": now - timedelta(minutes=2)},
            "cluster_1": {"event-b": now - timedelta(minutes=1)},
        },
        next_cluster_id=2,
        split_records={
            "cluster_0": ClusterSplitRecord(
                original_cluster_id="cluster_0",
                segment_ids=["cluster_0", "abcdef1234567890"],
                input_hash="sha256hex",
            )
        },
    )

    await cluster_state_storage.save_state(set_id="set-split-a", state=state)

    loaded = await cluster_state_storage.get_state(set_id="set-split-a")
    assert loaded == state


@pytest.mark.asyncio
async def test_overwrite_state_replaces_split_records(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    now = datetime.now(tz=UTC)
    initial_state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        event_to_cluster={"event-a": "cluster_0"},
        pending_events={"cluster_0": {"event-a": now}},
        next_cluster_id=1,
        split_records={
            "cluster_0": ClusterSplitRecord(
                original_cluster_id="cluster_0",
                segment_ids=["cluster_0", "old-seg"],
                input_hash="hash-a",
            )
        },
    )
    await cluster_state_storage.save_state(
        set_id="set-split-b",
        state=initial_state,
    )

    new_state = ClusterState(
        clusters={
            "cluster_1": ClusterInfo(
                centroid=[0.25, 0.75],
                count=1,
                last_ts=now + timedelta(minutes=5),
            )
        },
        event_to_cluster={"event-b": "cluster_1"},
        pending_events={"cluster_1": {"event-b": now + timedelta(minutes=4)}},
        next_cluster_id=2,
        split_records={
            "cluster_1": ClusterSplitRecord(
                original_cluster_id="cluster_1",
                segment_ids=["cluster_1", "new-seg"],
                input_hash="hash-b",
            )
        },
    )
    await cluster_state_storage.save_state(set_id="set-split-b", state=new_state)

    loaded = await cluster_state_storage.get_state(set_id="set-split-b")
    assert loaded == new_state


@pytest.mark.asyncio
async def test_delete_state_removes_split_records(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        event_to_cluster={"event-a": "cluster_0"},
        pending_events={"cluster_0": {"event-a": now}},
        next_cluster_id=1,
        split_records={
            "cluster_0": ClusterSplitRecord(
                original_cluster_id="cluster_0",
                segment_ids=["cluster_0", "seg-1"],
                input_hash="hash-a",
            )
        },
    )
    await cluster_state_storage.save_state(set_id="set-split-c", state=state)

    await cluster_state_storage.delete_state(set_id="set-split-c")

    loaded = await cluster_state_storage.get_state(set_id="set-split-c")
    assert loaded is None


@pytest.mark.asyncio
async def test_round_trip_empty_split_records(
    cluster_state_storage: ClusterStateStorage,
) -> None:
    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        event_to_cluster={"event-a": "cluster_0"},
        pending_events={"cluster_0": {"event-a": now}},
        next_cluster_id=1,
        split_records={},
    )

    await cluster_state_storage.save_state(set_id="set-split-d", state=state)

    loaded = await cluster_state_storage.get_state(set_id="set-split-d")
    assert loaded is not None
    assert loaded.split_records == {}
