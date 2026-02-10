from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from memmachine.semantic_memory.cluster_manager import ClusterInfo, ClusterState
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
    loaded.next_cluster_id = 3

    await cluster_state_storage.save_state(set_id="set-f", state=loaded)

    reloaded = await cluster_state_storage.get_state(set_id="set-f")
    assert reloaded == loaded
