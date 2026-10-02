"""History ordering contracts shared by local and integration backends."""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import create_autospec
from uuid import UUID

import pytest
import pytest_asyncio
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store import VectorStoreCollectionConfig
from memmachine_server.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
)
from memmachine_server.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    BaseSemanticStorage,
    SqlAlchemyPgVectorSemanticStorage,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorStoreSemanticStorage,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_collection import (
    InMemoryVectorStoreCollection,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def sqlite_pgvector_history_storage(sqlalchemy_sqlite_engine: AsyncEngine):
    # Only the portable history table is needed to execute the real SQL methods.
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(
            BaseSemanticStorage.metadata.create_all,
            tables=[BaseSemanticStorage.metadata.tables["set_ingested_history"]],
        )
    return SqlAlchemyPgVectorSemanticStorage(sqlalchemy_sqlite_engine)


@pytest_asyncio.fixture
async def sqlite_vector_history_storage(sqlalchemy_sqlite_engine: AsyncEngine):
    collection = InMemoryVectorStoreCollection(
        VectorStoreCollectionConfig(
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.COSINE,
        )
    )
    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, collection)
    await storage.startup()
    return storage


@pytest.fixture(
    params=[
        "in_memory_semantic_storage",
        "sqlite_pgvector_history_storage",
        "sqlite_vector_history_storage",
        pytest.param("pgvector_semantic_storage", marks=pytest.mark.integration),
        pytest.param("neo4j_semantic_storage", marks=pytest.mark.integration),
    ]
)
def history_storage(request: pytest.FixtureRequest) -> SemanticStorage:
    return request.getfixturevalue(request.param)


async def test_history_registration_times_are_scoped_to_set(
    history_storage: SemanticStorage,
):
    history_id = UUID("10000000-0000-4000-8000-000000000000")
    other_id = UUID("20000000-0000-4000-8000-000000000000")
    registered_at = datetime(2025, 1, 1, tzinfo=timezone(timedelta(hours=5)))
    await history_storage.add_history_to_set(
        "target", history_id, registered_at=registered_at
    )
    await history_storage.add_history_to_set("other", other_id)

    assert await history_storage.get_history_registration_times(
        "target", [history_id, other_id]
    ) == {history_id: registered_at.astimezone(UTC)}


async def test_history_limit_selects_oldest_episode_times(
    history_storage: SemanticStorage,
):
    # Lexical UUID order and registration order both oppose episode chronology.
    ids = [UUID(f"{prefix}0000000-0000-4000-8000-000000000000") for prefix in "fedcba0"]
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for offset in reversed(range(len(ids))):
        await history_storage.add_history_to_set(
            "ordered", ids[offset], created_at=start + timedelta(hours=offset)
        )

    unrelated = UUID("10000000-0000-4000-8000-000000000000")
    done = UUID("20000000-0000-4000-8000-000000000000")
    await history_storage.add_history_to_set("other", unrelated, created_at=start)
    await history_storage.add_history_to_set("ordered", done, created_at=start)
    await history_storage.mark_messages_ingested(set_id="ordered", history_ids=[done])

    first_batch = [
        history_id
        async for history_id in history_storage.get_history_messages(
            set_ids=["ordered"], is_ingested=False, limit=5
        )
    ]
    assert first_batch == ids[:5]
    await history_storage.mark_messages_ingested(
        set_id="ordered", history_ids=first_batch
    )
    assert [
        history_id
        async for history_id in history_storage.get_history_messages(
            set_ids=["ordered"], is_ingested=False, limit=5
        )
    ] == ids[5:]


async def test_history_equal_times_preserve_global_sequence(
    history_storage: SemanticStorage,
):
    created_at = datetime(2025, 1, 1, tzinfo=UTC)
    ids = [
        UUID("f0000000-0000-4000-8000-000000000000"),
        UUID("00000000-0000-4000-8000-000000000000"),
    ]
    # The later sequence registers first and has the earlier registration time.
    await history_storage.add_history_to_set(
        "ties",
        ids[1],
        created_at=created_at,
        registered_at=created_at,
        sequence_num=11,
    )
    await history_storage.add_history_to_set(
        "ties",
        ids[0],
        created_at=created_at,
        registered_at=created_at + timedelta(hours=1),
        sequence_num=10,
    )

    assert [
        history_id
        async for history_id in history_storage.get_history_messages(
            set_ids=["ties"], is_ingested=False
        )
    ] == ids


async def test_history_debounce_uses_registration_time(
    history_storage: SemanticStorage,
):
    now = datetime.now(UTC)
    await history_storage.add_history_to_set(
        "past",
        UUID("10000000-0000-4000-8000-000000000000"),
        created_at=now - timedelta(days=5),
        registered_at=now,
    )
    await history_storage.add_history_to_set(
        "future",
        UUID("20000000-0000-4000-8000-000000000000"),
        created_at=now + timedelta(days=5),
        registered_at=now,
    )

    assert [
        sid
        async for sid in history_storage.get_history_set_ids(
            older_than=now - timedelta(minutes=5)
        )
    ] == []
    assert {
        sid async for sid in history_storage.get_history_set_ids(older_than=now)
    } == {"past", "future"}


async def test_history_order_compares_timestamp_instants(
    history_storage: SemanticStorage,
):
    earlier = UUID("f0000000-0000-4000-8000-000000000000")
    later = UUID("00000000-0000-4000-8000-000000000000")
    await history_storage.add_history_to_set(
        "zones",
        later,
        created_at=datetime(2025, 1, 1, 9, tzinfo=timezone(timedelta(hours=5))),
    )
    await history_storage.add_history_to_set(
        "zones", earlier, created_at=datetime(2025, 1, 1, 3, tzinfo=UTC)
    )

    assert [
        uid async for uid in history_storage.get_history_messages(set_ids=["zones"])
    ] == [earlier, later]


async def test_history_default_time_is_recent(
    history_storage: SemanticStorage,
):
    ids = [UUID(f"{prefix}0000000-0000-4000-8000-000000000000") for prefix in "fedcba0"]
    before = datetime.now(UTC) - timedelta(seconds=1)
    for history_id in ids:
        await history_storage.add_history_to_set("defaults", history_id)

    assert {
        history_id
        async for history_id in history_storage.get_history_messages(
            set_ids=["defaults"], is_ingested=False
        )
    } == set(ids)
    assert [
        sid async for sid in history_storage.get_history_set_ids(older_than=before)
    ] == []
    assert [
        sid
        async for sid in history_storage.get_history_set_ids(
            older_than=datetime.now(UTC)
        )
    ] == ["defaults"]


async def test_neo4j_history_query_orders_before_limit():
    driver = create_autospec(AsyncDriver, instance=True)
    driver.execute_query.return_value = ([], None, None)
    storage = Neo4jSemanticStorage(driver)

    assert [
        history_id
        async for history_id in storage.get_history_messages(
            set_ids=["ordered"], is_ingested=False, limit=5
        )
    ] == []
    query = driver.execute_query.call_args.args[0]
    assert (
        "ORDER BY coalesce(h.episode_created_at, h.created_at), h.sequence_num, h.created_at, h.history_id LIMIT $limit"
        in " ".join(query.text.split())
    )
    assert driver.execute_query.call_args.kwargs == {
        "set_ids": ["ordered"],
        "is_ingested": False,
        "limit": 5,
    }


async def test_neo4j_history_write_preserves_episode_time():
    driver = create_autospec(AsyncDriver, instance=True)
    storage = Neo4jSemanticStorage(driver)
    created_at = datetime(2025, 1, 1, 12, 34, 56, 123456, tzinfo=UTC)
    episode_id = UUID("550e8400-e29b-41d4-a716-446655440001")
    before = datetime.now(UTC)
    await storage.add_history_to_set("ordered", episode_id, created_at=created_at)
    after = datetime.now(UTC)
    kwargs = dict(driver.execute_query.call_args.kwargs)
    assert before <= kwargs.pop("created_at") <= after
    assert kwargs == {
        "set_id": "ordered",
        "history_id": str(episode_id),
        "episode_created_at": created_at,
        "sequence_num": 0,
    }
