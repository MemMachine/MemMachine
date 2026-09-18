"""Unit tests for service_locator helpers."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.configuration.episodic_config import (
    EventLongTermMemoryConf,
)
from memmachine_server.common.episode_store import EpisodeStorage
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.resource_manager.database_manager import (
    enable_sqlite_foreign_keys,
)
from memmachine_server.common.vector_store.sqlite_vector_store import (
    SQLiteVectorStore,
    SQLiteVectorStoreParams,
)
from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
    USearchVectorSearchEngine,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionAlreadyExistsError,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.utils import (
    PARTITION_KEY_MAX_BYTES,
    validate_partition_key,
)
from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
    EventBackendParams,
)
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    SessionPartitionMissingError,
    create_event_backend_partitions,
    delete_event_backend_partitions,
    event_backend_collection,
    event_backend_indexed_properties,
    long_term_memory_params_from_config,
    partition_key_for_session,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import FakeEmbedder


def _is_valid_partition_key(value: str) -> bool:
    try:
        validate_partition_key(value)
    except ValueError:
        return False
    return True


def test_partition_key_passes_through_when_already_valid():
    assert partition_key_for_session("abc_123") == "abc_123"
    assert partition_key_for_session("session_42") == "session_42"


def test_partition_key_hashes_when_session_id_invalid():
    # Hyphens, uppercase, and other non-`[a-z0-9_]` chars trigger hashing.
    key = partition_key_for_session("Session-Mixed-Case-123")
    assert key != "Session-Mixed-Case-123"
    assert _is_valid_partition_key(key)
    assert len(key) == PARTITION_KEY_MAX_BYTES


def test_partition_key_hashes_when_too_long():
    long_id = "a" * 64
    key = partition_key_for_session(long_id)
    assert _is_valid_partition_key(key)
    assert len(key) == PARTITION_KEY_MAX_BYTES
    assert key != long_id


def test_partition_key_hashes_trailing_newline():
    """`$` matches before a trailing newline; the store's validator decides."""
    key = partition_key_for_session("abc\n")
    assert key != "abc\n"
    assert _is_valid_partition_key(key)


def test_partition_key_is_deterministic():
    """Same session_id always produces the same partition_key."""
    sid = "abc-123-uuid-shaped"
    assert partition_key_for_session(sid) == partition_key_for_session(sid)


def test_partition_key_distinct_inputs_produce_distinct_outputs():
    a = partition_key_for_session("abc-123-different-input-1")
    b = partition_key_for_session("abc-123-different-input-2")
    assert a != b


def test_partition_key_handles_unicode():
    # UTF-8 multi-byte input forces hashing because non-ASCII chars don't
    # match `[a-z0-9_]`.
    key = partition_key_for_session("日本語_セッション")
    assert _is_valid_partition_key(key)


def test_partition_key_empty_string_passthrough():
    """Empty session_id has length 0 but does not match `[a-z0-9_]+` (requires +)."""
    # The regex `^[a-z0-9_]+$` requires at least one char, so empty string
    # should be hashed (deterministic 32-hex digest).
    key = partition_key_for_session("")
    assert _is_valid_partition_key(key)
    assert len(key) == PARTITION_KEY_MAX_BYTES


# ===================================================================
# A session's partitions are created with the session, never by a request
# ===================================================================


@pytest_asyncio.fixture
async def stores(tmp_path):
    """A segment store and a vector store on one SQLite file, both started."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'stores.db'}")
    enable_sqlite_foreign_keys(engine)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()
    vector_store = SQLiteVectorStore(
        SQLiteVectorStoreParams(
            collection=event_backend_collection("fake"),
            vector_dimensions=2,
            indexed_properties=event_backend_indexed_properties(),
            sqlalchemy_engine=engine,
            vector_search_engine_factory=lambda ndim: USearchVectorSearchEngine(
                num_dimensions=ndim
            ),
        )
    )
    await vector_store.provision()
    await vector_store.startup()
    yield segment_store, vector_store
    await vector_store.shutdown()
    await engine.dispose()


@pytest.fixture
def resource_manager(stores):
    segment_store, vector_store = stores
    manager = MagicMock(spec=CommonResourceManager)
    manager.get_segment_store = AsyncMock(return_value=segment_store)
    manager.get_vector_store = AsyncMock(return_value=vector_store)
    manager.get_embedder = AsyncMock(return_value=FakeEmbedder())
    manager.get_episode_storage = AsyncMock(return_value=MagicMock(spec=EpisodeStorage))
    manager.get_metrics_factory = AsyncMock(return_value=None)
    return manager


_EVENT_CONF = EventLongTermMemoryConf(
    session_id="sess_1", vector_store="vs", segment_store="ss", embedder="fake"
)


@pytest.mark.asyncio
async def test_a_request_never_creates_a_partition(stores, resource_manager):
    segment_store, vector_store = stores

    with pytest.raises(SessionPartitionMissingError, match="sess_1"):
        await long_term_memory_params_from_config(_EVENT_CONF, resource_manager)

    key = partition_key_for_session("sess_1")
    assert await vector_store.get_partition(key) is None
    assert await segment_store.get_partition(key) is None


@pytest.mark.asyncio
async def test_partitions_created_with_the_session_are_what_a_request_binds(
    resource_manager,
):
    key = partition_key_for_session("sess_1")

    await create_event_backend_partitions(_EVENT_CONF, resource_manager)
    params = await long_term_memory_params_from_config(_EVENT_CONF, resource_manager)

    assert isinstance(params, EventBackendParams)
    assert params.partition_key == key
    assert params.vector_store_partition.partition_key == key
    assert params.segment_store_partition is not None
    # Strict: the session is created once.
    with pytest.raises(SegmentStorePartitionAlreadyExistsError):
        await create_event_backend_partitions(_EVENT_CONF, resource_manager)


@pytest.mark.asyncio
async def test_deleting_partitions_needs_none_to_exist(stores, resource_manager):
    segment_store, vector_store = stores
    key = partition_key_for_session("sess_1")

    await delete_event_backend_partitions(_EVENT_CONF, resource_manager)

    await create_event_backend_partitions(_EVENT_CONF, resource_manager)
    # Half-created storage, as a crash between the two creates would leave.
    await vector_store.delete_partition(key)
    await delete_event_backend_partitions(_EVENT_CONF, resource_manager)

    assert await vector_store.get_partition(key) is None
    assert await segment_store.get_partition(key) is None
