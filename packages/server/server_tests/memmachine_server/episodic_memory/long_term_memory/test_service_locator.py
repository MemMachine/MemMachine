"""Unit tests for service_locator helpers."""

from unittest.mock import NonCallableMagicMock, create_autospec

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    EventLongTermMemoryConf,
    LongTermMemoryConfPartial,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeStorage
from memmachine_server.common.errors import ConfigurationError
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.sql import (
    enable_sqlite_explicit_begin,
    enable_sqlite_foreign_keys,
)
from memmachine_server.common.vector_store import (
    VectorStore,
    VectorStorePartition,
    VectorStoreSchema,
)
from memmachine_server.common.vector_store.sqlite_vector_store import (
    SQLiteVectorStore,
    SQLiteVectorStoreParams,
    SQLiteVectorStorePartition,
)
from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
    USearchVectorSearchEngine,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
    SegmentStorePartition,
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
    _event_params,
    create_event_backend_partitions,
    delete_event_backend_partitions,
    event_backend_indexed_properties,
    event_backend_vector_store,
    long_term_memory_params_from_config,
    partition_key_for_session,
)


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


@pytest.mark.asyncio
async def test_every_backend_and_embedder_name_the_one_event_store():
    """The event store's name is fixed, whatever the backend and embedder."""
    names = set()
    for vector_store, embedder_id in (("vs_a", "e"), ("vs_b", "e"), ("vs_a", "f")):
        resource_manager = _deployment_resource_manager(
            LongTermMemoryConfPartial(
                backend="event",
                vector_store=vector_store,
                segment_store="ss",
                embedder=embedder_id,
            )
        )
        await event_backend_vector_store(resource_manager)
        names.add(
            resource_manager.get_vector_store.await_args.kwargs["vector_store_name"]
        )

    assert names == {"event_memory"}


def _deployment_resource_manager(
    deployment: LongTermMemoryConfPartial | None,
) -> NonCallableMagicMock:
    """A resource manager whose deployment configures `deployment` and whose stores open autospec partitions."""
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.get_partition.return_value = create_autospec(
        VectorStorePartition, instance=True
    )
    embedder = create_autospec(Embedder, instance=True)
    embedder.dimensions = 3
    embedder.similarity_metric = SimilarityMetric.COSINE
    resource_manager = create_autospec(CommonResourceManager, instance=True)
    resource_manager.config = create_autospec(Configuration, instance=True)
    resource_manager.config.episodic_memory = EpisodicMemoryConfPartial(
        long_term_memory=deployment
    )
    resource_manager.get_vector_store.return_value = vector_store
    resource_manager.get_segment_store.return_value = create_autospec(
        SegmentStore, instance=True
    )
    resource_manager.get_segment_store.return_value.get_partition.return_value = (
        create_autospec(SegmentStorePartition, instance=True)
    )
    resource_manager.get_embedder.return_value = embedder
    resource_manager.get_episode_storage.return_value = create_autospec(
        EpisodeStorage, instance=True
    )
    resource_manager.get_metrics_factory.return_value = None
    return resource_manager


_DEPLOYMENT = LongTermMemoryConfPartial(
    backend="event", vector_store="vs", segment_store="ss", embedder="e"
)


@pytest.mark.asyncio
async def test_event_params_opens_the_session_partition_in_the_event_store():
    """The session's partition is opened in the deployment's event store, never created."""
    resource_manager = _deployment_resource_manager(_DEPLOYMENT)
    config = EventLongTermMemoryConf(session_id="raced")

    params = await _event_params(config, resource_manager)

    vector_store = resource_manager.get_vector_store.return_value
    assert params.vector_store_partition is vector_store.get_partition.return_value
    vector_store.get_partition.assert_awaited_once_with("raced")
    vector_store.create_partition.assert_not_awaited()
    resource_manager.get_vector_store.assert_awaited_once_with(
        "vs",
        vector_store_name="event_memory",
        store_schema=VectorStoreSchema(
            vector_dimensions=3,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties=event_backend_indexed_properties(),
        ),
    )


@pytest.mark.asyncio
async def test_event_params_uses_the_deployments_stores_whatever_a_saved_session_names():
    """A session saved with its own stores and embedder is served from the deployment's."""
    resource_manager = _deployment_resource_manager(_DEPLOYMENT)
    config = EventLongTermMemoryConf.model_validate(
        {
            "session_id": "s",
            "vector_store": "other_vs",
            "segment_store": "other_ss",
            "embedder": "other_e",
        }
    )

    await _event_params(config, resource_manager)

    assert resource_manager.get_vector_store.await_args.args == ("vs",)
    resource_manager.get_segment_store.assert_awaited_once_with("ss")
    assert {call.args[0] for call in resource_manager.get_embedder.await_args_list} == {
        "e"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    [
        None,
        LongTermMemoryConfPartial(
            backend="declarative", vector_graph_store="g", embedder="e"
        ),
    ],
    ids=["no_long_term_memory", "declarative_memory"],
)
async def test_event_params_refuses_a_session_of_a_deployment_without_event_memory(
    deployment: LongTermMemoryConfPartial | None,
):
    resource_manager = _deployment_resource_manager(deployment)
    config = EventLongTermMemoryConf(session_id="s")

    with pytest.raises(ConfigurationError, match="no event memory"):
        await _event_params(config, resource_manager)

    resource_manager.get_vector_store.assert_not_awaited()
    resource_manager.get_segment_store.assert_not_awaited()


# ===================================================================
# A session's partitions are created with the session, never by a request
# ===================================================================


@pytest_asyncio.fixture
async def stores(tmp_path):
    """A segment store and the event memory's vector store on one SQLite file, both started."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'stores.db'}")
    enable_sqlite_explicit_begin(engine)
    enable_sqlite_foreign_keys(engine)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()
    vector_store = SQLiteVectorStore(
        SQLiteVectorStoreParams(
            vector_store_name="event_memory",
            store_schema=VectorStoreSchema(
                vector_dimensions=3,
                similarity_metric=SimilarityMetric.COSINE,
                indexed_properties=event_backend_indexed_properties(),
            ),
            sqlalchemy_engine=engine,
            vector_search_engine_factory=lambda ndim, metric: USearchVectorSearchEngine(
                num_dimensions=ndim, similarity_metric=metric
            ),
        )
    )
    await vector_store.startup()
    yield segment_store, vector_store
    await vector_store.shutdown()
    await engine.dispose()


@pytest.fixture
def resource_manager(stores):
    """The deployment's resource manager, serving the real stores."""
    segment_store, vector_store = stores
    manager = _deployment_resource_manager(_DEPLOYMENT)
    manager.get_segment_store.return_value = segment_store
    manager.get_vector_store.return_value = vector_store
    return manager


_EVENT_CONF = EventLongTermMemoryConf(session_id="sess_1")


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
    assert isinstance(params.vector_store_partition, SQLiteVectorStorePartition)
    assert params.vector_store_partition._partition_key == key
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
