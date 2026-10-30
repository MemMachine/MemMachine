"""Unit tests for service_locator helpers."""

from unittest.mock import NonCallableMagicMock, create_autospec

import pytest

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
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import (
    VectorStore,
    VectorStorePartition,
    VectorStoreSchema,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
    SegmentStorePartition,
)
from memmachine_server.episodic_memory.event_memory.segment_store.utils import (
    PARTITION_KEY_MAX_BYTES,
    validate_partition_key,
)
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    _event_params,
    event_backend_indexed_properties,
    event_backend_vector_store,
    event_memory_from_config,
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
    vector_store.open_or_create_partition.return_value = create_autospec(
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
    resource_manager.get_segment_store.return_value.open_or_create_partition.return_value = create_autospec(
        SegmentStorePartition, instance=True
    )
    resource_manager.get_embedder.return_value = embedder
    resource_manager.get_reranker.return_value = create_autospec(
        Reranker, instance=True
    )
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
    """The session's partition is opened in the deployment's event store.

    Opening creates the partition when it is absent; the vector store's
    registry arbitrates creation across processes, so a worker that loses
    the race to another opens the winner's partition instead of failing
    the request.
    """
    resource_manager = _deployment_resource_manager(_DEPLOYMENT)
    resource_manager.get_event_memory.return_value = await event_memory_from_config(
        resource_manager
    )
    config = EventLongTermMemoryConf(session_id="raced")

    params = await _event_params(config, resource_manager)

    vector_store = resource_manager.get_vector_store.return_value
    assert params.event_memory_partition.similarity_metric is SimilarityMetric.COSINE
    vector_store.open_or_create_partition.assert_awaited_once_with("raced")
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
    resource_manager.get_event_memory.return_value = await event_memory_from_config(
        resource_manager
    )
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
async def test_event_params_serves_a_session_with_the_deployments_reranker():
    """The reranker is the deployment's whatever a saved session names."""
    deployment = _DEPLOYMENT.model_copy(update={"reranker": "deployment_reranker"})
    resource_manager = _deployment_resource_manager(deployment)
    resource_manager.get_event_memory.return_value = await event_memory_from_config(
        resource_manager
    )
    resource_manager.get_reranker.reset_mock()
    config = EventLongTermMemoryConf.model_validate(
        {"session_id": "s", "reranker": "session_reranker"}
    )

    params = await _event_params(config, resource_manager)

    assert params.reranker is resource_manager.get_reranker.return_value
    resource_manager.get_reranker.assert_awaited_once_with(
        "deployment_reranker", validate=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    [
        None,
        LongTermMemoryConfPartial(
            backend="declarative", vector_graph_store="g", embedder="e"
        ),
        LongTermMemoryConfPartial(segment_store="ss", embedder="e"),
        LongTermMemoryConfPartial(vector_store="vs", embedder="e"),
        LongTermMemoryConfPartial(vector_store="vs", segment_store="ss"),
    ],
    ids=[
        "no_long_term_memory",
        "declarative_memory",
        "no_vector_store",
        "no_segment_store",
        "no_embedder",
    ],
)
async def test_event_memory_from_config_refuses_a_deployment_without_event_memory(
    deployment: LongTermMemoryConfPartial | None,
):
    resource_manager = _deployment_resource_manager(deployment)

    with pytest.raises(ConfigurationError, match="no event memory"):
        await event_memory_from_config(resource_manager)

    resource_manager.get_vector_store.assert_not_awaited()
    resource_manager.get_segment_store.assert_not_awaited()
