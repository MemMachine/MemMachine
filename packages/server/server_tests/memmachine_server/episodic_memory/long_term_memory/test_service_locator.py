"""Unit tests for service_locator helpers."""

from unittest.mock import NonCallableMagicMock, create_autospec
from uuid import uuid4

import pytest

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.configuration.episodic_config import (
    DeclarativeLongTermMemoryConf,
    EpisodicMemoryConfPartial,
    EventLongTermMemoryConf,
    LongTermMemoryConfPartial,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeStorage
from memmachine_server.common.errors import ConfigurationError
from memmachine_server.common.partition_registry import (
    PartitionRegistry,
    Registration,
    validate_partition_name,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_graph_store import VectorGraphStore
from memmachine_server.common.vector_store import (
    VectorStore,
    VectorStorePartition,
    VectorStoreSchema,
)
from memmachine_server.episodic_memory.declarative_memory.data_types import (
    Episode as DeclarativeMemoryEpisode,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemoryPartition,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
    SegmentStorePartition,
)
from memmachine_server.episodic_memory.long_term_memory import service_locator
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    delete_long_term_memory_storage,
    event_backend_indexed_properties,
    event_backend_vector_store,
    long_term_memory_from_config,
    partition_key_for_session,
)


def test_a_session_id_the_name_rule_accepts_is_the_partition_name():
    for session_id in ("abc_123", "Session-Mixed-Case-123", "org/project", "日本語"):
        assert partition_key_for_session(session_id) == session_id


@pytest.mark.parametrize(
    "session_id", ["", "nul\x00", "a" * 256], ids=["empty", "nul", "too_long"]
)
def test_a_session_id_the_name_rule_refuses_is_hashed(session_id: str):
    name = partition_key_for_session(session_id)

    assert name != session_id
    validate_partition_name(name)
    assert partition_key_for_session(session_id) == name


def test_distinct_hashed_session_ids_name_distinct_partitions():
    assert partition_key_for_session("a" * 256) != partition_key_for_session("b" * 256)


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
                partition_registry="pr",
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
    vector_store.store_schema = VectorStoreSchema(
        vector_dimensions=3,
        similarity_metric=SimilarityMetric.COSINE,
        indexed_properties=event_backend_indexed_properties(),
    )
    vector_store.open_partition_incarnation.return_value = create_autospec(
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
    resource_manager.get_segment_store.return_value.open_partition_incarnation.return_value = create_autospec(
        SegmentStorePartition, instance=True
    )
    registration = create_autospec(Registration, instance=True)
    registration.incarnation = uuid4()
    partition_registry = create_autospec(PartitionRegistry, instance=True)
    partition_registry.resolve.return_value = registration
    resource_manager.get_partition_registry.return_value = partition_registry
    resource_manager.get_embedder.return_value = embedder
    resource_manager.get_episode_storage.return_value = create_autospec(
        EpisodeStorage, instance=True
    )
    resource_manager.get_metrics_factory.return_value = None
    return resource_manager


_DEPLOYMENT = LongTermMemoryConfPartial(
    backend="event",
    vector_store="vs",
    segment_store="ss",
    partition_registry="pr",
    embedder="e",
)


@pytest.mark.asyncio
async def test_event_params_opens_the_session_partition_in_the_event_store():
    """The session's partition is opened in the deployment's event store.

    Opening creates the partition when it is absent; event memory's
    registry arbitrates creation across processes, so a worker that loses
    the race to another opens the winner's partition instead of failing
    the request.
    """
    resource_manager = _deployment_resource_manager(_DEPLOYMENT)
    config = EventLongTermMemoryConf(session_id="raced")

    long_term_memory = await long_term_memory_from_config(config, resource_manager)

    vector_store = resource_manager.get_vector_store.return_value
    partition = long_term_memory._memory
    assert isinstance(partition, EventMemoryPartition)
    assert partition.similarity_metric is SimilarityMetric.COSINE
    partition_registry = resource_manager.get_partition_registry.return_value
    partition_registry.resolve.assert_awaited_once_with("raced")
    vector_store.open_partition_incarnation.assert_awaited_once_with(
        partition_registry.resolve.return_value.incarnation
    )
    registry_call = resource_manager.get_partition_registry.await_args
    assert registry_call.args == ("pr",)
    assert registry_call.kwargs["partition_namespace"] == "event_memory"
    assert registry_call.kwargs["tombstone_retention_seconds"] == 86400
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

    await long_term_memory_from_config(config, resource_manager)

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
        await long_term_memory_from_config(config, resource_manager)

    resource_manager.get_vector_store.assert_not_awaited()
    resource_manager.get_segment_store.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleting_an_event_sessions_storage_deletes_its_partitions_without_opening_them():
    """A session's partitions are deleted by name, so deleting creates none."""
    resource_manager = _deployment_resource_manager(_DEPLOYMENT)
    config = EventLongTermMemoryConf(session_id="s")

    await delete_long_term_memory_storage(config, resource_manager)

    partition_registry = resource_manager.get_partition_registry.return_value
    vector_store = resource_manager.get_vector_store.return_value
    segment_store = resource_manager.get_segment_store.return_value
    partition_registry.unregister.assert_awaited_once_with("s")
    partition_registry.reserve.assert_not_awaited()
    vector_store.prepare_partition_incarnation.assert_not_awaited()
    segment_store.prepare_partition_incarnation.assert_not_awaited()
    # Reclamation is the purge rounds'; the delete path never purges.
    vector_store.purge_partition_incarnation.assert_not_awaited()
    segment_store.purge_partition_incarnation.assert_not_awaited()


@pytest.mark.asyncio
async def test_event_memory_without_a_registry_database_is_refused():
    resource_manager = _deployment_resource_manager(
        _DEPLOYMENT.model_copy(update={"partition_registry": None})
    )
    config = EventLongTermMemoryConf(session_id="s")

    with pytest.raises(ConfigurationError, match="partition_registry"):
        await long_term_memory_from_config(config, resource_manager)


@pytest.mark.asyncio
async def test_deleting_a_declarative_sessions_storage_deletes_its_episodes(
    monkeypatch: pytest.MonkeyPatch,
):
    """The declarative backend deletes the episodes its memory of the session matches."""
    declarative_memory_cls = create_autospec(service_locator.DeclarativeMemory)
    monkeypatch.setattr(service_locator, "DeclarativeMemory", declarative_memory_cls)
    declarative_memory = declarative_memory_cls.return_value
    episodes = [
        create_autospec(DeclarativeMemoryEpisode, instance=True, uid=uid)
        for uid in ("a", "b")
    ]
    declarative_memory.get_matching_episodes.return_value = episodes
    resource_manager = create_autospec(CommonResourceManager, instance=True)
    resource_manager.get_vector_graph_store.return_value = create_autospec(
        VectorGraphStore, instance=True
    )
    resource_manager.get_embedder.return_value = create_autospec(
        Embedder, instance=True
    )
    resource_manager.get_reranker.return_value = create_autospec(
        Reranker, instance=True
    )
    config = DeclarativeLongTermMemoryConf(
        session_id="s", vector_graph_store="g", embedder="e", reranker="r"
    )

    await delete_long_term_memory_storage(config, resource_manager)

    (params,) = declarative_memory_cls.call_args.args
    assert params.session_id == "s"
    assert (
        params.vector_graph_store
        is resource_manager.get_vector_graph_store.return_value
    )
    declarative_memory.get_matching_episodes.assert_awaited_once_with()
    (deleted,) = declarative_memory.delete_episodes.await_args.args
    assert list(deleted) == ["a", "b"]
