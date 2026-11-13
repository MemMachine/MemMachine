"""Helpers for building long-term memory from configuration."""

import hashlib
import logging

from pydantic import InstanceOf

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.configuration.episodic_config import (
    DeclarativeLongTermMemoryConf,
    DeriverConf,
    EventLongTermMemoryConf,
    LongTermMemoryConf,
    PassthroughSegmenterConf,
    SegmenterConf,
    SentenceTextDeriverConf,
    TextSegmenterConf,
    WholeTextDeriverConf,
)
from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.errors import ConfigurationError
from memmachine_server.common.partition_registry import (
    PartitionRegistry,
    validate_partition_name,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import VectorStore, VectorStoreSchema
from memmachine_server.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStore,
    EventMemoryStorePartition,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import (
    PassthroughSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)

from .long_term_memory import (
    EVENT_BACKEND_SYSTEM_FIELDS,
    LongTermMemory,
)

logger = logging.getLogger(__name__)

_EVENT_MEMORY_VECTOR_STORE_NAME = "event_memory"
"""The name of event memory's one vector store, whatever its backend and embedder.

Fixed, because the name locates the store's data.
"""


async def long_term_memory_from_config(
    config: LongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> LongTermMemory:
    """Build a session's LongTermMemory from its configuration and the deployment's resources."""
    match config:
        case DeclarativeLongTermMemoryConf():
            return LongTermMemory(await _declarative_memory(config, resource_manager))
        case EventLongTermMemoryConf():
            reranker = (
                await resource_manager.get_reranker(config.reranker, validate=True)
                if config.reranker is not None
                else None
            )
            event_memory = await _event_memory(
                config, resource_manager, reranker=reranker
            )
            return LongTermMemory(
                await event_memory.get_or_create_partition(
                    partition_key_for_session(config.session_id)
                ),
                episode_storage=await resource_manager.get_episode_storage(),
                reranker=reranker,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported long-term memory backend: {type(config).__name__}"
            )


async def delete_long_term_memory_storage(
    config: LongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> None:
    """
    Delete a session's long-term memory storage.

    The event backend deletes the session's partitions by key, getting
    neither, so storage the session never created stays uncreated.
    """
    match config:
        case DeclarativeLongTermMemoryConf():
            declarative_memory = await _declarative_memory(config, resource_manager)
            episodes = await declarative_memory.get_matching_episodes()
            await declarative_memory.delete_episodes(
                episode.uid for episode in episodes
            )
        case EventLongTermMemoryConf():
            # Deleting a partition uses no reranker.
            event_memory = await _event_memory(config, resource_manager, reranker=None)
            await event_memory.delete_partition(
                partition_key_for_session(config.session_id)
            )
        case _:
            raise NotImplementedError(
                f"Unsupported long-term memory backend: {type(config).__name__}"
            )


async def _declarative_memory(
    config: DeclarativeLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> DeclarativeMemory:
    return DeclarativeMemory(
        DeclarativeMemoryParams(
            session_id=config.session_id,
            vector_graph_store=await resource_manager.get_vector_graph_store(
                config.vector_graph_store,
            ),
            embedder=await resource_manager.get_embedder(
                config.embedder, validate=True
            ),
            reranker=await resource_manager.get_reranker(
                config.reranker, validate=True
            ),
            message_sentence_chunking=config.message_sentence_chunking,
        ),
    )


async def _event_memory(
    config: EventLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
    *,
    reranker: Reranker | None,
) -> EventMemory:
    """An EventMemory on the deployment's stores and embedder, with the session's segmenter and deriver."""
    _, segment_store_id, embedder_id = deployment_event_memory_ids(
        resource_manager.config
    )
    embedder = await resource_manager.get_embedder(embedder_id, validate=True)
    event_memory_store = EventMemoryStore(
        segment_store=await resource_manager.get_segment_store(segment_store_id),
        vector_store=await event_backend_vector_store(resource_manager),
    )
    return EventMemory(
        EventMemoryParams(
            partition_registry=await _event_memory_partition_registry(
                resource_manager, event_memory_store
            ),
            event_memory_store=event_memory_store,
            segmenter=_build_segmenter(config.segmenter),
            deriver=_build_deriver(config.deriver),
            embedder=embedder,
            reranker=reranker,
            metrics_factory=await resource_manager.get_metrics_factory("prometheus"),
        )
    )


async def _event_memory_partition_registry(
    resource_manager: InstanceOf[CommonResourceManager],
    event_memory_store: EventMemoryStore,
) -> PartitionRegistry[EventMemoryStorePartition]:
    """
    The registry of event memory's partition namespace, over event memory's stores, whose purge rounds run in the background.

    Raises:
        ConfigurationError: If the deployment names no registry database
            for event memory.

    """
    deployment = resource_manager.config.episodic_memory.long_term_memory
    if deployment is None or deployment.partition_registry is None:
        raise ConfigurationError(
            "The deployment's event memory names no partition_registry database"
        )
    return await resource_manager.get_partition_registry(
        deployment.partition_registry,
        partition_namespace=EventMemory.PARTITION_NAMESPACE,
        tombstone_retention_seconds=deployment.tombstone_retention_seconds,
        partitioned_store=event_memory_store,
    )


def deployment_event_memory_ids(configuration: Configuration) -> tuple[str, str, str]:
    """
    The deployment's event memory vector store, segment store, and embedder ids.

    Every session's event memory uses them.

    Raises:
        ConfigurationError: If the deployment configures no event memory.

    """
    deployment = configuration.episodic_memory.long_term_memory
    if (
        deployment is None
        or deployment.vector_store is None
        or deployment.segment_store is None
        or deployment.embedder is None
    ):
        raise ConfigurationError("The deployment configures no event memory")
    return deployment.vector_store, deployment.segment_store, deployment.embedder


async def event_backend_vector_store(
    resource_manager: InstanceOf[CommonResourceManager],
) -> VectorStore:
    """The deployment's event memory vector store, built for its embedder."""
    vector_store_id, _, embedder_id = deployment_event_memory_ids(
        resource_manager.config
    )
    embedder = await resource_manager.get_embedder(embedder_id, validate=True)
    return await resource_manager.get_vector_store(
        vector_store_id,
        vector_store_name=_EVENT_MEMORY_VECTOR_STORE_NAME,
        store_schema=VectorStoreSchema(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            indexed_properties=event_backend_indexed_properties(),
        ),
    )


def event_backend_indexed_properties() -> dict[str, PropertyType]:
    """The system keys the event backend declares for its vector records."""
    return {
        **EventMemory.expected_vector_store_indexed_properties(),
        **EVENT_BACKEND_SYSTEM_FIELDS,
    }


def partition_key_for_session(session_id: str) -> str:
    """
    Derive the name of a session's event memory partition.

    A session id the partition registry's name rule accepts is the name as it
    is, which keeps debug paths legible. Any other is hashed to a stable hex
    digest, with a DEBUG log of the original-to-hashed mapping so operators
    can correlate partition names back to sessions during incident response.
    """
    try:
        validate_partition_name(session_id)
    except ValueError:
        pass
    else:
        return session_id
    partition_name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    logger.debug(
        "partition_key_for_session: hashed session_id %r -> partition name %r",
        session_id,
        partition_name,
    )
    return partition_name


def _build_segmenter(conf: SegmenterConf) -> Segmenter:
    match conf:
        case PassthroughSegmenterConf():
            return PassthroughSegmenter()
        case TextSegmenterConf(max_chunk_length=max_chunk_length):
            return TextSegmenter(max_chunk_length=max_chunk_length)
        case _:
            raise NotImplementedError(
                f"Unsupported segmenter config: {type(conf).__name__}"
            )


def _build_deriver(conf: DeriverConf) -> Deriver:
    match conf:
        case WholeTextDeriverConf():
            return WholeTextDeriver()
        case SentenceTextDeriverConf():
            return SentenceTextDeriver()
        case _:
            raise NotImplementedError(
                f"Unsupported deriver config: {type(conf).__name__}"
            )
