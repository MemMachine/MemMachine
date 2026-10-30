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
    SegmenterConf,
    SentenceTextDeriverConf,
    WholeTextDeriverConf,
)
from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.errors import ConfigurationError
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import VectorStore, VectorStoreSchema
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.utils import (
    PARTITION_KEY_MAX_BYTES,
    validate_partition_key,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)

from .long_term_memory import (
    EVENT_BACKEND_SYSTEM_FIELDS,
    DeclarativeBackendParams,
    EventBackendParams,
    LongTermMemoryParams,
)

logger = logging.getLogger(__name__)

_EVENT_MEMORY_VECTOR_STORE_NAME = "event_memory"
"""The name of event memory's one vector store, whatever its backend and embedder.

Fixed, because the name locates the store's data.
"""


async def long_term_memory_params_from_config(
    config: LongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> LongTermMemoryParams:
    """Build LongTermMemoryParams from configuration and resources."""
    match config:
        case DeclarativeLongTermMemoryConf():
            return await _declarative_params(config, resource_manager)
        case EventLongTermMemoryConf():
            return await _event_params(config, resource_manager)
        case _:
            raise NotImplementedError(
                f"Unsupported long-term memory backend: {type(config).__name__}"
            )


async def _declarative_params(
    config: DeclarativeLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> DeclarativeBackendParams:
    vector_graph_store = await resource_manager.get_vector_graph_store(
        config.vector_graph_store,
    )
    embedder = await resource_manager.get_embedder(config.embedder, validate=True)
    reranker = await resource_manager.get_reranker(config.reranker, validate=True)
    return DeclarativeBackendParams(
        session_id=config.session_id,
        vector_graph_store=vector_graph_store,
        embedder=embedder,
        reranker=reranker,
        message_sentence_chunking=config.message_sentence_chunking,
    )


async def _event_params(
    config: EventLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> EventBackendParams:
    # The deployment's EventMemory serves every session, with the
    # deployment's segmenter, deriver, and reranker.
    event_memory = await resource_manager.get_event_memory()
    deployment = resource_manager.config.episodic_memory.long_term_memory
    assert deployment is not None
    reranker = (
        await resource_manager.get_reranker(deployment.reranker, validate=True)
        if deployment.reranker is not None
        else None
    )
    episode_storage = await resource_manager.get_episode_storage()

    partition_key = partition_key_for_session(config.session_id)
    event_memory_partition = await event_memory.open_or_create_partition(partition_key)

    return EventBackendParams(
        session_id=config.session_id,
        event_memory=event_memory,
        event_memory_partition=event_memory_partition,
        partition_key=partition_key,
        episode_storage=episode_storage,
        reranker=reranker,
    )


def deployment_event_memory_ids(configuration: Configuration) -> tuple[str, str, str]:
    """
    The deployment's event memory vector store, event memory store, and embedder ids.

    Every session's event memory uses them.

    Raises:
        ConfigurationError: If the deployment configures no event memory.

    """
    deployment = configuration.episodic_memory.long_term_memory
    if (
        deployment is None
        or deployment.vector_store is None
        or deployment.event_memory_store is None
        or deployment.embedder is None
    ):
        raise ConfigurationError("The deployment configures no event memory")
    return deployment.vector_store, deployment.event_memory_store, deployment.embedder


async def event_memory_from_config(
    resource_manager: InstanceOf[CommonResourceManager],
) -> EventMemory:
    """
    Build the deployment's EventMemory from its long-term memory configuration.

    Raises:
        ConfigurationError: If the deployment configures no event memory.

    """
    _, event_memory_store_id, embedder_id = deployment_event_memory_ids(
        resource_manager.config
    )
    deployment = resource_manager.config.episodic_memory.long_term_memory
    assert deployment is not None
    embedder = await resource_manager.get_embedder(embedder_id, validate=True)
    return EventMemory(
        EventMemoryParams(
            event_memory_store=await resource_manager.get_event_memory_store(
                event_memory_store_id
            ),
            vector_store=await event_backend_vector_store(resource_manager),
            store_schema=event_backend_store_schema(embedder),
            segmenter=_build_segmenter(deployment.segmenter),
            deriver=_build_deriver(
                deployment.deriver
                if deployment.deriver is not None
                else WholeTextDeriverConf()
            ),
            embedder=embedder,
            metrics_factory=await resource_manager.get_metrics_factory("prometheus"),
        )
    )


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
        store_schema=event_backend_store_schema(embedder),
    )


def event_backend_store_schema(embedder: Embedder) -> VectorStoreSchema:
    """The schema of the deployment's event memory vector store, built for its embedder."""
    return VectorStoreSchema(
        vector_dimensions=embedder.dimensions,
        similarity_metric=embedder.similarity_metric,
        indexed_properties=event_backend_indexed_properties(),
    )


def event_backend_indexed_properties() -> dict[str, PropertyType]:
    """The system keys the event backend declares for its vector records."""
    return {
        **EventMemory.expected_vector_store_indexed_properties(),
        **EVENT_BACKEND_SYSTEM_FIELDS,
    }


def partition_key_for_session(session_id: str) -> str:
    """
    Derive a partition key satisfying the event memory store's contract.

    If the session_id already satisfies it (checked by the store's own
    validator, so the two can never disagree), use it directly to keep
    debug paths legible. Otherwise hash to a stable 32-char hex digest and
    emit a DEBUG log of the original→hashed mapping so operators can
    correlate partition keys back to sessions during incident response.
    """
    try:
        validate_partition_key(session_id)
    except ValueError:
        pass
    else:
        return session_id
    partition_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[
        :PARTITION_KEY_MAX_BYTES
    ]
    logger.debug(
        "partition_key_for_session: hashed session_id %r -> partition_key %r",
        session_id,
        partition_key,
    )
    return partition_key


def _build_segmenter(conf: SegmenterConf | None) -> Segmenter:
    # No configured handler: the table's own fallback, one segment per
    # block, unchanged.
    if conf is None:
        return Segmenter()
    return Segmenter([TextSegmenter(max_chunk_length=conf.max_chunk_length)])


def _build_deriver(conf: DeriverConf) -> Deriver:
    match conf:
        case WholeTextDeriverConf():
            return Deriver([WholeTextDeriver()])
        case SentenceTextDeriverConf():
            return Deriver([SentenceTextDeriver()])
        case _:
            raise NotImplementedError(
                f"Unsupported deriver config: {type(conf).__name__}"
            )
