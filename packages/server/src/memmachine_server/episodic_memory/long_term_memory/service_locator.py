"""Helpers for building long-term memory from configuration."""

import hashlib
import logging
from uuid import UUID, uuid5

from pydantic import InstanceOf

from memmachine_server.common.configuration.episodic_config import (
    DeclarativeLongTermMemoryConf,
    DeriverConf,
    EventLongTermMemoryConf,
    LongTermMemoryConf,
    LongTermMemoryConfPartial,
    PassthroughSegmenterConf,
    SegmenterConf,
    SentenceTextDeriverConf,
    TextSegmenterConf,
    WholeTextDeriverConf,
)
from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.errors import ConfigurationError
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import VectorStore
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.utils import (
    PARTITION_KEY_MAX_BYTES,
    validate_partition_key,
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
    DeclarativeBackendParams,
    EventBackendParams,
    LongTermMemoryParams,
)

logger = logging.getLogger(__name__)

_EVENT_BACKEND_VECTOR_STORE_NAMESPACE = UUID("d545833c-59ce-46ee-a325-c1a6222159a8")
"""The UUIDv5 namespace of the event backend's vector store names.

The event backend keeps one store per vector store backend and embedder. Its
name is the UUIDv5 of the embedder id in the UUIDv5 of the backend's key in
this namespace, in hex: a vector store name whatever the key and the id are,
never another backend's store name, so stores whose registries share a
database stay apart, and never the name of another memory's store. Fixed,
because the name locates the store's data.
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
    # The deployment's EventMemory serves every session. A session's vector
    # store, segment store, and embedder hold its data, so they must be the
    # deployment's; the deployment's segmenter, deriver, and reranker serve
    # it whatever the session names.
    event_memory = await resource_manager.get_event_memory()
    deployment = resource_manager.config.episodic_memory.long_term_memory
    assert deployment is not None
    if (config.vector_store, config.segment_store, config.embedder) != (
        deployment.vector_store,
        deployment.segment_store,
        deployment.embedder,
    ):
        raise ConfigurationError(
            f"Session {config.session_id!r} names a vector store, segment "
            "store, or embedder other than the deployment's event memory's"
        )
    reranker = (
        await resource_manager.get_reranker(deployment.reranker, validate=True)
        if deployment.reranker is not None
        else None
    )

    partition_key = partition_key_for_session(config.session_id)
    event_memory_partition = await event_memory.open_or_create_partition(partition_key)

    return EventBackendParams(
        session_id=config.session_id,
        event_memory=event_memory,
        event_memory_partition=event_memory_partition,
        partition_key=partition_key,
        episode_storage=await resource_manager.get_episode_storage(),
        reranker=reranker,
    )


async def event_memory_from_config(
    config: LongTermMemoryConfPartial | None,
    resource_manager: InstanceOf[CommonResourceManager],
) -> EventMemory:
    """Build a deployment's EventMemory from its long-term memory configuration.

    The configuration names every resource the EventMemory works with.

    Raises:
        ConfigurationError:
            If the configuration names no vector store, segment store, or
            embedder.
    """
    if (
        config is None
        or config.vector_store is None
        or config.segment_store is None
        or config.embedder is None
    ):
        raise ConfigurationError(
            "Event memory needs the long-term memory configuration to name "
            "a vector store, a segment store, and an embedder"
        )
    reranker = (
        await resource_manager.get_reranker(config.reranker, validate=True)
        if config.reranker is not None
        else None
    )
    return EventMemory(
        EventMemoryParams(
            segment_store=await resource_manager.get_segment_store(
                config.segment_store
            ),
            vector_store=await event_backend_vector_store(
                resource_manager,
                vector_store=config.vector_store,
                embedder=config.embedder,
            ),
            segmenter=_build_segmenter(
                config.segmenter
                if config.segmenter is not None
                else PassthroughSegmenterConf()
            ),
            deriver=_build_deriver(
                config.deriver if config.deriver is not None else WholeTextDeriverConf()
            ),
            embedder=await resource_manager.get_embedder(
                config.embedder, validate=True
            ),
            reranker=reranker,
            metrics_factory=await resource_manager.get_metrics_factory("prometheus"),
        )
    )


async def event_backend_vector_store(
    resource_manager: InstanceOf[CommonResourceManager],
    *,
    vector_store: str,
    embedder: str,
) -> VectorStore:
    """The event backend's vector store: the store of its backend and embedder, built for it."""
    embedder_instance = await resource_manager.get_embedder(embedder, validate=True)
    return await resource_manager.get_vector_store(
        vector_store,
        vector_store_name=uuid5(
            uuid5(_EVENT_BACKEND_VECTOR_STORE_NAMESPACE, vector_store),
            embedder,
        ).hex,
        vector_dimensions=embedder_instance.dimensions,
        similarity_metric=embedder_instance.similarity_metric,
        indexed_properties=event_backend_indexed_properties(),
    )


def event_backend_indexed_properties() -> dict[str, PropertyType]:
    """The system keys the event backend declares for its vector records.

    EventMemory's reserved keys and the adapter's own event fields; the
    vector store is built with these.
    """
    return {
        **EventMemory.expected_vector_store_collection_schema(),
        **EVENT_BACKEND_SYSTEM_FIELDS,
    }


def partition_key_for_session(session_id: str) -> str:
    """
    Derive a partition key satisfying the segment store's contract.

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
