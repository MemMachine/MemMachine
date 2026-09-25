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
    PassthroughSegmenterConf,
    SegmenterConf,
    SentenceTextDeriverConf,
    TextSegmenterConf,
    WholeTextDeriverConf,
)
from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import VectorStore
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionConfig,
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

The event backend keeps one store per embedder, named by the UUIDv5 of the
embedder id in this namespace, in hex: a vector store name whatever the id
is, and never the name of another memory's store of the same embedder.
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


class SessionPartitionMissingError(RuntimeError):
    """A session's row exists, but a store holds no partition under its key."""

    def __init__(self, session_id: str, partition_key: str, store: str) -> None:
        """Initialize with the session, its partition key, and the store lacking it."""
        self.session_id = session_id
        self.partition_key = partition_key
        super().__init__(
            f"Session {session_id!r} has no partition {partition_key!r} in {store}: "
            "its storage was never created, or has been deleted"
        )


async def create_event_backend_partitions(
    config: EventLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> None:
    """Create the session's partition in its segment store and its vector store.

    Strict, like the stores' own `create_partition`: called once, when the
    session is created.
    """
    segment_store = await resource_manager.get_segment_store(config.segment_store)
    vector_store = await event_backend_vector_store(config, resource_manager)
    partition_key = partition_key_for_session(config.session_id)
    await segment_store.create_partition(partition_key, SegmentStorePartitionConfig())
    await vector_store.create_partition(partition_key)


async def delete_event_backend_partitions(
    config: EventLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> None:
    """Delete the session's partitions by key, whether or not both exist."""
    segment_store = await resource_manager.get_segment_store(config.segment_store)
    vector_store = await event_backend_vector_store(config, resource_manager)
    partition_key = partition_key_for_session(config.session_id)
    await vector_store.delete_partition(partition_key)
    await segment_store.delete_partition(partition_key)


async def _event_params(
    config: EventLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> EventBackendParams:
    segment_store = await resource_manager.get_segment_store(config.segment_store)
    embedder = await resource_manager.get_embedder(config.embedder, validate=True)
    vector_store = await event_backend_vector_store(config, resource_manager)
    reranker = (
        await resource_manager.get_reranker(config.reranker, validate=True)
        if config.reranker is not None
        else None
    )
    episode_storage = await resource_manager.get_episode_storage()

    partition_key = partition_key_for_session(config.session_id)

    # No memory request creates storage: the session's partitions were created
    # with the session, and a session without them is broken, not new.
    vector_store_partition = await vector_store.get_partition(partition_key)
    if vector_store_partition is None:
        raise SessionPartitionMissingError(
            config.session_id,
            partition_key,
            f"vector store {vector_store.vector_store_name!r}",
        )
    partition = await segment_store.get_partition(partition_key)
    if partition is None:
        raise SessionPartitionMissingError(
            config.session_id, partition_key, "the segment store"
        )

    segmenter = _build_segmenter(config.segmenter)
    deriver = _build_deriver(config.deriver)

    return EventBackendParams(
        session_id=config.session_id,
        vector_store=vector_store,
        vector_store_partition=vector_store_partition,
        segment_store=segment_store,
        segment_store_partition=partition,
        partition_key=partition_key,
        episode_storage=episode_storage,
        embedder=embedder,
        reranker=reranker,
        segmenter=segmenter,
        deriver=deriver,
        metrics_factory=await resource_manager.get_metrics_factory("prometheus"),
    )


async def event_backend_vector_store(
    config: EventLongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> VectorStore:
    """The event backend's vector store: the store of its embedder, built for it."""
    embedder = await resource_manager.get_embedder(config.embedder, validate=True)
    return await resource_manager.get_vector_store(
        config.vector_store,
        vector_store_name=uuid5(
            _EVENT_BACKEND_VECTOR_STORE_NAMESPACE, config.embedder
        ).hex,
        vector_dimensions=embedder.dimensions,
        similarity_metric=embedder.similarity_metric,
        indexed_properties=event_backend_indexed_properties(),
    )


def event_backend_indexed_properties() -> dict[str, PropertyType]:
    """The system keys the event backend writes into every vector record.

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
