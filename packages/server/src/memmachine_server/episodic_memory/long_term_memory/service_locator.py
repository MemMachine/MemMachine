"""Helpers for building long-term memory from configuration."""

import hashlib
import re

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
from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PropertyValue,
)
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import VectorStoreCollectionConfig
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionConfig,
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

_EVENT_BACKEND_NAMESPACE = "long_term_memory"

_PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")
_PARTITION_KEY_MAX_LEN = 32


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
    vector_store = await resource_manager.get_vector_store(config.vector_store)
    segment_store = await resource_manager.get_segment_store(config.segment_store)
    embedder = await resource_manager.get_embedder(config.embedder, validate=True)
    reranker = (
        await resource_manager.get_reranker(config.reranker, validate=True)
        if config.reranker is not None
        else None
    )
    episode_storage = await resource_manager.get_episode_storage()

    partition_key = partition_key_for_session(config.session_id)

    # Open the existing collection if any (preserves the original schema). Only
    # create with our merged schema if the partition does not yet exist.
    collection = await vector_store.open_collection(
        namespace=_EVENT_BACKEND_NAMESPACE,
        name=partition_key,
    )
    if collection is None:
        user_schema = _resolve_user_properties_schema(config.properties_schema)
        collection_config = VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            indexed_properties_schema={
                **EventMemory.expected_vector_store_collection_schema(),
                **EVENT_BACKEND_SYSTEM_FIELDS,
                **user_schema,
            },
        )
        await vector_store.create_collection(
            namespace=_EVENT_BACKEND_NAMESPACE,
            name=partition_key,
            config=collection_config,
        )
        collection = await vector_store.open_collection(
            namespace=_EVENT_BACKEND_NAMESPACE,
            name=partition_key,
        )
        if collection is None:
            raise RuntimeError(
                f"Failed to open vector store collection after creation for "
                f"partition {partition_key!r}"
            )

    partition = await segment_store.open_or_create_partition(
        partition_key,
        SegmentStorePartitionConfig(),
    )

    segmenter = _build_segmenter(config.segmenter)
    deriver = _build_deriver(config.deriver)

    return EventBackendParams(
        session_id=config.session_id,
        vector_store=vector_store,
        vector_store_collection=collection,
        vector_store_collection_namespace=_EVENT_BACKEND_NAMESPACE,
        segment_store=segment_store,
        segment_store_partition=partition,
        partition_key=partition_key,
        episode_storage=episode_storage,
        embedder=embedder,
        reranker=reranker,
        segmenter=segmenter,
        deriver=deriver,
    )


def partition_key_for_session(session_id: str) -> str:
    """
    Derive a partition key matching `[a-z0-9_]+` (≤32 chars) from a session id.

    If the session_id already satisfies the constraint, use it directly to keep
    debug paths legible. Otherwise hash to a stable 32-char hex digest.
    """
    if (
        _PARTITION_KEY_RE.match(session_id)
        and len(session_id) <= _PARTITION_KEY_MAX_LEN
    ):
        return session_id
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[
        :_PARTITION_KEY_MAX_LEN
    ]


def _resolve_user_properties_schema(
    raw: dict[str, str],
) -> dict[str, type[PropertyValue]]:
    resolved: dict[str, type[PropertyValue]] = {}
    for key, type_name in raw.items():
        prop_type = PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE.get(type_name)
        if prop_type is None:
            raise ValueError(f"Property {key!r}: unknown type name {type_name!r}")
        resolved[key] = prop_type
    return resolved


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
