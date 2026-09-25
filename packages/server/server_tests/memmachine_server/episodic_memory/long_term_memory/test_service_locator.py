"""Unit tests for service_locator helpers."""

import re
from unittest.mock import create_autospec

import pytest

from memmachine_server.common.configuration.episodic_config import (
    EventLongTermMemoryConf,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeStorage
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import (
    VectorStore,
    VectorStorePartition,
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
async def test_any_embedder_id_names_a_vector_store():
    """The store's name derives from the embedder id, so the id itself is unconstrained."""
    config = EventLongTermMemoryConf(
        session_id="s",
        vector_store="vs",
        segment_store="ss",
        embedder="OpenAI text-embedding-3-large, 3072 dimensions",
    )
    embedder = create_autospec(Embedder, instance=True)
    embedder.dimensions = 3
    embedder.similarity_metric = SimilarityMetric.COSINE
    resource_manager = create_autospec(CommonResourceManager, instance=True)
    resource_manager.get_embedder.return_value = embedder

    await event_backend_vector_store(config, resource_manager)

    vector_store_name = resource_manager.get_vector_store.await_args.kwargs[
        "vector_store_name"
    ]
    assert re.fullmatch(r"[0-9a-f]{32}", vector_store_name)


@pytest.mark.asyncio
async def test_event_params_opens_the_session_partition_of_the_embedders_collection():
    """The store is the embedder's collection, and the session's partition is opened in it.

    Opening creates the partition when it is absent; the vector store's
    registry arbitrates creation across processes, so a worker that loses
    the race to another opens the winner's partition instead of failing
    the request.
    """
    config = EventLongTermMemoryConf(
        session_id="raced", vector_store="vs", segment_store="ss", embedder="e"
    )
    partition = create_autospec(VectorStorePartition, instance=True)
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.open_or_create_partition.return_value = partition
    embedder = create_autospec(Embedder, instance=True)
    embedder.dimensions = 3
    embedder.similarity_metric = SimilarityMetric.COSINE
    resource_manager = create_autospec(CommonResourceManager, instance=True)
    resource_manager.get_vector_store.return_value = vector_store
    resource_manager.get_segment_store.return_value = create_autospec(
        SegmentStore, instance=True
    )
    resource_manager.get_segment_store.return_value.open_or_create_partition.return_value = create_autospec(
        SegmentStorePartition, instance=True
    )
    resource_manager.get_embedder.return_value = embedder
    resource_manager.get_episode_storage.return_value = create_autospec(
        EpisodeStorage, instance=True
    )
    resource_manager.get_metrics_factory.return_value = None

    params = await _event_params(config, resource_manager)

    assert params.vector_store_partition is partition
    vector_store.open_or_create_partition.assert_awaited_once_with("raced")
    resource_manager.get_vector_store.assert_awaited_once_with(
        "vs",
        vector_store_name="9f2137ad9dd259d3b15516148bd5d7cc",
        vector_dimensions=3,
        similarity_metric=SimilarityMetric.COSINE,
        indexed_properties=event_backend_indexed_properties(),
    )
