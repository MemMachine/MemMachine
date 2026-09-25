"""Unit tests for service_locator helpers."""

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
    VectorStoreCollection,
    VectorStoreCollectionAlreadyExistsError,
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
    _EVENT_BACKEND_NAMESPACE,
    _event_params,
    _resolve_user_properties_schema,
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


def test_resolve_user_properties_schema_accepts_normal_keys():
    resolved = _resolve_user_properties_schema({"customer_tier": "str", "score": "int"})
    assert resolved == {"customer_tier": str, "score": int}


def test_resolve_user_properties_schema_rejects_underscore_prefixed_keys():
    """`_`-prefixed keys collide with system-defined event fields
    (`_episode_uid`, `_session_key`, ...). The merged collection schema is
    a dict-spread with user_schema last, so allowing them would silently
    overwrite the system slot and may change its declared type."""
    with pytest.raises(ValueError, match="reserved"):
        _resolve_user_properties_schema({"_episode_uid": "str"})

    with pytest.raises(ValueError, match="reserved"):
        _resolve_user_properties_schema({"_my_field": "int"})


def test_resolve_user_properties_schema_rejects_unknown_type_name():
    with pytest.raises(ValueError, match="unknown type name"):
        _resolve_user_properties_schema({"customer_tier": "date"})


@pytest.mark.asyncio
async def test_event_params_opens_the_collection_a_racing_creator_won():
    """Two workers can create a session's collection at once; the loser opens the winner's.

    The vector store's registry arbitrates creation across processes, so the
    strict create this locator issues when the collection is absent can lose
    to another worker's; the locator then opens the collection that exists
    instead of failing the request.
    """
    config = EventLongTermMemoryConf(
        session_id="raced", vector_store="vs", segment_store="ss", embedder="e"
    )
    collection = create_autospec(VectorStoreCollection, instance=True)
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.open_collection.side_effect = [None, collection]
    vector_store.create_collection.side_effect = (
        VectorStoreCollectionAlreadyExistsError(_EVENT_BACKEND_NAMESPACE, "raced")
    )
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

    assert params.vector_store_collection is collection
    vector_store.create_collection.assert_awaited_once()
    assert vector_store.open_collection.await_count == 2
