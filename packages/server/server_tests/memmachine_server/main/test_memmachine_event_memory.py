"""Unit tests for MemMachine event memory wiring."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from memmachine_common.api.event_memory.config import EventMemoryConf

from memmachine_server.common.configuration.event_memory_config import (
    EventMemoryStoreConf,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import Episode
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import VectorStoreCollection
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    MessageContext,
    Text,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartition,
    SegmentStorePartitionConfig,
)
from memmachine_server.main.memmachine import MemMachine


class DummySessionData:
    def __init__(self, session_key="org/project"):
        self._session_key = session_key

    @property
    def session_key(self):
        return self._session_key

    @property
    def org_id(self):
        return "org"

    @property
    def project_id(self):
        return "project"


def _make_episode(uid="ep1", producer_role="user"):
    return Episode(
        uid=uid,
        content="hello world",
        session_key="org/project",
        created_at=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
        producer_id="alice",
        producer_role=producer_role,
    )


def _make_conf():
    return MagicMock(
        event_memory_store=EventMemoryStoreConf(vector_store="vs", segment_store="ss"),
        event_memory=EventMemoryConf(embedder="emb", reranker="rr"),
    )


def _make_resources(conf):
    resources = AsyncMock()

    # VectorStore mock
    mock_collection = MagicMock(spec=VectorStoreCollection)
    mock_collection.config.properties_schema = {
        "_segment_uuid": str,
        "_timestamp": datetime,
    }
    mock_vector_store = AsyncMock()
    mock_vector_store.open_collection = AsyncMock(return_value=mock_collection)
    mock_vector_store.create_collection = AsyncMock()
    mock_vector_store.delete_collection = AsyncMock()
    resources.get_vector_store = AsyncMock(return_value=mock_vector_store)

    # SegmentStore mock
    mock_partition = MagicMock(spec=SegmentStorePartition)
    mock_segment_store = AsyncMock()
    mock_segment_store.open_or_create_partition = AsyncMock(return_value=mock_partition)
    mock_segment_store.delete_partition = AsyncMock()
    resources.get_segment_store = AsyncMock(return_value=mock_segment_store)

    # Embedder mock
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.dimensions = 1536
    mock_embedder.similarity_metric = SimilarityMetric.COSINE
    resources.get_embedder = AsyncMock(return_value=mock_embedder)

    # Reranker mock
    mock_reranker = MagicMock(spec=Reranker)
    resources.get_reranker = AsyncMock(return_value=mock_reranker)

    # Session data manager mock
    mock_session_manager = AsyncMock()
    mock_session_manager.get_event_memory_conf = AsyncMock(return_value=None)
    resources.get_session_data_manager = AsyncMock(return_value=mock_session_manager)

    return resources


class TestEpisodeToEvent:
    def test_basic_conversion(self):
        episode = _make_episode()
        event = MemMachine._episode_to_event(episode)

        assert isinstance(event.uuid, UUID)
        assert event.timestamp == episode.created_at
        assert isinstance(event.body, Content)
        assert isinstance(event.body.context, MessageContext)
        assert event.body.context.source == "alice"
        assert event.body.context.type == "message"
        assert isinstance(event.body.items[0], Text)
        assert event.body.items[0].text == "hello world"
        assert event.properties["source_role"] == "user"

    def test_deterministic_uuid(self):
        ep1 = _make_episode(uid="same-uid")
        ep2 = _make_episode(uid="same-uid")
        assert (
            MemMachine._episode_to_event(ep1).uuid
            == MemMachine._episode_to_event(ep2).uuid
        )

    def test_different_uid_different_uuid(self):
        ep1 = _make_episode(uid="uid-a")
        ep2 = _make_episode(uid="uid-b")
        assert (
            MemMachine._episode_to_event(ep1).uuid
            != MemMachine._episode_to_event(ep2).uuid
        )

    def test_produced_for_id(self):
        episode = _make_episode()
        episode.produced_for_id = "bob"
        event = MemMachine._episode_to_event(episode)
        assert event.properties["target_id"] == "bob"

    def test_no_produced_for_id(self):
        episode = _make_episode()
        event = MemMachine._episode_to_event(episode)
        assert "target_id" not in event.properties

    def test_filterable_metadata(self):
        episode = _make_episode()
        episode.filterable_metadata = {"custom_key": "custom_value"}
        event = MemMachine._episode_to_event(episode)
        assert event.properties["custom_key"] == "custom_value"


class TestEpisodeUidToEventUuid:
    def test_deterministic(self):
        uuid1 = MemMachine._episode_uid_to_event_uuid("test-uid")
        uuid2 = MemMachine._episode_uid_to_event_uuid("test-uid")
        assert uuid1 == uuid2
        assert isinstance(uuid1, UUID)

    def test_matches_episode_to_event(self):
        episode = _make_episode(uid="my-uid")
        event = MemMachine._episode_to_event(episode)
        assert event.uuid == MemMachine._episode_uid_to_event_uuid("my-uid")


class TestResolvePropertiesSchema:
    def test_valid_types(self):
        result = MemMachine._resolve_properties_schema(
            {"name": "str", "count": "int", "ts": "datetime"}
        )
        assert result["name"] is str
        assert result["count"] is int
        assert result["ts"] is datetime

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="unknown type name"):
            MemMachine._resolve_properties_schema({"bad": "unknown_type"})

    def test_empty(self):
        assert MemMachine._resolve_properties_schema({}) == {}


@pytest.mark.asyncio
async def test_open_event_memory_disabled():
    conf = MagicMock()
    conf.event_memory_store = None
    conf.event_memory = None
    resources = AsyncMock()
    mm = MemMachine(conf, resources)
    result = await mm._open_event_memory("partition")
    assert result is None


@pytest.mark.asyncio
async def test_open_event_memory_creates_new():
    conf = _make_conf()
    resources = _make_resources(conf)

    # open_collection returns None → triggers create path
    mock_vs = await resources.get_vector_store("vs")
    mock_vs.open_collection = AsyncMock(return_value=None)

    mock_collection_new = MagicMock(spec=VectorStoreCollection)
    mock_collection_new.config.properties_schema = {
        "_segment_uuid": str,
        "_timestamp": datetime,
    }
    # After create, open returns the collection
    mock_vs.open_collection = AsyncMock(side_effect=[None, mock_collection_new])

    mm = MemMachine(conf, resources)
    result = await mm._open_event_memory("org/project")
    assert result is not None
    mock_vs.create_collection.assert_called_once()
    mock_ss = await resources.get_segment_store("ss")
    mock_ss.open_or_create_partition.assert_called_once_with(
        "org/project",
        SegmentStorePartitionConfig(),
    )


@pytest.mark.asyncio
async def test_delete_event_memory_partition():
    conf = _make_conf()
    resources = _make_resources(conf)
    mm = MemMachine(conf, resources)

    await mm._delete_event_memory_partition("org/project")

    mock_vs = await resources.get_vector_store("vs")
    mock_ss = await resources.get_segment_store("ss")
    mock_vs.delete_collection.assert_called_once()
    mock_ss.delete_partition.assert_called_once_with("org/project")
