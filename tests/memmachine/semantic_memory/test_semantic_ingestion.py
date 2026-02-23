"""Tests for the ingestion service using the in-memory semantic storage."""

from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import EpisodeEntry, EpisodeIdT, EpisodeStorage
from memmachine.common.filter.filter_parser import parse_filter
from memmachine.common.reranker import Reranker
from memmachine.semantic_memory.cluster_manager import (
    ClusterInfo,
    ClusterParams,
    ClusterSplitParams,
    ClusterState,
)
from memmachine.semantic_memory.cluster_splitter import (
    RerankerClusterSplitter,
    segment_cluster_id,
)
from memmachine.semantic_memory.cluster_store.in_memory_cluster_store import (
    InMemoryClusterStateStorage,
)
from memmachine.semantic_memory.semantic_ingestion import IngestionService
from memmachine.semantic_memory.semantic_llm import (
    LLMReducedFeature,
    SemanticConsolidateMemoryRes,
)
from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    Resources,
    SemanticCategory,
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
    SemanticPrompt,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    MockEmbedder,
    MockResourceRetriever,
)


class KeywordEmbedder(Embedder):
    def __init__(self) -> None:
        self.ingest_calls: list[list[str]] = []

    async def ingest_embed(
        self, inputs: list[str], max_attempts: int = 1
    ) -> list[list[float]]:
        self.ingest_calls.append(list(inputs))
        embeddings: list[list[float]] = []
        for text in inputs:
            lowered = text.lower()
            if "alpha" in lowered:
                embeddings.append([1.0, 0.0])
            elif "beta" in lowered:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([1.0, 1.0])
        return embeddings

    async def search_embed(
        self, queries: list[str], max_attempts: int = 1
    ) -> list[list[float]]:
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return "keyword-embedder"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class StubReranker(Reranker):
    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = list(scores or [])
        self.score_calls = 0

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        self.score_calls += 1
        if self._scores:
            score = float(self._scores.pop(0))
            return [score for _ in candidates]
        return [0.0 for _ in candidates]


async def _collect_feature_set(storage: SemanticStorage, **kwargs):
    return [item async for item in storage.get_feature_set(**kwargs)]


async def _collect_history_messages(storage: SemanticStorage, **kwargs):
    return [item async for item in storage.get_history_messages(**kwargs)]


@pytest.fixture
def semantic_prompt() -> SemanticPrompt:
    return RawSemanticPrompt(
        update_prompt="update-prompt",
        consolidation_prompt="consolidation-prompt",
    )


@pytest.fixture
def semantic_category(semantic_prompt: SemanticPrompt) -> SemanticCategory:
    return SemanticCategory(
        name="Profile",
        prompt=semantic_prompt,
    )


@pytest.fixture
def embedder_double() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def llm_model(mock_llm_model):
    return mock_llm_model


async def add_history(
    history_storage: EpisodeStorage,
    content: str,
    *,
    created_at: datetime | None = None,
) -> EpisodeIdT:
    episode = EpisodeEntry(
        content=content,
        producer_id="profile_id",
        producer_role="dev",
        created_at=created_at,
    )
    ret_episode = await history_storage.add_episodes(
        session_key="session_id",
        episodes=[episode],
    )

    assert len(ret_episode) == 1
    return ret_episode[0].uid


def _make_keyword_resource_retriever(
    *,
    llm_model,
    semantic_category: SemanticCategory,
    reranker: Reranker | None = None,
) -> MockResourceRetriever:
    resources = Resources(
        embedder=KeywordEmbedder(),
        language_model=llm_model,
        reranker=reranker,
        semantic_categories=[semantic_category],
    )
    return MockResourceRetriever(resources)


def _make_split_ingestion_service(
    *,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    cluster_state_storage: InMemoryClusterStateStorage,
    similarity_threshold: float = 0.0,
    min_cluster_size: int = 6,
    low_similarity_threshold: float = 0.8,
) -> IngestionService:
    splitter = RerankerClusterSplitter(
        ClusterSplitParams(
            enabled=True,
            min_cluster_size=min_cluster_size,
            low_similarity_threshold=low_similarity_threshold,
        )
    )
    params = IngestionService.Params(
        semantic_storage=semantic_storage,
        history_store=episode_storage,
        resource_retriever=resource_retriever.get_resources,
        consolidated_threshold=100,
        ingestion_trigger_messages=1,
        cluster_idle_ttl=timedelta(days=3650),
        cluster_state_storage=cluster_state_storage,
        cluster_params=ClusterParams(similarity_threshold=similarity_threshold),
        cluster_splitter=splitter,
    )
    return IngestionService(params)


async def _add_keyword_messages(
    *,
    episode_storage: EpisodeStorage,
    semantic_storage: SemanticStorage,
    set_id: str,
    contents: list[str],
) -> list[EpisodeIdT]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    message_ids: list[EpisodeIdT] = []
    for idx, content in enumerate(contents):
        message_id = await add_history(
            episode_storage,
            content=content,
            created_at=start + timedelta(minutes=idx),
        )
        await semantic_storage.add_history_to_set(
            set_id=set_id,
            history_id=message_id,
        )
        message_ids.append(message_id)
    return message_ids


@pytest.fixture
def resources(
    embedder_double: MockEmbedder,
    llm_model,
    semantic_category: SemanticCategory,
) -> Resources:
    return Resources(
        embedder=embedder_double,
        language_model=llm_model,
        semantic_categories=[semantic_category],
    )


@pytest.fixture
def resource_retriever(resources: Resources) -> MockResourceRetriever:
    return MockResourceRetriever(resources)


@pytest_asyncio.fixture
async def ingestion_service(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    in_memory_cluster_state_storage,
) -> IngestionService:
    params = IngestionService.Params(
        semantic_storage=semantic_storage,
        history_store=episode_storage,
        resource_retriever=resource_retriever.get_resources,
        consolidated_threshold=2,
        ingestion_trigger_messages=1,
        cluster_idle_ttl=timedelta(days=3650),
        cluster_state_storage=in_memory_cluster_state_storage,
    )
    return IngestionService(params)


@pytest.mark.asyncio
async def test_process_single_set_returns_when_no_messages(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    resource_retriever: MockResourceRetriever,
):
    await ingestion_service._process_single_set("user-123")

    assert resource_retriever.seen_ids == ["user-123"]
    assert (
        await _collect_feature_set(
            semantic_storage,
            filter_expr=parse_filter("set_id IN ('user-123')"),
        )
        == []
    )
    assert (
        await _collect_history_messages(
            semantic_storage,
            set_ids=["user-123"],
            is_ingested=False,
        )
        == []
    )


@pytest.mark.asyncio
async def test_process_single_set_applies_commands(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    embedder_double: MockEmbedder,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    message_id = await add_history(episode_storage, content="I love blue cars")
    await semantic_storage.add_history_to_set(set_id="user-123", history_id=message_id)

    await semantic_storage.add_feature(
        set_id="user-123",
        category_name=semantic_category.name,
        feature="favorite_motorcycle",
        value="old bike",
        tag="bike",
        embedding=np.array([1.0, 1.0]),
    )

    commands = [
        SemanticCommand(
            command=SemanticCommandType.ADD,
            feature="favorite_car",
            tag="car",
            value="blue",
        ),
        SemanticCommand(
            command=SemanticCommandType.DELETE,
            feature="favorite_motorcycle",
            tag="bike",
            value="",
        ),
    ]
    llm_feature_update_mock = AsyncMock(return_value=commands)
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set("user-123")

    llm_feature_update_mock.assert_awaited_once()
    filter_str = (
        f"set_id IN ('user-123') AND category_name IN ('{semantic_category.name}')"
    )
    features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
        load_citations=True,
    )
    assert len(features) == 1
    feature = features[0]
    assert feature.feature_name == "favorite_car"
    assert feature.value == "blue"
    assert feature.tag == "car"
    assert feature.metadata.citations is not None
    assert list(feature.metadata.citations) == [message_id]
    assert feature.metadata.other == {"cluster_id": "cluster_0"}

    filter_str = "set_id IN ('user-123') AND feature_name IN ('favorite_motorcycle')"
    remaining = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
    )
    assert remaining == []

    assert (
        await _collect_history_messages(
            semantic_storage,
            set_ids=["user-123"],
            is_ingested=False,
        )
        == []
    )
    ingested = await _collect_history_messages(
        semantic_storage,
        set_ids=["user-123"],
        is_ingested=True,
    )
    assert list(ingested) == [message_id]
    assert embedder_double.ingest_calls == [["I love blue cars"], ["blue"]]


@pytest.mark.asyncio
async def test_delete_scopes_cluster_or_null_metadata(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    embedder_double: MockEmbedder,
    semantic_category: SemanticCategory,
):
    feature_cluster = await semantic_storage.add_feature(
        set_id="user-555",
        category_name=semantic_category.name,
        feature="favorite_food",
        value="pizza",
        tag="food",
        embedding=np.array([1.0, 1.0]),
        metadata={"cluster_id": "cluster_a"},
    )
    feature_null = await semantic_storage.add_feature(
        set_id="user-555",
        category_name=semantic_category.name,
        feature="favorite_food",
        value="pizza",
        tag="food",
        embedding=np.array([1.0, 1.0]),
    )
    feature_other = await semantic_storage.add_feature(
        set_id="user-555",
        category_name=semantic_category.name,
        feature="favorite_food",
        value="pizza",
        tag="food",
        embedding=np.array([1.0, 1.0]),
        metadata={"cluster_id": "cluster_b"},
    )

    await ingestion_service._apply_commands(
        commands=[
            SemanticCommand(
                command=SemanticCommandType.DELETE,
                feature="favorite_food",
                tag="food",
                value="",
            )
        ],
        set_id="user-555",
        category_name=semantic_category.name,
        citation_ids=None,
        embedder=embedder_double,
        cluster_id="cluster_a",
    )

    filter_str = (
        f"set_id IN ('user-555') AND category_name IN ('{semantic_category.name}')"
    )
    remaining = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
    )
    remaining_ids = {feature.metadata.id for feature in remaining}
    assert feature_other in remaining_ids
    assert feature_cluster not in remaining_ids
    assert feature_null not in remaining_ids


@pytest.mark.asyncio
async def test_clustered_messages_group_llm_calls(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    embedder = KeywordEmbedder()
    resources = Resources(
        embedder=embedder,
        language_model=llm_model,
        semantic_categories=[semantic_category],
    )
    resource_retriever = MockResourceRetriever(resources)
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            ingestion_trigger_messages=1,
            cluster_idle_ttl=timedelta(days=3650),
            cluster_state_storage=in_memory_cluster_state_storage,
            cluster_params=ClusterParams(similarity_threshold=0.8),
        )
    )

    msg1 = await add_history(episode_storage, content="alpha message 1")
    msg2 = await add_history(episode_storage, content="alpha message 2")
    await semantic_storage.add_history_to_set(set_id="user-abc", history_id=msg1)
    await semantic_storage.add_history_to_set(set_id="user-abc", history_id=msg2)

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set("user-abc")

    assert llm_feature_update_mock.await_count == 1
    ingested = await _collect_history_messages(
        semantic_storage,
        set_ids=["user-abc"],
        is_ingested=True,
    )
    assert set(ingested) == {msg1, msg2}


@pytest.mark.asyncio
async def test_distinct_clusters_call_llm_per_cluster(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    embedder = KeywordEmbedder()
    resources = Resources(
        embedder=embedder,
        language_model=llm_model,
        semantic_categories=[semantic_category],
    )
    resource_retriever = MockResourceRetriever(resources)
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            ingestion_trigger_messages=1,
            cluster_idle_ttl=timedelta(days=3650),
            cluster_state_storage=in_memory_cluster_state_storage,
            cluster_params=ClusterParams(similarity_threshold=0.8),
        )
    )

    msg1 = await add_history(episode_storage, content="alpha message")
    msg2 = await add_history(episode_storage, content="beta message")
    await semantic_storage.add_history_to_set(set_id="user-def", history_id=msg1)
    await semantic_storage.add_history_to_set(set_id="user-def", history_id=msg2)

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set("user-def")

    assert llm_feature_update_mock.await_count == 2


@pytest.mark.asyncio
async def test_cluster_pending_persists_across_batches(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    embedder = KeywordEmbedder()
    resources = Resources(
        embedder=embedder,
        language_model=llm_model,
        semantic_categories=[semantic_category],
    )
    resource_retriever = MockResourceRetriever(resources)
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            ingestion_trigger_messages=3,
            ingestion_trigger_age=timedelta(days=3650),
            cluster_idle_ttl=timedelta(days=3650),
            cluster_state_storage=in_memory_cluster_state_storage,
            cluster_params=ClusterParams(similarity_threshold=0.8),
        )
    )

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    set_id = "user-pending-1"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    msg_alpha_1 = await add_history(
        episode_storage, content="alpha message 1", created_at=start
    )
    msg_beta_1 = await add_history(
        episode_storage,
        content="beta message 1",
        created_at=start + timedelta(minutes=1),
    )
    await semantic_storage.add_history_to_set(set_id=set_id, history_id=msg_alpha_1)
    await semantic_storage.add_history_to_set(set_id=set_id, history_id=msg_beta_1)

    await ingestion_service._process_single_set(set_id)

    pending = await _collect_history_messages(
        semantic_storage,
        set_ids=[set_id],
        is_ingested=False,
    )
    assert set(pending) == {msg_alpha_1, msg_beta_1}

    msg_alpha_2 = await add_history(
        episode_storage,
        content="alpha message 2",
        created_at=start + timedelta(minutes=2),
    )
    msg_alpha_3 = await add_history(
        episode_storage,
        content="alpha message 3",
        created_at=start + timedelta(minutes=3),
    )
    await semantic_storage.add_history_to_set(set_id=set_id, history_id=msg_alpha_2)
    await semantic_storage.add_history_to_set(set_id=set_id, history_id=msg_alpha_3)

    await ingestion_service._process_single_set(set_id)

    ingested = await _collect_history_messages(
        semantic_storage,
        set_ids=[set_id],
        is_ingested=True,
    )
    assert set(ingested) == {msg_alpha_1, msg_alpha_2, msg_alpha_3}
    remaining = await _collect_history_messages(
        semantic_storage,
        set_ids=[set_id],
        is_ingested=False,
    )
    assert set(remaining) == {msg_beta_1}

    state = await in_memory_cluster_state_storage.get_state(set_id=set_id)
    assert state is not None
    assert msg_beta_1 in state.event_to_cluster
    assert msg_alpha_1 not in state.event_to_cluster


@pytest.mark.asyncio
async def test_splitter_topic_shift_calls_feature_update_per_segment(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    reranker = StubReranker(
        scores=[0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9],
    )
    resource_retriever = _make_keyword_resource_retriever(
        llm_model=llm_model,
        semantic_category=semantic_category,
        reranker=reranker,
    )
    ingestion_service = _make_split_ingestion_service(
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        resource_retriever=resource_retriever,
        cluster_state_storage=in_memory_cluster_state_storage,
        similarity_threshold=0.0,
    )

    set_id = "user-split-1"
    message_ids = await _add_keyword_messages(
        episode_storage=episode_storage,
        semantic_storage=semantic_storage,
        set_id=set_id,
        contents=[
            *[f"alpha message {i}" for i in range(4)],
            *[f"beta message {i}" for i in range(4)],
        ],
    )

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set(set_id)

    assert reranker.score_calls == 7
    assert llm_feature_update_mock.await_count == 2
    ingested = await _collect_history_messages(
        semantic_storage,
        set_ids=[set_id],
        is_ingested=True,
    )
    assert set(ingested) == set(message_ids)

    state = await in_memory_cluster_state_storage.get_state(set_id=set_id)
    assert state is not None
    assert len(state.split_records) == 1
    record = next(iter(state.split_records.values()))
    assert len(record.segment_ids) == 2
    assert record.input_hash
    assert record.segment_ids[0] == record.original_cluster_id
    assert record.segment_ids[1] == segment_cluster_id(
        record.original_cluster_id,
        1,
    )


@pytest.mark.asyncio
async def test_splitter_reranker_no_split_keeps_cluster_intact(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    reranker = StubReranker(scores=[0.9] * 7)
    resource_retriever = _make_keyword_resource_retriever(
        llm_model=llm_model,
        semantic_category=semantic_category,
        reranker=reranker,
    )
    ingestion_service = _make_split_ingestion_service(
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        resource_retriever=resource_retriever,
        cluster_state_storage=in_memory_cluster_state_storage,
        similarity_threshold=0.0,
    )

    set_id = "user-split-2"
    await _add_keyword_messages(
        episode_storage=episode_storage,
        semantic_storage=semantic_storage,
        set_id=set_id,
        contents=[
            *[f"alpha message {i}" for i in range(6)],
            *[f"beta message {i}" for i in range(2)],
        ],
    )

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set(set_id)

    assert reranker.score_calls == 7
    assert llm_feature_update_mock.await_count == 1
    state = await in_memory_cluster_state_storage.get_state(set_id=set_id)
    assert state is not None
    assert len(state.split_records) == 1
    record = next(iter(state.split_records.values()))
    assert record.segment_ids == []


@pytest.mark.asyncio
async def test_splitter_reingestion_skips_reranker_scoring(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    reranker = StubReranker(
        scores=[0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9],
    )
    resource_retriever = _make_keyword_resource_retriever(
        llm_model=llm_model,
        semantic_category=semantic_category,
        reranker=reranker,
    )
    ingestion_service = _make_split_ingestion_service(
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        resource_retriever=resource_retriever,
        cluster_state_storage=in_memory_cluster_state_storage,
        similarity_threshold=0.0,
    )

    set_id = "user-split-3"
    message_ids = await _add_keyword_messages(
        episode_storage=episode_storage,
        semantic_storage=semantic_storage,
        set_id=set_id,
        contents=[
            *[f"alpha message {i}" for i in range(4)],
            *[f"beta message {i}" for i in range(4)],
        ],
    )

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set(set_id)

    assert reranker.score_calls == 7
    llm_feature_update_mock.reset_mock()

    await semantic_storage.delete_history(message_ids)
    for message_id in message_ids:
        await semantic_storage.add_history_to_set(
            set_id=set_id,
            history_id=message_id,
        )

    await ingestion_service._process_single_set(set_id)

    assert reranker.score_calls == 7
    assert llm_feature_update_mock.await_count == 2


@pytest.mark.asyncio
async def test_splitter_feature_metadata_uses_segment_ids(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    reranker = StubReranker(
        scores=[0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9],
    )
    resource_retriever = _make_keyword_resource_retriever(
        llm_model=llm_model,
        semantic_category=semantic_category,
        reranker=reranker,
    )
    ingestion_service = _make_split_ingestion_service(
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        resource_retriever=resource_retriever,
        cluster_state_storage=in_memory_cluster_state_storage,
        similarity_threshold=0.0,
    )

    set_id = "user-split-4"
    await _add_keyword_messages(
        episode_storage=episode_storage,
        semantic_storage=semantic_storage,
        set_id=set_id,
        contents=[
            *[f"alpha message {i}" for i in range(4)],
            *[f"beta message {i}" for i in range(4)],
        ],
    )

    llm_feature_update_mock = AsyncMock(
        side_effect=[
            [
                SemanticCommand(
                    command=SemanticCommandType.ADD,
                    feature="topic_a",
                    value="alpha stuff",
                    tag="test",
                )
            ],
            [
                SemanticCommand(
                    command=SemanticCommandType.ADD,
                    feature="topic_b",
                    value="beta stuff",
                    tag="test",
                )
            ],
        ]
    )
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set(set_id)

    filter_str = (
        f"set_id IN ('{set_id}') AND category_name IN ('{semantic_category.name}')"
    )
    features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
    )
    features_by_name = {feature.feature_name: feature for feature in features}
    assert set(features_by_name.keys()) == {"topic_a", "topic_b"}

    state = await in_memory_cluster_state_storage.get_state(set_id=set_id)
    assert state is not None
    record = next(iter(state.split_records.values()))
    child_segment_id = segment_cluster_id(record.original_cluster_id, 1)

    assert features_by_name["topic_a"].metadata.other == {
        "cluster_id": record.original_cluster_id
    }
    assert features_by_name["topic_b"].metadata.other == {
        "cluster_id": child_segment_id
    }


@pytest.mark.asyncio
async def test_noop_splitter_leaves_pipeline_unchanged(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category: SemanticCategory,
    llm_model,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    resource_retriever = _make_keyword_resource_retriever(
        llm_model=llm_model,
        semantic_category=semantic_category,
    )
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=100,
            ingestion_trigger_messages=1,
            cluster_idle_ttl=timedelta(days=3650),
            cluster_state_storage=in_memory_cluster_state_storage,
            cluster_params=ClusterParams(similarity_threshold=0.0),
        )
    )

    set_id = "user-split-5"
    await _add_keyword_messages(
        episode_storage=episode_storage,
        semantic_storage=semantic_storage,
        set_id=set_id,
        contents=[
            *[f"alpha message {i}" for i in range(4)],
            *[f"beta message {i}" for i in range(4)],
        ],
    )

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set(set_id)

    assert llm_feature_update_mock.await_count == 1
    state = await in_memory_cluster_state_storage.get_state(set_id=set_id)
    assert state is not None
    assert state.split_records == {}


@pytest.mark.asyncio
async def test_reingest_event_id_reuses_cluster(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    message_id = await add_history(episode_storage, content="alpha message")
    await semantic_storage.add_history_to_set(set_id="user-777", history_id=message_id)

    now = datetime.now(tz=UTC)
    state = ClusterState(
        clusters={
            "cluster_0": ClusterInfo(
                centroid=[1.0, 0.0],
                count=1,
                last_ts=now,
            )
        },
        event_to_cluster={message_id: "cluster_0"},
        pending_events={"cluster_0": {message_id: now}},
        next_cluster_id=1,
    )
    await in_memory_cluster_state_storage.save_state(set_id="user-777", state=state)

    llm_feature_update_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set("user-777")

    llm_feature_update_mock.assert_awaited_once()
    loaded = await in_memory_cluster_state_storage.get_state(set_id="user-777")
    assert loaded is not None
    assert loaded.next_cluster_id == 1
    assert message_id not in loaded.event_to_cluster
    assert loaded.clusters["cluster_0"].count == 1


@pytest.mark.asyncio
async def test_consolidation_groups_by_tag(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    first_history = await add_history(episode_storage, content="thin crust")
    second_history = await add_history(episode_storage, content="deep dish")

    first_feature = await semantic_storage.add_feature(
        set_id="user-456",
        category_name=semantic_category.name,
        feature="pizza_crust",
        value="thin crust",
        tag="food",
        embedding=np.array([1.0, -1.0]),
    )
    second_feature = await semantic_storage.add_feature(
        set_id="user-456",
        category_name=semantic_category.name,
        feature="pizza_style",
        value="deep dish",
        tag="food",
        embedding=np.array([2.0, -2.0]),
    )
    await semantic_storage.add_citations(first_feature, [first_history])
    await semantic_storage.add_citations(second_feature, [second_history])

    dedupe_mock = AsyncMock()
    monkeypatch.setattr(ingestion_service, "_deduplicate_features", dedupe_mock)

    await ingestion_service._consolidate_set_memories_if_applicable(
        set_id="user-456",
        resources=resources,
    )

    assert dedupe_mock.await_count == 1
    call = dedupe_mock.await_args_list[0]
    memories: list[SemanticFeature] = call.kwargs["memories"]
    assert {m.metadata.id for m in memories} == {first_feature, second_feature}
    assert call.kwargs["set_id"] == "user-456"
    assert call.kwargs["semantic_category"] == semantic_category
    assert call.kwargs["resources"] == resources
    assert call.kwargs["cluster_id"] is None


@pytest.mark.asyncio
async def test_consolidation_separates_clusters(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    resources: Resources,
    semantic_category: SemanticCategory,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            cluster_state_storage=in_memory_cluster_state_storage,
        )
    )

    await semantic_storage.add_feature(
        set_id="user-999",
        category_name=semantic_category.name,
        feature="pizza_crust",
        value="thin crust",
        tag="food",
        embedding=np.array([1.0, -1.0]),
        metadata={"cluster_id": "cluster_a"},
    )
    await semantic_storage.add_feature(
        set_id="user-999",
        category_name=semantic_category.name,
        feature="pizza_style",
        value="deep dish",
        tag="food",
        embedding=np.array([2.0, -2.0]),
        metadata={"cluster_id": "cluster_a"},
    )
    await semantic_storage.add_feature(
        set_id="user-999",
        category_name=semantic_category.name,
        feature="favorite_drink",
        value="tea",
        tag="food",
        embedding=np.array([3.0, -3.0]),
        metadata={"cluster_id": "cluster_b"},
    )
    await semantic_storage.add_feature(
        set_id="user-999",
        category_name=semantic_category.name,
        feature="favorite_snack",
        value="chips",
        tag="food",
        embedding=np.array([4.0, -4.0]),
        metadata={"cluster_id": "cluster_b"},
    )

    dedupe_mock = AsyncMock()
    monkeypatch.setattr(ingestion_service, "_deduplicate_features", dedupe_mock)

    await ingestion_service._consolidate_set_memories_if_applicable(
        set_id="user-999",
        resources=resources,
    )

    assert dedupe_mock.await_count == 2
    cluster_ids = {call.kwargs["cluster_id"] for call in dedupe_mock.await_args_list}
    assert cluster_ids == {"cluster_a", "cluster_b"}


@pytest.mark.asyncio
async def test_consolidation_skips_small_groups(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    resources: Resources,
    semantic_category: SemanticCategory,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=3,
            cluster_state_storage=in_memory_cluster_state_storage,
        )
    )

    await semantic_storage.add_feature(
        set_id="user-321",
        category_name=semantic_category.name,
        feature="pizza_crust",
        value="thin crust",
        tag="food",
        embedding=np.array([1.0, -1.0]),
    )
    await semantic_storage.add_feature(
        set_id="user-321",
        category_name=semantic_category.name,
        feature="pizza_style",
        value="deep dish",
        tag="food",
        embedding=np.array([2.0, -2.0]),
    )

    dedupe_mock = AsyncMock()
    monkeypatch.setattr(ingestion_service, "_deduplicate_features", dedupe_mock)

    await ingestion_service._consolidate_set_memories_if_applicable(
        set_id="user-321",
        resources=resources,
    )

    dedupe_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_consolidation_runs_when_threshold_met(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    resources: Resources,
    semantic_category: SemanticCategory,
    in_memory_cluster_state_storage: InMemoryClusterStateStorage,
    monkeypatch,
):
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=3,
            cluster_state_storage=in_memory_cluster_state_storage,
        )
    )

    await semantic_storage.add_feature(
        set_id="user-654",
        category_name=semantic_category.name,
        feature="pizza_crust",
        value="thin crust",
        tag="food",
        embedding=np.array([1.0, -1.0]),
    )
    await semantic_storage.add_feature(
        set_id="user-654",
        category_name=semantic_category.name,
        feature="pizza_style",
        value="deep dish",
        tag="food",
        embedding=np.array([2.0, -2.0]),
    )
    await semantic_storage.add_feature(
        set_id="user-654",
        category_name=semantic_category.name,
        feature="pizza_topping",
        value="pepperoni",
        tag="food",
        embedding=np.array([3.0, -3.0]),
    )

    dedupe_mock = AsyncMock()
    monkeypatch.setattr(ingestion_service, "_deduplicate_features", dedupe_mock)

    await ingestion_service._consolidate_set_memories_if_applicable(
        set_id="user-654",
        resources=resources,
    )

    dedupe_mock.assert_awaited_once()
    call = dedupe_mock.await_args_list[0]
    memories: list[SemanticFeature] = call.kwargs["memories"]
    assert {memory.value for memory in memories} == {
        "thin crust",
        "deep dish",
        "pepperoni",
    }


@pytest.mark.asyncio
async def test_deduplicate_features_merges_and_relabels(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    keep_history = await add_history(episode_storage, content="keep")
    drop_history = await add_history(episode_storage, content="drop")

    keep_feature_id = await semantic_storage.add_feature(
        set_id="user-789",
        category_name=semantic_category.name,
        feature="pizza",
        value="original pizza",
        tag="food",
        embedding=np.array([1.0, 0.5]),
    )
    drop_feature_id = await semantic_storage.add_feature(
        set_id="user-789",
        category_name=semantic_category.name,
        feature="pizza",
        value="duplicate pizza",
        tag="food",
        embedding=np.array([2.0, 1.0]),
    )

    await semantic_storage.add_citations(keep_feature_id, [keep_history])
    await semantic_storage.add_citations(drop_feature_id, [drop_history])

    filter_str = (
        f"set_id IN ('user-789') AND category_name IN ('{semantic_category.name}')"
    )
    memories = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
        load_citations=True,
    )

    consolidated_feature = LLMReducedFeature(
        tag="food",
        feature="pizza",
        value="consolidated pizza",
    )
    llm_consolidate_mock = AsyncMock(
        return_value=SemanticConsolidateMemoryRes(
            consolidated_memories=[consolidated_feature],
            keep_memories=[keep_feature_id],
        ),
    )
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_consolidate_mock,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-789",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
        cluster_id=None,
    )

    llm_consolidate_mock.assert_awaited_once()
    assert (
        await semantic_storage.get_feature(drop_feature_id, load_citations=True) is None
    )
    kept_feature = await semantic_storage.get_feature(
        keep_feature_id,
        load_citations=True,
    )
    assert kept_feature is not None
    assert kept_feature.value == "original pizza"

    all_features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
        load_citations=True,
    )
    consolidated = next(
        (f for f in all_features if f.value == "consolidated pizza"),
        None,
    )
    assert consolidated is not None
    assert consolidated.tag == "food"
    assert consolidated.feature_name == "pizza"
    assert consolidated.metadata.citations is not None
    assert list(consolidated.metadata.citations) == [drop_history]
    embedder = cast(MockEmbedder, resources.embedder)
    assert embedder.ingest_calls == [["consolidated pizza"]]


@pytest.mark.asyncio
async def test_consolidation_preserves_cluster_metadata(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    keep_history = await add_history(episode_storage, content="keep")
    drop_history = await add_history(episode_storage, content="drop")

    keep_feature_id = await semantic_storage.add_feature(
        set_id="user-888",
        category_name=semantic_category.name,
        feature="pizza",
        value="original pizza",
        tag="food",
        embedding=np.array([1.0, 0.5]),
        metadata={"cluster_id": "cluster_7"},
    )
    drop_feature_id = await semantic_storage.add_feature(
        set_id="user-888",
        category_name=semantic_category.name,
        feature="pizza",
        value="duplicate pizza",
        tag="food",
        embedding=np.array([2.0, 1.0]),
        metadata={"cluster_id": "cluster_7"},
    )

    await semantic_storage.add_citations(keep_feature_id, [keep_history])
    await semantic_storage.add_citations(drop_feature_id, [drop_history])

    filter_str = (
        f"set_id IN ('user-888') AND category_name IN ('{semantic_category.name}')"
    )
    memories = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
        load_citations=True,
    )

    consolidated_feature = LLMReducedFeature(
        tag="food",
        feature="pizza",
        value="consolidated pizza",
    )
    llm_consolidate_mock = AsyncMock(
        return_value=SemanticConsolidateMemoryRes(
            consolidated_memories=[consolidated_feature],
            keep_memories=[keep_feature_id],
        ),
    )
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_consolidate_mock,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-888",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
        cluster_id="cluster_7",
    )

    all_features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
        load_citations=True,
    )
    consolidated = next(
        (feature for feature in all_features if feature.value == "consolidated pizza"),
        None,
    )
    assert consolidated is not None
    assert consolidated.metadata.other == {"cluster_id": "cluster_7"}
