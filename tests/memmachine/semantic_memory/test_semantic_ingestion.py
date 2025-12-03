"""Tests for the ingestion service using the in-memory semantic storage."""

from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from memmachine.common.episode_store import EpisodeEntry, EpisodeIdT, EpisodeStorage
from memmachine.common.filter.filter_parser import parse_filter
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
    SemanticFeature,
    SemanticPrompt,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    MockEmbedder,
    MockResourceRetriever,
)


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


async def add_history(history_storage: EpisodeStorage, content: str) -> EpisodeIdT:
    episode = EpisodeEntry(
        content=content,
        producer_id="profile_id",
        producer_role="dev",
    )
    ret_episode = await history_storage.add_episodes(
        session_key="session_id",
        episodes=[episode],
    )

    assert len(ret_episode) == 1
    return ret_episode[0].uid


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
) -> IngestionService:
    params = IngestionService.Params(
        semantic_storage=semantic_storage,
        history_store=episode_storage,
        resource_retriever=resource_retriever,
        consolidated_threshold=2,
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
        await semantic_storage.get_feature_set(
            filter_expr=parse_filter("set_id IN ('user-123')")
        )
        == []
    )
    assert (
        await semantic_storage.get_history_messages(
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
            command="add",
            feature="favorite_car",
            tag="car",
            value="blue",
        ),
        SemanticCommand(
            command="delete",
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
    features = await semantic_storage.get_feature_set(
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

    filter_str = "set_id IN ('user-123') AND feature_name IN ('favorite_motorcycle')"
    remaining = await semantic_storage.get_feature_set(
        filter_expr=parse_filter(filter_str),
    )
    assert remaining == []

    assert (
        await semantic_storage.get_history_messages(
            set_ids=["user-123"],
            is_ingested=False,
        )
        == []
    )
    ingested = await semantic_storage.get_history_messages(
        set_ids=["user-123"],
        is_ingested=True,
    )
    assert list(ingested) == [message_id]
    assert embedder_double.ingest_calls == [["blue"]]


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
        feature="pizza",
        value="thin crust",
        tag="food",
        embedding=np.array([1.0, -1.0]),
    )
    second_feature = await semantic_storage.add_feature(
        set_id="user-456",
        category_name=semantic_category.name,
        feature="pizza",
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
    memories = await semantic_storage.get_feature_set(
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

    all_features = await semantic_storage.get_feature_set(
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
    assert resources.embedder.ingest_calls == [["consolidated pizza"]]
