"""Tests for the ingestion service using the in-memory semantic storage."""

from typing import cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.episode_store import (
    EpisodeEntry,
    EpisodeIdT,
    EpisodeStorage,
)
from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.semantic_memory.semantic_ingestion import IngestionService
from memmachine_server.semantic_memory.semantic_llm import (
    LLMReducedFeature,
    SemanticConsolidateMemoryRes,
)
from memmachine_server.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    Resources,
    SemanticCategory,
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
    SemanticPrompt,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from server_tests.memmachine_server.semantic_memory.mock_semantic_memory_objects import (
    MockEmbedder,
    MockResourceRetriever,
)


async def _collect(async_iter):
    return [item async for item in async_iter]


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
        resource_retriever=resource_retriever.get_resources,
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
        await _collect(
            semantic_storage.get_feature_set(
                filter_expr=parse_filter("set_id IN ('user-123')")
            )
        )
        == []
    )
    assert (
        await _collect(
            semantic_storage.get_history_messages(
                set_ids=["user-123"],
                is_ingested=False,
            )
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
        "memmachine_server.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set("user-123")

    llm_feature_update_mock.assert_awaited_once()
    filter_str = (
        f"set_id IN ('user-123') AND category_name IN ('{semantic_category.name}')"
    )
    features = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
            load_citations=True,
        )
    )
    assert len(features) == 1
    feature = features[0]
    assert feature.feature_name == "favorite_car"
    assert feature.value == "blue"
    assert feature.tag == "car"
    assert feature.metadata.citations is not None
    assert list(feature.metadata.citations) == [message_id]

    filter_str = "set_id IN ('user-123') AND feature_name IN ('favorite_motorcycle')"
    remaining = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
        )
    )
    assert remaining == []

    assert (
        await _collect(
            semantic_storage.get_history_messages(
                set_ids=["user-123"],
                is_ingested=False,
            )
        )
        == []
    )
    ingested = await _collect(
        semantic_storage.get_history_messages(
            set_ids=["user-123"],
            is_ingested=True,
        )
    )
    assert list(ingested) == [message_id]
    assert embedder_double.ingest_calls == [["blue"]]


@pytest.mark.asyncio
async def test_process_single_set_marks_context_length_errors_as_ingested(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    monkeypatch,
):
    class ContextLengthExceededError(Exception):
        code = "context_length_exceeded"

    message_id = await add_history(
        episode_storage,
        content="very long message that exceeds model context",
    )
    await semantic_storage.add_history_to_set(
        set_id="user-context-overflow",
        history_id=message_id,
    )

    async def mock_context_length_error(*args, **kwargs):
        err = ContextLengthExceededError("input exceeds context window")
        raise ExternalServiceAPIError("LLM request failed") from err

    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_feature_update",
        mock_context_length_error,
    )

    await ingestion_service._process_single_set("user-context-overflow")

    assert (
        await _collect(
            semantic_storage.get_history_messages(
                set_ids=["user-context-overflow"],
                is_ingested=False,
            )
        )
        == []
    )
    assert await _collect(
        semantic_storage.get_history_messages(
            set_ids=["user-context-overflow"],
            is_ingested=True,
        )
    ) == [message_id]


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


@pytest.mark.asyncio
async def test_consolidation_skips_small_groups(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=3,
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
    monkeypatch,
):
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=3,
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
    memories = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
            load_citations=True,
        )
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
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
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

    all_features = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
            load_citations=True,
        )
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
async def test_process_single_set_deletes_invalid_episode_ids(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    embedder_double: MockEmbedder,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """When some episode_ids resolve to None, the service deletes them from
    semantic storage and continues processing the valid ones."""
    valid_id = await add_history(episode_storage, content="valid message")
    # Use a numeric string so the episode store doesn't reject the format,
    # but one that doesn't correspond to any real episode (returns None).
    invalid_id = "99999"

    await semantic_storage.add_history_to_set(set_id="user-999", history_id=valid_id)
    await semantic_storage.add_history_to_set(set_id="user-999", history_id=invalid_id)

    commands = [
        SemanticCommand(
            command=SemanticCommandType.ADD,
            feature="greeting",
            tag="social",
            value="hello",
        ),
    ]
    llm_mock = AsyncMock(return_value=commands)
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_mock,
    )

    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            debug_fail_loudly=False,
        )
    )

    await ingestion_service._process_single_set("user-999")

    # The invalid episode_id should have been delisted from semantic storage
    remaining_history = await _collect(
        semantic_storage.get_history_messages(
            set_ids=["user-999"],
            is_ingested=False,
        )
    )
    assert invalid_id not in remaining_history

    # The valid message should still have been processed
    llm_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_single_set_raises_in_debug_mode_for_invalid_ids(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    semantic_category: SemanticCategory,
):
    """When debug_fail_loudly is True and invalid episode_ids are found,
    the service raises a ValueError with set_id and invalid ids in the message."""
    valid_id = await add_history(episode_storage, content="valid message")
    invalid_id = "99999"

    await semantic_storage.add_history_to_set(set_id="user-888", history_id=valid_id)
    await semantic_storage.add_history_to_set(set_id="user-888", history_id=invalid_id)

    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            debug_fail_loudly=True,
        )
    )

    with pytest.raises(ValueError, match="user-888"):
        await ingestion_service._process_single_set("user-888")


@pytest.mark.asyncio
async def test_process_single_set_limits_features_sent_to_llm(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """The feature lookup for each message must pass a page_size cap so that a
    very large profile does not cause the LLM response to overflow its token budget."""
    message_id = await add_history(episode_storage, content="I love sushi")
    await semantic_storage.add_history_to_set(set_id="user-222", history_id=message_id)

    # Seed more features than the expected cap
    for i in range(60):
        await semantic_storage.add_feature(
            set_id="user-222",
            category_name=semantic_category.name,
            feature=f"feature_{i}",
            value=f"value_{i}",
            tag="misc",
            embedding=np.array([float(i), float(-i)]),
        )

    llm_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_mock,
    )

    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=100,
            debug_fail_loudly=False,
        )
    )

    await ingestion_service._process_single_set("user-222")

    # get_feature_set must have been called with a non-None page_size
    # to prevent sending all 60 features to the LLM at once.
    llm_mock.assert_awaited_once()
    assert llm_mock.await_args is not None
    passed_features: list[SemanticFeature] = llm_mock.await_args.kwargs["features"]
    assert len(passed_features) < 60, (
        "Expected feature set to be capped before calling the LLM, "
        f"but {len(passed_features)} features were passed"
    )


@pytest.mark.asyncio
async def test_consolidation_deletes_features_not_in_keep_list(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """Any feature whose ID is absent from ``keep_memories``
    is unconditionally deleted, even when it holds valid user data."""
    hist1 = await add_history(episode_storage, content="fixed observer bug")
    hist2 = await add_history(episode_storage, content="added more agents")

    bugfix_id = await semantic_storage.add_feature(
        set_id="user-1160a",
        category_name=semantic_category.name,
        feature="observer_fix",
        value="Fixed observer subagent bug",
        tag="bugfix",
        embedding=np.array([1.0, 0.0]),
    )
    progress_id = await semantic_storage.add_feature(
        set_id="user-1160a",
        category_name=semantic_category.name,
        feature="more_agents",
        value="User added more agents",
        tag="bugfix",
        embedding=np.array([0.0, 1.0]),
    )
    await semantic_storage.add_citations(bugfix_id, [hist1])
    await semantic_storage.add_citations(progress_id, [hist2])

    filter_str = (
        f"set_id IN ('user-1160a') AND category_name IN ('{semantic_category.name}')"
    )
    memories = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
            load_citations=True,
        )
    )

    # LLM keeps only bugfix_id — progress_id is omitted and will be deleted.
    llm_mock = AsyncMock(
        return_value=SemanticConsolidateMemoryRes(
            consolidated_memories=[],
            keep_memories=[bugfix_id],
        ),
    )
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_mock,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-1160a",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
    )

    assert await semantic_storage.get_feature(progress_id) is None, (
        "Feature deleted because its ID was not in keep_memories"
    )
    assert await semantic_storage.get_feature(bugfix_id) is not None


@pytest.mark.asyncio
async def test_empty_keep_memories_deletes_all_features(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """An empty ``keep_memories`` list causes every feature in the
    consolidated group to be deleted."""
    hist1 = await add_history(episode_storage, content="msg1")
    hist2 = await add_history(episode_storage, content="msg2")

    id1 = await semantic_storage.add_feature(
        set_id="user-1160b",
        category_name=semantic_category.name,
        feature="feat_a",
        value="value a",
        tag="bugfix",
        embedding=np.array([1.0, 0.0]),
    )
    id2 = await semantic_storage.add_feature(
        set_id="user-1160b",
        category_name=semantic_category.name,
        feature="feat_b",
        value="value b",
        tag="bugfix",
        embedding=np.array([0.0, 1.0]),
    )
    await semantic_storage.add_citations(id1, [hist1])
    await semantic_storage.add_citations(id2, [hist2])

    filter_str = (
        f"set_id IN ('user-1160b') AND category_name IN ('{semantic_category.name}')"
    )
    memories = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
            load_citations=True,
        )
    )

    consolidated = LLMReducedFeature(
        tag="bugfix", feature="combined", value="combined a + b"
    )
    llm_mock = AsyncMock(
        return_value=SemanticConsolidateMemoryRes(
            consolidated_memories=[consolidated],
            keep_memories=[],
        ),
    )
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_mock,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-1160b",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
    )

    assert await semantic_storage.get_feature(id1) is None
    assert await semantic_storage.get_feature(id2) is None

    remaining = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
        )
    )
    assert len(remaining) == 1
    assert remaining[0].value == "combined a + b"


@pytest.mark.asyncio
async def test_consolidation_rejects_llm_tag_rename(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """Consolidated features keep their original tag even when the LLM
    invents a new one like 'Productivity Style'."""
    hist1 = await add_history(episode_storage, content="msg1")
    hist2 = await add_history(episode_storage, content="msg2")

    id1 = await semantic_storage.add_feature(
        set_id="user-1160c",
        category_name=semantic_category.name,
        feature="observer_fix",
        value="Fixed observer subagent bug",
        tag="bugfix",
        embedding=np.array([1.0, 0.0]),
    )
    id2 = await semantic_storage.add_feature(
        set_id="user-1160c",
        category_name=semantic_category.name,
        feature="step_disable",
        value="Steps 7 and 8 disabled",
        tag="bugfix",
        embedding=np.array([0.0, 1.0]),
    )
    await semantic_storage.add_citations(id1, [hist1])
    await semantic_storage.add_citations(id2, [hist2])

    filter_str = (
        f"set_id IN ('user-1160c') AND category_name IN ('{semantic_category.name}')"
    )
    memories = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
            load_citations=True,
        )
    )

    # LLM tries to replace user tag "bugfix" with "Productivity Style"
    consolidated = LLMReducedFeature(
        tag="Productivity Style",
        feature="development_practices",
        value="User fixes bugs and disables untested steps",
    )
    llm_mock = AsyncMock(
        return_value=SemanticConsolidateMemoryRes(
            consolidated_memories=[consolidated],
            keep_memories=[],
        ),
    )
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_mock,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-1160c",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
    )

    remaining = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
        )
    )
    bugfix_features = [f for f in remaining if f.tag == "bugfix"]
    renamed_features = [f for f in remaining if f.tag == "Productivity Style"]

    assert len(bugfix_features) == 1, "Original 'bugfix' tag must be preserved"
    assert len(renamed_features) == 0, "LLM-invented tag must be rejected"


@pytest.mark.asyncio
async def test_user_tags_preserved_after_ingestion_and_consolidation(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    embedder_double: MockEmbedder,
    semantic_category: SemanticCategory,
    mock_llm_model,
    monkeypatch,
):
    """End-to-end: user-defined tags survive consolidation even when the
    LLM invents new tag names.

    1. Pre-populate features with user tags (bugfix, decision, progress).
    2. Ingest a message that adds another "progress" entry.
    3. Consolidation fires (threshold=2) on the "progress" group.
    4. LLM tries to replace "progress" with "Productivity Style".
    5. Tag validation reverts to "progress" — all user tags survive."""
    set_id = "user-1160-e2e"

    await semantic_storage.add_feature(
        set_id=set_id,
        category_name=semantic_category.name,
        feature="observer_fix",
        value="Fixed observer subagent bug",
        tag="bugfix",
        embedding=np.array([1.0, 0.0]),
    )
    await semantic_storage.add_feature(
        set_id=set_id,
        category_name=semantic_category.name,
        feature="more_agents",
        value="User decided to add more agents",
        tag="decision",
        embedding=np.array([0.5, 0.5]),
    )
    await semantic_storage.add_feature(
        set_id=set_id,
        category_name=semantic_category.name,
        feature="design_progress",
        value="User completed 50% of design",
        tag="progress",
        embedding=np.array([0.0, 1.0]),
    )

    message_id = await add_history(
        episode_storage, content="Completed 60% of design now"
    )
    await semantic_storage.add_history_to_set(set_id=set_id, history_id=message_id)

    # LLM update adds another progress entry (reaches threshold=2)
    update_commands = [
        SemanticCommand(
            command=SemanticCommandType.ADD,
            feature="design_progress_60",
            tag="progress",
            value="User completed 60% of design",
        ),
    ]
    update_mock = AsyncMock(return_value=update_commands)
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_feature_update",
        update_mock,
    )

    # Consolidation LLM tries to replace "progress" with "Productivity Style"
    consolidated = LLMReducedFeature(
        tag="Productivity Style",
        feature="design_progress",
        value="User has completed 60% of design",
    )
    consolidate_mock = AsyncMock(
        return_value=SemanticConsolidateMemoryRes(
            consolidated_memories=[consolidated],
            keep_memories=[],
        ),
    )
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        consolidate_mock,
    )

    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
        )
    )

    await ingestion_service._process_single_set(set_id)

    filter_str = (
        f"set_id IN ('{set_id}') AND category_name IN ('{semantic_category.name}')"
    )
    remaining = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
        )
    )
    remaining_tags = {f.tag for f in remaining}

    # Tag validation reverts LLM-invented tag back to "progress"
    assert "Productivity Style" not in remaining_tags, (
        "LLM-invented tag must be rejected"
    )
    assert "progress" in remaining_tags, (
        "Original 'progress' tag must survive consolidation"
    )

    # "bugfix" and "decision" survive (each had < 2 entries)
    assert "bugfix" in remaining_tags
    assert "decision" in remaining_tags


@pytest.mark.asyncio
async def test_consolidation_retries_on_none_response(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """When llm_consolidate_features returns None on the first attempt but
    succeeds on the second, consolidation completes and features are removed."""
    hist1 = await add_history(episode_storage, content="msg1")
    hist2 = await add_history(episode_storage, content="msg2")

    id1 = await semantic_storage.add_feature(
        set_id="user-retry-none",
        category_name=semantic_category.name,
        feature="feat_x",
        value="value x",
        tag="retry_tag",
        embedding=np.array([1.0, 0.0]),
    )
    id2 = await semantic_storage.add_feature(
        set_id="user-retry-none",
        category_name=semantic_category.name,
        feature="feat_y",
        value="value y",
        tag="retry_tag",
        embedding=np.array([0.0, 1.0]),
    )
    await semantic_storage.add_citations(id1, [hist1])
    await semantic_storage.add_citations(id2, [hist2])

    filter_str = f"set_id IN ('user-retry-none') AND category_name IN ('{semantic_category.name}')"
    memories = await _collect(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(filter_str),
            load_citations=True,
        )
    )

    valid_response = SemanticConsolidateMemoryRes(
        consolidated_memories=[
            LLMReducedFeature(tag="retry_tag", feature="combined", value="combined x+y")
        ],
        keep_memories=[],
    )
    llm_mock = AsyncMock(side_effect=[None, valid_response])
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_mock,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-retry-none",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
    )

    assert llm_mock.await_count == 2
    assert await semantic_storage.get_feature(id1) is None
    assert await semantic_storage.get_feature(id2) is None
    remaining = await _collect(
        semantic_storage.get_feature_set(filter_expr=parse_filter(filter_str))
    )
    assert len(remaining) == 1
    assert remaining[0].value == "combined x+y"


@pytest.mark.asyncio
async def test_consolidation_exhausts_retries_and_returns(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """When llm_consolidate_features always returns None, _deduplicate_features
    logs a warning and returns without raising and without touching features."""
    id1 = await semantic_storage.add_feature(
        set_id="user-retry-exhaust",
        category_name=semantic_category.name,
        feature="feat_a",
        value="value a",
        tag="exhaust_tag",
        embedding=np.array([1.0, 0.0]),
    )
    id2 = await semantic_storage.add_feature(
        set_id="user-retry-exhaust",
        category_name=semantic_category.name,
        feature="feat_b",
        value="value b",
        tag="exhaust_tag",
        embedding=np.array([0.0, 1.0]),
    )

    filter_str = f"set_id IN ('user-retry-exhaust') AND category_name IN ('{semantic_category.name}')"
    memories = await _collect(
        semantic_storage.get_feature_set(filter_expr=parse_filter(filter_str))
    )

    llm_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_mock,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-retry-exhaust",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
    )

    assert llm_mock.await_count == 2
    assert await semantic_storage.get_feature(id1) is not None
    assert await semantic_storage.get_feature(id2) is not None


@pytest.mark.asyncio
async def test_consolidation_skips_context_length_error(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """A context-length exception during consolidation is handled gracefully:
    no exception propagates and features are left unchanged."""

    class ContextLengthExceededError(Exception):
        code = "context_length_exceeded"

    id1 = await semantic_storage.add_feature(
        set_id="user-consolidate-ctx",
        category_name=semantic_category.name,
        feature="feat_p",
        value="value p",
        tag="ctx_tag",
        embedding=np.array([1.0, 0.0]),
    )
    id2 = await semantic_storage.add_feature(
        set_id="user-consolidate-ctx",
        category_name=semantic_category.name,
        feature="feat_q",
        value="value q",
        tag="ctx_tag",
        embedding=np.array([0.0, 1.0]),
    )

    filter_str = f"set_id IN ('user-consolidate-ctx') AND category_name IN ('{semantic_category.name}')"
    memories = await _collect(
        semantic_storage.get_feature_set(filter_expr=parse_filter(filter_str))
    )

    async def raise_ctx_error(*args, **kwargs):
        raise ExternalServiceAPIError(
            "context window exceeded"
        ) from ContextLengthExceededError("input exceeds context window")

    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        raise_ctx_error,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-consolidate-ctx",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
    )

    assert await semantic_storage.get_feature(id1) is not None
    assert await semantic_storage.get_feature(id2) is not None


@pytest.mark.asyncio
async def test_consolidation_retries_on_exception(
    ingestion_service: IngestionService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resources: Resources,
    semantic_category: SemanticCategory,
    semantic_prompt,
    monkeypatch,
):
    """A non-context exception on the first call must not short-circuit the
    loop — the helper retries once and proceeds with the successful response."""
    id1 = await semantic_storage.add_feature(
        set_id="user-retry-exc",
        category_name=semantic_category.name,
        feature="feat_a",
        value="value a",
        tag="exc_tag",
        embedding=np.array([1.0, 0.0]),
    )
    await semantic_storage.add_feature(
        set_id="user-retry-exc",
        category_name=semantic_category.name,
        feature="feat_b",
        value="value b",
        tag="exc_tag",
        embedding=np.array([0.0, 1.0]),
    )

    valid_response = SemanticConsolidateMemoryRes(
        consolidated_memories=[
            LLMReducedFeature(tag="exc_tag", feature="combined", value="combined a+b")
        ],
        keep_memories=[id1],
    )

    async def flaky(*args, **kwargs):
        if llm_mock.await_count == 1:
            raise ValueError("structured-output validation failed")
        return valid_response

    llm_mock = AsyncMock(side_effect=flaky)
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_mock,
    )

    result = await ingestion_service._try_consolidate(
        set_id="user-retry-exc",
        features=[],
        model=resources.language_model,
        consolidation_prompt=semantic_prompt.consolidation_prompt,
    )

    assert llm_mock.await_count == 2
    assert result is valid_response


@pytest.mark.asyncio
async def test_consolidation_exhausted_exceptions_respects_debug_fail_loudly(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    resources: Resources,
    semantic_category: SemanticCategory,
    semantic_prompt,
    monkeypatch,
):
    """When every attempt raises a non-context exception, _try_consolidate
    re-raises under debug_fail_loudly only after the retry is exhausted."""
    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            debug_fail_loudly=True,
        )
    )

    async def always_raise(*args, **kwargs):
        raise ValueError("structured-output validation failed")

    llm_mock = AsyncMock(side_effect=always_raise)
    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_mock,
    )

    with pytest.raises(ValueError, match="structured-output validation failed"):
        await ingestion_service._try_consolidate(
            set_id="user-exc-loud",
            features=[],
            model=resources.language_model,
            consolidation_prompt=semantic_prompt.consolidation_prompt,
        )

    # Retry must still happen before the loud re-raise.
    assert llm_mock.await_count == 2


@pytest.mark.asyncio
async def test_deduplicate_features_context_length_does_not_raise_in_debug_mode(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
    resources: Resources,
    semantic_category: SemanticCategory,
    monkeypatch,
):
    """Context-length errors during consolidation must be handled gracefully
    even when debug_fail_loudly is True, matching the feature-update path."""

    class ContextLengthExceededError(Exception):
        code = "context_length_exceeded"

    ingestion_service = IngestionService(
        IngestionService.Params(
            semantic_storage=semantic_storage,
            history_store=episode_storage,
            resource_retriever=resource_retriever.get_resources,
            consolidated_threshold=2,
            debug_fail_loudly=True,
        )
    )

    id1 = await semantic_storage.add_feature(
        set_id="user-ctx-loud",
        category_name=semantic_category.name,
        feature="feat_p",
        value="value p",
        tag="ctx_tag",
        embedding=np.array([1.0, 0.0]),
    )
    id2 = await semantic_storage.add_feature(
        set_id="user-ctx-loud",
        category_name=semantic_category.name,
        feature="feat_q",
        value="value q",
        tag="ctx_tag",
        embedding=np.array([0.0, 1.0]),
    )

    filter_str = (
        f"set_id IN ('user-ctx-loud') AND category_name IN ('{semantic_category.name}')"
    )
    memories = await _collect(
        semantic_storage.get_feature_set(filter_expr=parse_filter(filter_str))
    )

    async def raise_ctx_error(*args, **kwargs):
        raise ExternalServiceAPIError(
            "context window exceeded"
        ) from ContextLengthExceededError("input exceeds context window")

    monkeypatch.setattr(
        "memmachine_server.semantic_memory.semantic_ingestion.llm_consolidate_features",
        raise_ctx_error,
    )

    # Must not raise even though debug_fail_loudly=True.
    await ingestion_service._deduplicate_features(
        set_id="user-ctx-loud",
        memories=memories,
        semantic_category=semantic_category,
        resources=resources,
    )

    assert await semantic_storage.get_feature(id1) is not None
    assert await semantic_storage.get_feature(id2) is not None
