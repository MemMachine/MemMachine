"""Tests for SemanticService background ingestion functionality."""

import asyncio

import numpy as np
import pytest

from memmachine.common.episode_store import EpisodeStorage
from memmachine.common.filter.filter_parser import parse_filter
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    SemanticCommandType,
)
from tests.memmachine.semantic_memory.semantic_test_utils import (
    SpyEmbedder,
    add_history,
)
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    SemanticStorage,
)


async def _collect_feature_set(storage: SemanticStorage, **kwargs):
    return [item async for item in storage.get_feature_set(**kwargs)]


pytestmark = pytest.mark.asyncio


async def test_service_starts_and_stops_cleanly(
    semantic_service: SemanticService,
):
    # When starting the service
    await semantic_service.start()

    # Then the ingestion task is created
    assert semantic_service._ingestion_task is not None
    assert not semantic_service._is_shutting_down

    # When stopping the service
    await semantic_service.stop()

    # Then the shutdown flag is set
    assert semantic_service._is_shutting_down


async def test_start_idempotent(
    semantic_service: SemanticService,
):
    # Given a semantic service

    # When starting the service multiple times
    await semantic_service.start()
    first_task = semantic_service._ingestion_task
    await semantic_service.start()
    second_task = semantic_service._ingestion_task

    # Then the same task is reused
    assert first_task is second_task

    await semantic_service.stop()


async def test_stop_when_not_started(
    semantic_service: SemanticService,
):
    # Given a semantic service that has not been started
    # When stopping the service
    await semantic_service.stop()

    # Then no error occurs
    assert semantic_service._ingestion_task is None


async def test_background_ingestion_processes_messages_on_message_limit(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category_retriever,
    semantic_config_storage: SemanticConfigStorage,
    semantic_resource_manager,
    mock_llm_model,
    spy_embedder: SpyEmbedder,
    in_memory_cluster_state_storage,
    monkeypatch,
):
    semantic_type = semantic_category_retriever("test")[0]

    from memmachine.semantic_memory.semantic_model import SemanticCommand

    params = SemanticService.Params(
        feature_update_interval_sec=0.05,
        uningested_message_limit=0,
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        semantic_config_storage=semantic_config_storage,
        cluster_state_storage=in_memory_cluster_state_storage,
        resource_manager=semantic_resource_manager,
        default_embedder=spy_embedder,
        default_embedder_name="default_embedder",
        default_language_model=mock_llm_model,
        default_category_retriever=semantic_category_retriever,
    )
    service = SemanticService(params)
    await service.start()

    commands = [
        SemanticCommand(
            command=SemanticCommandType.ADD,
            feature="favorite_color",
            tag="preferences",
            value="blue",
        ),
    ]

    async def mock_llm_update(*args, **kwargs):
        return commands

    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        mock_llm_update,
    )

    msg1 = await add_history(history_storage=episode_storage, content="I like blue")
    await service.add_messages(set_id="user-123", history_ids=[msg1])

    msg2 = await add_history(
        history_storage=episode_storage,
        content="Blue is my favorite",
    )
    await service.add_messages(set_id="user-123", history_ids=[msg2])

    await asyncio.sleep(0.6)

    uningested = await semantic_storage.get_history_messages_count(
        set_ids=["user-123"],
        is_ingested=False,
    )

    filter_str = f"set_id IN ('user-123') AND category_name IN ('{semantic_type.name}')"
    features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
    )

    await service.stop()

    assert uningested == 0, f"Expected 0 uningested messages, got {uningested}"
    assert len(features) > 0, f"Expected features to be created, got {len(features)}"


async def test_background_ingestion_handles_errors_gracefully(
    semantic_service: SemanticService,
    episode_storage: EpisodeStorage,
    monkeypatch,
):
    async def mock_llm_error(*args, **kwargs):
        raise ValueError("LLM processing error")

    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        mock_llm_error,
    )

    msg1 = await add_history(history_storage=episode_storage, content="Test message 1")
    msg2 = await add_history(history_storage=episode_storage, content="Test message 2")
    await semantic_service.add_messages(set_id="user-error", history_ids=[msg1, msg2])

    # Wait for background processing
    await asyncio.sleep(0.2)

    # Service should still be running despite error
    assert semantic_service._ingestion_task is not None
    assert not semantic_service._ingestion_task.done()


async def test_consolidation_threshold_not_reached(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category_retriever,
):
    semantic_type = semantic_category_retriever("test")[0]

    # Given a service with high consolidation threshold

    # Add a few features (less than threshold)
    await semantic_storage.add_feature(
        set_id="user-consolidate",
        category_name=semantic_type.name,
        feature="color1",
        value="red",
        tag="colors",
        embedding=np.array([1.0, 0.0]),
    )
    await semantic_storage.add_feature(
        set_id="user-consolidate",
        category_name=semantic_type.name,
        feature="color2",
        value="blue",
        tag="colors",
        embedding=np.array([0.0, 1.0]),
    )

    # Get initial count
    filter_str = (
        f"set_id IN ('user-consolidate') AND category_name IN ('{semantic_type.name}')"
    )
    features_before = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
    )
    count_before = len(features_before)

    # Add messages to trigger background processing
    msg_id = await add_history(history_storage=episode_storage, content="I like colors")
    await semantic_service.add_messages(set_id="user-consolidate", history_ids=[msg_id])

    # Wait for processing
    await asyncio.sleep(0.2)

    # Features should not be consolidated
    features_after = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_str),
    )

    # Should have same or more features (no consolidation)
    assert len(features_after) >= count_before


async def test_multiple_sets_processed_independently(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_category_retriever,
    semantic_config_storage: SemanticConfigStorage,
    semantic_resource_manager,
    mock_llm_model,
    spy_embedder: SpyEmbedder,
    in_memory_cluster_state_storage,
    monkeypatch,
):
    semantic_type = semantic_category_retriever("test")[0]

    from memmachine.semantic_memory.semantic_model import SemanticCommand

    params = SemanticService.Params(
        feature_update_interval_sec=0.05,
        uningested_message_limit=0,
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        semantic_config_storage=semantic_config_storage,
        cluster_state_storage=in_memory_cluster_state_storage,
        resource_manager=semantic_resource_manager,
        default_embedder=spy_embedder,
        default_embedder_name="default_embedder",
        default_language_model=mock_llm_model,
        default_category_retriever=semantic_category_retriever,
    )
    service = SemanticService(params)
    await service.start()

    async def mock_llm_update(*args, **kwargs):
        message = kwargs.get("message_content", "")
        if "user-a" in str(message):
            return [
                SemanticCommand(
                    command=SemanticCommandType.ADD,
                    feature="trait_a",
                    tag="traits",
                    value="value_a",
                ),
            ]
        return [
            SemanticCommand(
                command=SemanticCommandType.ADD,
                feature="trait_b",
                tag="traits",
                value="value_b",
            ),
        ]

    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        mock_llm_update,
    )

    msg_a1 = await add_history(
        history_storage=episode_storage,
        content="user-a message 1",
    )
    await service.add_messages(set_id="user-a", history_ids=[msg_a1])
    msg_a2 = await add_history(
        history_storage=episode_storage,
        content="user-a message 2",
    )
    await service.add_messages(set_id="user-a", history_ids=[msg_a2])

    msg_b1 = await add_history(
        history_storage=episode_storage,
        content="user-b message 1",
    )
    await service.add_messages(set_id="user-b", history_ids=[msg_b1])
    msg_b2 = await add_history(
        history_storage=episode_storage,
        content="user-b message 2",
    )
    await service.add_messages(set_id="user-b", history_ids=[msg_b2])

    # Wait for background processing
    async def wait_for_ingestion():
        while (  # noqa: ASYNC110
            await semantic_storage.get_history_messages_count(
                set_ids=["user-a"],
                is_ingested=False,
            )
            != 0
            or await semantic_storage.get_history_messages_count(
                set_ids=["user-b"],
                is_ingested=False,
            )
            != 0
        ):
            await asyncio.sleep(0.1)

    await asyncio.wait_for(wait_for_ingestion(), timeout=15)

    # Verify that messages were ingested for both sets
    uningested_a = await semantic_storage.get_history_messages_count(
        set_ids=["user-a"],
        is_ingested=False,
    )
    uningested_b = await semantic_storage.get_history_messages_count(
        set_ids=["user-b"],
        is_ingested=False,
    )

    # Verify both sets were processed independently
    filter_a = f"set_id IN ('user-a') AND category_name IN ('{semantic_type.name}')"
    filter_b = f"set_id IN ('user-b') AND category_name IN ('{semantic_type.name}')"
    features_a = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_a),
    )
    features_b = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(filter_b),
    )

    await service.stop()

    # Both sets should have been fully processed
    assert uningested_a == 0, (
        f"Expected user-a to have 0 uningested, got {uningested_a}"
    )
    assert uningested_b == 0, (
        f"Expected user-b to have 0 uningested, got {uningested_b}"
    )
    assert len(features_a) > 0, (
        f"Expected user-a to have features, got {len(features_a)}"
    )
    assert len(features_b) > 0, (
        f"Expected user-b to have features, got {len(features_b)}"
    )
