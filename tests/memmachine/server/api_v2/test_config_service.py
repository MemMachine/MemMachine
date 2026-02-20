"""Tests for the configuration service."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from memmachine.common.api.config_spec import (
    ResourceStatus,
    UpdateEpisodicMemorySpec,
    UpdateSemanticMemorySpec,
)
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.server.api_v2.config_service import ConfigService


@pytest.fixture
def mock_resource_manager():
    """Create a mock resource manager."""
    resource_manager = MagicMock()

    # Mock config with file path set
    mock_config = MagicMock()
    mock_config.config_file_path = "/tmp/test_config.yml"
    resource_manager.config = mock_config
    resource_manager.save_config = MagicMock()

    # Mock embedder manager
    embedder_manager = MagicMock()
    embedder_manager.add_embedder_config = MagicMock()
    embedder_manager.remove_embedder = MagicMock(return_value=True)
    resource_manager.embedder_manager = embedder_manager

    # Mock language model manager
    lm_manager = MagicMock()
    lm_manager.add_language_model_config = MagicMock()
    lm_manager.remove_language_model = MagicMock(return_value=True)
    resource_manager.language_model_manager = lm_manager

    return resource_manager


@pytest.fixture
def mock_resource_manager_no_file():
    """Create a mock resource manager with no config file path."""
    resource_manager = MagicMock()

    # Mock config without file path
    mock_config = MagicMock()
    mock_config.config_file_path = None
    resource_manager.config = mock_config
    resource_manager.save_config = MagicMock()

    # Mock embedder manager
    embedder_manager = MagicMock()
    embedder_manager.add_embedder_config = MagicMock()
    embedder_manager.remove_embedder = MagicMock(return_value=True)
    resource_manager.embedder_manager = embedder_manager

    # Mock language model manager
    lm_manager = MagicMock()
    lm_manager.add_language_model_config = MagicMock()
    lm_manager.remove_language_model = MagicMock(return_value=True)
    resource_manager.language_model_manager = lm_manager

    return resource_manager


def test_persist_config_saves_when_file_path_set(mock_resource_manager):
    """Test that _persist_config calls save_config when file path is set."""
    service = ConfigService(mock_resource_manager)
    service._persist_config()
    mock_resource_manager.save_config.assert_called_once()


def test_persist_config_skips_when_no_file_path(mock_resource_manager_no_file):
    """Test that _persist_config does not save when no file path is set."""
    service = ConfigService(mock_resource_manager_no_file)
    service._persist_config()
    mock_resource_manager_no_file.save_config.assert_not_called()


def test_persist_config_handles_save_error(mock_resource_manager):
    """Test that _persist_config handles save errors gracefully."""
    mock_resource_manager.save_config.side_effect = Exception("Write error")

    service = ConfigService(mock_resource_manager)
    # Should not raise, just log warning
    service._persist_config()
    mock_resource_manager.save_config.assert_called_once()


@pytest.mark.asyncio
async def test_add_embedder_persists_config(mock_resource_manager):
    """Test that add_embedder persists configuration."""
    # Mock successful embedder creation (get_embedder is async)
    mock_resource_manager.embedder_manager.get_embedder = AsyncMock(return_value=None)

    service = ConfigService(mock_resource_manager)
    config = {"api_key": "test-key", "model": "text-embedding-3-small"}
    status = await service.add_embedder("test-embedder", "openai", config)

    assert status == ResourceStatus.READY
    mock_resource_manager.save_config.assert_called_once()


@pytest.mark.asyncio
async def test_add_embedder_persists_even_on_build_failure(mock_resource_manager):
    """Test that add_embedder persists configuration even when build fails."""
    # Mock failed embedder creation (get_embedder is async)
    mock_resource_manager.embedder_manager.get_embedder = AsyncMock(
        side_effect=Exception("Build failed")
    )

    service = ConfigService(mock_resource_manager)
    config = {"api_key": "test-key", "model": "text-embedding-3-small"}
    status = await service.add_embedder("test-embedder", "openai", config)

    assert status == ResourceStatus.FAILED
    # Config should still be persisted (configuration is added before build attempt)
    mock_resource_manager.save_config.assert_called_once()


@pytest.mark.asyncio
async def test_add_language_model_persists_config(mock_resource_manager):
    """Test that add_language_model persists configuration."""
    # Mock successful model creation (get_language_model is async)
    mock_resource_manager.language_model_manager.get_language_model = AsyncMock(
        return_value=None
    )

    service = ConfigService(mock_resource_manager)
    config = {"api_key": "test-key", "model": "gpt-4"}
    status = await service.add_language_model("test-model", "openai-responses", config)

    assert status == ResourceStatus.READY
    mock_resource_manager.save_config.assert_called_once()


def test_remove_embedder_persists_config(mock_resource_manager):
    """Test that remove_embedder persists configuration when removal succeeds."""
    service = ConfigService(mock_resource_manager)
    removed = service.remove_embedder("test-embedder")

    assert removed is True
    mock_resource_manager.save_config.assert_called_once()


def test_remove_embedder_does_not_persist_when_not_found(mock_resource_manager):
    """Test that remove_embedder does not persist when embedder not found."""
    mock_resource_manager.embedder_manager.remove_embedder.return_value = False

    service = ConfigService(mock_resource_manager)
    removed = service.remove_embedder("nonexistent")

    assert removed is False
    mock_resource_manager.save_config.assert_not_called()


def test_remove_language_model_persists_config(mock_resource_manager):
    """Test that remove_language_model persists configuration when removal succeeds."""
    service = ConfigService(mock_resource_manager)
    removed = service.remove_language_model("test-model")

    assert removed is True
    mock_resource_manager.save_config.assert_called_once()


def test_remove_language_model_does_not_persist_when_not_found(mock_resource_manager):
    """Test that remove_language_model does not persist when model not found."""
    mock_resource_manager.language_model_manager.remove_language_model.return_value = (
        False
    )

    service = ConfigService(mock_resource_manager)
    removed = service.remove_language_model("nonexistent")

    assert removed is False
    mock_resource_manager.save_config.assert_not_called()


# --- update_memory_config tests ---


@pytest.fixture
def memory_resource_manager():
    """Create a resource manager with real config objects for memory tests."""
    resource_manager = MagicMock()
    resource_manager.config.config_file_path = "/tmp/test_config.yml"
    resource_manager.save_config = MagicMock()

    # Use real config objects so field assignment works
    resource_manager.config.episodic_memory = EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(
            embedder="old-embedder",
            reranker="old-reranker",
            vector_graph_store="old-store",
        ),
        short_term_memory=ShortTermMemoryConfPartial(
            llm_model="old-model",
            message_capacity=500,
        ),
        long_term_memory_enabled=True,
        short_term_memory_enabled=True,
        enabled=True,
    )

    semantic_memory = MagicMock()
    semantic_memory.database = "old-db"
    semantic_memory.llm_model = "old-llm"
    semantic_memory.embedding_model = "old-embedder"
    semantic_memory.ingestion_trigger_messages = 5
    semantic_memory.ingestion_trigger_age = timedelta(minutes=5)
    resource_manager.config.semantic_memory = semantic_memory

    return resource_manager


def test_update_episodic_ltm_embedder(memory_resource_manager):
    """Test updating the long-term memory embedder."""
    spec = UpdateEpisodicMemorySpec.model_validate(
        {"long_term_memory": {"embedder": "new-embedder"}}
    )
    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(spec, None)

    assert "long_term_memory.embedder=new-embedder" in message
    em = memory_resource_manager.config.episodic_memory
    assert em.long_term_memory.embedder == "new-embedder"
    # Unchanged fields stay the same
    assert em.long_term_memory.reranker == "old-reranker"
    memory_resource_manager.save_config.assert_called_once()


def test_update_episodic_ltm_multiple_fields(memory_resource_manager):
    """Test updating multiple long-term memory fields."""
    spec = UpdateEpisodicMemorySpec.model_validate(
        {
            "long_term_memory": {
                "embedder": "new-embedder",
                "reranker": "new-reranker",
                "vector_graph_store": "new-store",
            }
        }
    )
    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(spec, None)

    assert "embedder=new-embedder" in message
    assert "reranker=new-reranker" in message
    assert "vector_graph_store=new-store" in message

    ltm = memory_resource_manager.config.episodic_memory.long_term_memory
    assert ltm.embedder == "new-embedder"
    assert ltm.reranker == "new-reranker"
    assert ltm.vector_graph_store == "new-store"


def test_update_episodic_stm_llm_model(memory_resource_manager):
    """Test updating the short-term memory LLM model."""
    spec = UpdateEpisodicMemorySpec.model_validate(
        {"short_term_memory": {"llm_model": "new-llm"}}
    )
    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(spec, None)

    assert "llm_model=new-llm" in message
    stm = memory_resource_manager.config.episodic_memory.short_term_memory
    assert stm.llm_model == "new-llm"
    assert stm.message_capacity == 500  # unchanged


def test_update_episodic_enabled_flags(memory_resource_manager):
    """Test updating episodic memory enabled flags."""
    spec = UpdateEpisodicMemorySpec.model_validate(
        {
            "long_term_memory_enabled": False,
            "short_term_memory_enabled": False,
            "enabled": False,
        }
    )
    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(spec, None)

    em = memory_resource_manager.config.episodic_memory
    assert em.long_term_memory_enabled is False
    assert em.short_term_memory_enabled is False
    assert em.enabled is False
    assert "long_term_memory_enabled=False" in message


def test_update_semantic_memory_fields(memory_resource_manager):
    """Test updating semantic memory fields."""
    spec = UpdateSemanticMemorySpec.model_validate(
        {
            "database": "new-db",
            "llm_model": "new-llm",
            "embedding_model": "new-embedder",
        }
    )
    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(None, spec)

    sm = memory_resource_manager.config.semantic_memory
    assert sm.database == "new-db"
    assert sm.llm_model == "new-llm"
    assert sm.embedding_model == "new-embedder"
    assert "database=new-db" in message
    memory_resource_manager.save_config.assert_called_once()


def test_update_semantic_ingestion_settings(memory_resource_manager):
    """Test updating semantic memory ingestion settings."""
    spec = UpdateSemanticMemorySpec.model_validate(
        {"ingestion_trigger_messages": 10, "ingestion_trigger_age_seconds": 600}
    )
    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(None, spec)

    sm = memory_resource_manager.config.semantic_memory
    assert sm.ingestion_trigger_messages == 10
    assert sm.ingestion_trigger_age == timedelta(seconds=600)
    assert "ingestion_trigger_messages=10" in message
    assert "ingestion_trigger_age=600s" in message


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_update_semantic_memory_spec_rejects_invalid_cluster_threshold(value):
    with pytest.raises(ValidationError):
        UpdateSemanticMemorySpec.model_validate({"cluster_similarity_threshold": value})


@pytest.mark.parametrize("value", [0, -1])
def test_update_semantic_memory_spec_rejects_invalid_cluster_time_gap(value):
    with pytest.raises(ValidationError):
        UpdateSemanticMemorySpec.model_validate({"cluster_max_time_gap_seconds": value})


def test_update_both_episodic_and_semantic(memory_resource_manager):
    """Test updating both memory sections in one call."""
    episodic_spec = UpdateEpisodicMemorySpec.model_validate(
        {"long_term_memory": {"embedder": "new-embedder"}}
    )
    semantic_spec = UpdateSemanticMemorySpec.model_validate({"llm_model": "new-llm"})

    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(episodic_spec, semantic_spec)

    assert "long_term_memory.embedder=new-embedder" in message
    assert "llm_model=new-llm" in message
    memory_resource_manager.save_config.assert_called_once()


def test_update_no_changes(memory_resource_manager):
    """Test that no-op updates return appropriate message."""
    # All fields are None (no actual changes)
    episodic_spec = UpdateEpisodicMemorySpec.model_validate({})
    service = ConfigService(memory_resource_manager)
    message = service.update_memory_config(episodic_spec, None)

    assert message == "No changes applied."
    memory_resource_manager.save_config.assert_not_called()


def test_update_ltm_creates_partial_when_none(memory_resource_manager):
    """Test that LTM update creates a partial config when none exists."""
    memory_resource_manager.config.episodic_memory = EpisodicMemoryConfPartial()

    spec = UpdateEpisodicMemorySpec.model_validate(
        {"long_term_memory": {"embedder": "brand-new-embedder"}}
    )
    service = ConfigService(memory_resource_manager)
    service.update_memory_config(spec, None)

    ltm = memory_resource_manager.config.episodic_memory.long_term_memory
    assert ltm is not None
    assert ltm.embedder == "brand-new-embedder"


def test_update_stm_creates_partial_when_none(memory_resource_manager):
    """Test that STM update creates a partial config when none exists."""
    memory_resource_manager.config.episodic_memory = EpisodicMemoryConfPartial()

    spec = UpdateEpisodicMemorySpec.model_validate(
        {"short_term_memory": {"llm_model": "brand-new-model"}}
    )
    service = ConfigService(memory_resource_manager)
    service.update_memory_config(spec, None)

    stm = memory_resource_manager.config.episodic_memory.short_term_memory
    assert stm is not None
    assert stm.llm_model == "brand-new-model"
