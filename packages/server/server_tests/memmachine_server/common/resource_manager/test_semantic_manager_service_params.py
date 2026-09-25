from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.configuration import PromptConf, SemanticMemoryConf
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeStorage
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.resource_manager.semantic_manager import (
    SemanticResourceManager,
)
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.semantic_memory import (
    ResourceManager,
    SemanticService,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage


@pytest.mark.asyncio
async def test_semantic_manager_wires_ingestion_settings_into_semantic_service(
    monkeypatch,
):
    """ingestion_poll_interval_seconds and consolidation_threshold must reach
    SemanticService.Params instead of silently falling back to defaults."""
    resource_manager = MagicMock(spec=ResourceManager)
    manager = SemanticResourceManager(
        semantic_conf=SemanticMemoryConf(
            database="db",
            config_database="config_db",
            llm_model="llm",
            embedding_model="embedder",
            ingestion_poll_interval_seconds=42.0,
            consolidation_threshold=7,
        ),
        prompt_conf=PromptConf(),
        resource_manager=resource_manager,
        episode_storage=MagicMock(spec=EpisodeStorage),
    )

    monkeypatch.setattr(
        manager,
        "get_semantic_storage",
        AsyncMock(return_value=MagicMock(spec=SemanticStorage)),
    )
    monkeypatch.setattr(
        manager,
        "get_semantic_config_storage",
        AsyncMock(return_value=MagicMock(spec=SemanticConfigStorage)),
    )
    monkeypatch.setattr(
        manager,
        "_get_default_embedder",
        AsyncMock(return_value=MagicMock(spec=Embedder)),
    )
    monkeypatch.setattr(
        manager,
        "_get_default_language_model",
        AsyncMock(return_value=MagicMock(spec=LanguageModel)),
    )

    captured_params = {}

    class _FakeSemanticService:
        # Reuse the real Params class (not a hand-rolled stand-in) so a
        # future rename/drop of a field makes this test fail loudly instead
        # of silently passing kwargs through.
        Params = SemanticService.Params

        def __init__(self, params):
            captured_params["params"] = params

    monkeypatch.setattr(
        "memmachine_server.common.resource_manager.semantic_manager.SemanticService",
        _FakeSemanticService,
    )

    await manager.get_semantic_service()

    params = captured_params["params"]
    assert params.feature_update_interval_sec == 42.0
    assert params.consolidation_threshold == 7
