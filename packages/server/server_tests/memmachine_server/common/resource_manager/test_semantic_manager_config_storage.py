from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memmachine_server.common.configuration import PromptConf, SemanticMemoryConf
from memmachine_server.common.errors import ResourceNotReadyError
from memmachine_server.common.resource_manager.semantic_manager import (
    SemanticResourceManager,
)


@pytest.mark.asyncio
async def test_get_semantic_config_storage_requires_config_database():
    resource_manager = MagicMock()
    manager = SemanticResourceManager(
        semantic_conf=SemanticMemoryConf(enabled=False),
        prompt_conf=PromptConf(),
        resource_manager=resource_manager,
        episode_storage=MagicMock(),
    )

    with pytest.raises(ResourceNotReadyError):
        await manager.get_semantic_config_storage()

    resource_manager.get_sql_engine.assert_not_called()
