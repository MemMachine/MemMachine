from unittest.mock import MagicMock

import pytest
from memmachine_server.common.configuration.retrieval_config import (
    ProgressiveRetrievalConf,
)
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.reranker.reranker import Reranker

from evaluation.utils.agent_utils import init_agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_name", "expected"),
    [
        ("ToolSelectAgent", "ProgressiveQueryAgent"),
        ("ProgressiveQueryAgent", "ProgressiveQueryAgent"),
        ("MemMachineAgent", "MemMachineAgent"),
        ("ChainOfQueryAgent", "ChainOfQueryAgent"),
    ],
)
async def test_progressive_configuration_preserves_explicit_baselines(
    agent_name, expected
):
    agent = await init_agent(
        model=MagicMock(spec=LanguageModel),
        reranker=MagicMock(spec=Reranker),
        agent_name=agent_name,
        progressive=ProgressiveRetrievalConf(),
    )
    assert agent.agent_name == expected
