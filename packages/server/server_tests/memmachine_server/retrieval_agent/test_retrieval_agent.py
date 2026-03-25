from __future__ import annotations

from typing import Any

import pytest

from memmachine_server.common.configuration.retrieval_config import (
    RetrievalAgentConf,
    RetrievalAgentSessionProvider,
)
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.retrieval_agent.service_locator import create_retrieval_agent


class DummyLanguageModel(LanguageModel):
    """Lightweight language model stub for unit tests."""

    def __init__(self, responses: list[str] | str) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = responses
        self.call_count = 0

    async def generate_parsed_response(
        self,
        output_format: type[Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any | None:
        return None

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        return "", None

    async def generate_response_with_token_usage(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Any, int, int]:
        idx = min(self.call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self.call_count += 1
        return response, None, 1, 1


class DummyReranker(Reranker):
    """Reranker stub returning predefined scores."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores or []
        self.call_count = 0

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        self.call_count += 1
        if self._scores and len(self._scores) == len(candidates):
            return list(self._scores)
        return [float(len(candidates) - idx) for idx in range(len(candidates))]


def test_service_locator_defaults_to_retrieve_agent() -> None:
    model = DummyLanguageModel("direct-memory")
    agent = create_retrieval_agent(
        model=model,
        reranker=DummyReranker(),
    )
    assert agent.agent_name == "RetrievalAgent"


def test_service_locator_accepts_generic_language_model_without_session_model() -> None:
    agent = create_retrieval_agent(
        model=DummyLanguageModel("direct-memory"),
        reranker=DummyReranker(),
    )
    assert agent.agent_name == "RetrievalAgent"


def test_service_locator_rejects_legacy_agent_routes() -> None:
    model = DummyLanguageModel("direct-memory")
    reranker = DummyReranker()
    for agent_name in [
        "MemMachineSkill",
        "ChainOfQueryAgent",
        "ToolSelectSkill",
    ]:
        with pytest.raises(
            ValueError, match="only supports agent_name='RetrievalAgent'"
        ):
            _ = create_retrieval_agent(
                model=model,
                reranker=reranker,
                agent_name=agent_name,
            )

    agent = create_retrieval_agent(
        model=model,
        reranker=reranker,
        agent_name="RetrievalAgent",
    )
    assert agent.agent_name == "RetrievalAgent"


def test_service_locator_applies_retrieval_conf_budgets() -> None:
    model = DummyLanguageModel("direct-memory")
    reranker = DummyReranker()
    conf = RetrievalAgentConf(
        agent_session_provider=RetrievalAgentSessionProvider.ANTHROPIC,
        anthropic_api_key="anthropic-key",
        agent_session_timeout_seconds=180,
        agent_session_max_combined_calls=10,
    )

    agent = create_retrieval_agent(
        model=model,
        reranker=reranker,
        retrieval_conf=conf,
    )

    assert agent.agent_name == "RetrievalAgent"
    assert agent._global_timeout_seconds == 180
    assert agent._max_combined_calls == 10
    assert agent._available_sub_agents == ["coq"]
