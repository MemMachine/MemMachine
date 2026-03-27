from __future__ import annotations

from typing import Any

from memmachine_server.retrieval_agent.common.agent_api import (
    QueryParam,
    QueryPolicy,
    RetrievalAgentProtocol,
    RetrievalAgentResult,
)


class DemoAgent:
    @property
    def agent_name(self) -> str:
        return "DemoAgent"

    @property
    def agent_description(self) -> str:
        return "demo"

    @property
    def accuracy_score(self) -> int:
        return 1

    @property
    def token_cost(self) -> int:
        return 1

    @property
    def time_cost(self) -> int:
        return 1

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[RetrievalAgentResult, dict[str, Any]]:
        _ = policy, query
        return RetrievalAgentResult(), {}


def test_retrieval_agent_protocol_uses_agent_naming() -> None:
    agent: RetrievalAgentProtocol = DemoAgent()
    assert agent.agent_name == "DemoAgent"
    assert agent.agent_description == "demo"
