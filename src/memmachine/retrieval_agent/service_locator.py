"""Factory helpers for retrieval-agent construction."""

from memmachine.common.language_model import LanguageModel
from memmachine.common.reranker import Reranker
from memmachine.retrieval_agent.agents import (
    ChainOfQueryAgent,
    MemMachineAgent,
    SplitQueryAgent,
    ToolSelectAgent,
)
from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
)


def create_retrieval_agent(
    *,
    model: LanguageModel,
    reranker: Reranker,
    agent_name: str = "ToolSelectAgent",
) -> AgentToolBase:
    """Create the configured retrieval-agent strategy."""
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )
    if agent_name == memory_agent.agent_name:
        return memory_agent

    shared_param = AgentToolBaseParam(
        model=model,
        children_tools=[memory_agent],
        extra_params={},
        reranker=reranker,
    )

    coq_agent = ChainOfQueryAgent(shared_param)
    split_agent = SplitQueryAgent(shared_param)

    if agent_name == coq_agent.agent_name:
        return coq_agent
    if agent_name == split_agent.agent_name:
        return split_agent

    return ToolSelectAgent(
        AgentToolBaseParam(
            model=model,
            children_tools=[split_agent, coq_agent, memory_agent],
            extra_params={"default_tool_name": coq_agent.agent_name},
            reranker=reranker,
        ),
    )
