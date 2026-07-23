"""Factory helpers for retrieval-agent construction."""

from typing import Any

from memmachine_server.common.configuration.retrieval_config import OptimizedCoqConf
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.reranker import Reranker
from memmachine_server.retrieval_agent.agents import (
    ChainOfQueryAgent,
    MemMachineAgent,
    RaragQueryAgent,
    SplitQueryAgent,
    ToolSelectAgent,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
)


def create_retrieval_agent(
    *,
    model: LanguageModel,
    reranker: Reranker,
    agent_name: str = "ToolSelectAgent",
    use_optimized_coq: bool = False,
    optimized_coq: OptimizedCoqConf | None = None,
) -> AgentToolBase:
    """Create the configured retrieval-agent strategy.

    Args:
        model: Language model for agent operations.
        reranker: Reranker for result re-ranking.
        agent_name: Name of the agent to create.
        use_optimized_coq: If true, the ChainOfQueryAgent slot is filled by
            RaragQueryAgent, an optimized multi-hop retrieval variant. When false
            or unset, the original ChainOfQueryAgent is used.
        optimized_coq: Settings forwarded to RaragQueryAgent when
            use_optimized_coq is true (hop-splitting strategy and per-sub-search
            limit). Ignored otherwise. Defaults are applied when None.
    """
    optimized_coq = optimized_coq or OptimizedCoqConf()
    # Hop-splitting settings consumed by RaragQueryAgent (and SplitQueryAgent)
    # via extra_params, mirroring the evaluation path in agent_utils.init_agent.
    optimized_extra_params: dict[str, Any] = {
        "multi_hop_decomposer": bool(optimized_coq.multi_hop_decomposer),
        "multi_hop_sub_limit": int(optimized_coq.multi_hop_sub_limit),
    }

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

    # extra_params carrying the multi-hop settings reach the CoQ agent and the
    # split agent when they are constructed from this shared param.
    shared_param = AgentToolBaseParam(
        model=model,
        children_tools=[memory_agent],
        extra_params=optimized_extra_params,
        reranker=reranker,
    )

    # Use RaragQueryAgent (optimized variant) or ChainOfQueryAgent based on config.
    coq_agent = (
        RaragQueryAgent(shared_param)
        if use_optimized_coq
        else ChainOfQueryAgent(shared_param)
    )
    split_agent = SplitQueryAgent(shared_param)

    if agent_name == coq_agent.agent_name:
        return coq_agent
    if agent_name == split_agent.agent_name:
        return split_agent

    # For ToolSelectAgent, include the configured coq variant in children.
    children = [split_agent, coq_agent, memory_agent]
    return ToolSelectAgent(
        AgentToolBaseParam(
            model=model,
            children_tools=children,
            extra_params={
                "default_tool_name": coq_agent.agent_name,
                **optimized_extra_params,
            },
            reranker=reranker,
        ),
    )
