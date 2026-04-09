"""Factory helpers for retrieval-agent construction."""

from memmachine_server.common.configuration.retrieval_config import RetrievalAgentConf
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.reranker import Reranker
from memmachine_server.retrieval_agent.agents.retrieve_agent import RetrievalAgent
from memmachine_server.retrieval_agent.common.agent_api import (
    RetrievalAgentParams,
    RetrievalAgentProtocol,
)


def create_retrieval_agent(
    *,
    model: LanguageModel,
    reranker: Reranker,
    retrieval_conf: RetrievalAgentConf | None = None,
    agent_name: str = "RetrievalAgent",
) -> RetrievalAgentProtocol:
    """Create the top-level retrieval agent."""
    if agent_name != "RetrievalAgent":
        raise ValueError(
            "create_retrieval_agent only supports agent_name='RetrievalAgent'. "
            f"Received: {agent_name!r}"
        )

    retrieve_extra_params: dict[str, object] = {}
    if retrieval_conf is not None:
        retrieve_extra_params["retrieval_conf"] = retrieval_conf
        retrieve_extra_params["global_timeout_seconds"] = (
            retrieval_conf.agent_session_timeout_seconds
        )
        retrieve_extra_params["max_combined_calls"] = (
            retrieval_conf.agent_session_max_combined_calls
        )

    retrieve_agent = RetrievalAgent(
        RetrievalAgentParams(
            model=model,
            extra_params=retrieve_extra_params,
            reranker=reranker,
        )
    )
    return retrieve_agent
