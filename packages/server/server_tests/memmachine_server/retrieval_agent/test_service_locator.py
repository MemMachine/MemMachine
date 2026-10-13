import pytest
from pydantic import ValidationError

from memmachine_server.common.configuration.retrieval_config import (
    ProgressiveRetrievalConf,
    RetrievalAgentConf,
)
from memmachine_server.retrieval_agent.agents import (
    ChainOfQueryAgent,
    ProgressiveQueryAgent,
    RaragQueryAgent,
    ToolSelectAgent,
)
from memmachine_server.retrieval_agent.service_locator import create_retrieval_agent

from .test_retrieval_agent import DummyLanguageModel, DummyReranker


@pytest.mark.parametrize("optimized", [False, True])
def test_existing_strategy_selection_is_unchanged(optimized):
    agent = create_retrieval_agent(
        model=DummyLanguageModel("unused"),
        reranker=DummyReranker(),
        use_optimized_coq=optimized,
    )
    assert isinstance(agent, ToolSelectAgent)
    coq = next(
        child
        for child in agent.agent_tools()
        if child.agent_name == "ChainOfQueryAgent"
    )
    assert type(coq) is (RaragQueryAgent if optimized else ChainOfQueryAgent)


def test_progressive_is_selected_without_model_routing():
    agent = create_retrieval_agent(
        model=DummyLanguageModel("unused"),
        reranker=DummyReranker(),
        progressive=ProgressiveRetrievalConf(),
    )
    assert isinstance(agent, ProgressiveQueryAgent)


@pytest.mark.parametrize(
    "settings",
    [
        {"initial_limit": 0},
        {"max_candidates": -1},
        {"max_rounds": 0},
        {"max_rounds": True},
        {"max_candidates": "20"},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": float("inf")},
    ],
)
def test_invalid_progressive_budgets_are_rejected(settings):
    with pytest.raises(ValidationError):
        ProgressiveRetrievalConf.model_validate(settings)


def test_progressive_configuration_round_trips_without_changing_defaults():
    default = RetrievalAgentConf()
    assert default.progressive is None
    assert "progressive" not in default.to_yaml_dict()
    configured = RetrievalAgentConf(progressive=ProgressiveRetrievalConf(max_rounds=4))
    assert RetrievalAgentConf.model_validate(configured.to_yaml_dict()) == configured


@pytest.mark.parametrize("explicit_name", [False, True])
def test_conflicting_opt_in_strategies_are_rejected(explicit_name):
    with pytest.raises(ValidationError, match="cannot both be enabled"):
        RetrievalAgentConf(
            use_optimized_coq=True, progressive=ProgressiveRetrievalConf()
        )
    with pytest.raises(ValueError, match="another retrieval strategy"):
        create_retrieval_agent(
            model=DummyLanguageModel("unused"),
            reranker=DummyReranker(),
            agent_name="ProgressiveQueryAgent" if explicit_name else "ToolSelectAgent",
            use_optimized_coq=True,
            progressive=None if explicit_name else ProgressiveRetrievalConf(),
        )
