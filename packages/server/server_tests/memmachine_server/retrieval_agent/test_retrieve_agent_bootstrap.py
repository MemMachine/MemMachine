from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memmachine_server.common.episode_store import Episode, EpisodeResponse
from memmachine_server.common.language_model import (
    SkillLanguageModelError as AgentLanguageModelError,
)
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_agent.agents.retrieve_agent import RetrievalAgent
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryParam,
    QueryPolicy,
    RetrievalAgentParams,
)
from memmachine_server.retrieval_agent.service_locator import create_retrieval_agent
from server_tests.memmachine_server.retrieval_agent.provider_runner_stub import (
    FakeOpenAIInstalledAgentModel,
    openai_text_response,
)


class DummyReranker(Reranker):
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        _ = query
        return [float(len(candidates) - idx) for idx in range(len(candidates))]


class FakeEpisodicMemory(EpisodicMemory):
    def __init__(self, episodes_by_query: dict[str, list[Episode]]) -> None:
        self._episodes_by_query = episodes_by_query
        self._session_key = "test-session"

    async def query_memory(
        self,
        query: str,
        *,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        property_filter: Any | None = None,
        mode: EpisodicMemory.QueryMode = EpisodicMemory.QueryMode.BOTH,
    ) -> EpisodicMemory.QueryResponse | None:
        _ = expand_context, score_threshold, property_filter, mode
        episodes = self._episodes_by_query.get(query, [])
        search_limit = limit if limit is not None else len(episodes)
        return EpisodicMemory.QueryResponse(
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[
                    EpisodeResponse(score=1.0, **episode.model_dump())
                    for episode in episodes[:search_limit]
                ]
            ),
            short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[],
                episode_summary=[],
            ),
        )


@pytest.fixture
def query_policy() -> QueryPolicy:
    return QueryPolicy(
        token_cost=0,
        time_cost=0,
        accuracy_score=0.0,
        confidence_score=0.0,
    )


def _build_episode(uid: str = "ep-1") -> Episode:
    return Episode(
        uid=uid,
        content="hello",
        session_key="test-session",
        created_at=datetime.now(tz=UTC),
        producer_id="test",
        producer_role="assistant",
    )


def _build_skill(
    model: FakeOpenAIInstalledAgentModel,
    *,
    tmp_path,
    **extra_params: object,
) -> RetrievalAgent:
    return RetrievalAgent(
        RetrievalAgentParams(
            model=model,
            extra_params={
                "agent_install_cache_path": tmp_path / ".agent-cache.json",
                **extra_params,
            },
            reranker=DummyReranker(),
        ),
    )


@pytest.mark.asyncio
async def test_retrieve_agent_bootstrap_entry_path(query_policy: QueryPolicy) -> None:
    episode = _build_episode()
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel([openai_text_response("direct-memory")])
    agent = create_retrieval_agent(
        model=model,
        reranker=DummyReranker(),
    )

    result, metrics = await agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["route"] == "RetrievalAgent"
    assert metrics["orchestrator_tool_call_count"] >= 1


@pytest.mark.asyncio
async def test_retrieve_agent_bootstrap_fallback_reason_for_errors(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode(uid="fallback")
    memory = FakeEpisodicMemory({"hello": [episode]})
    retrieve_agent = _build_skill(
        FakeOpenAIInstalledAgentModel([RuntimeError("forced top-level model failure")]),
        tmp_path=tmp_path,
    )

    result, metrics = await retrieve_agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["route"] == "RetrievalAgent"
    assert metrics["fallback_trigger_reason"] == "downstream_tool_failure"
    assert metrics["agent_contract_error_code"] == "AGENT_CONTRACT_DOWNSTREAM_FAILURE"
    error_diagnostics = metrics.get("error_diagnostics")
    assert isinstance(error_diagnostics, dict)
    assert error_diagnostics.get("context") == "top_level_unhandled_exception"
    assert error_diagnostics.get("error_type") == "RuntimeError"
    assert isinstance(metrics.get("agent_contract_error_payload"), dict)


@pytest.mark.asyncio
async def test_retrieve_agent_bootstrap_invalid_entry_uses_fallback(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode(uid="empty-query")
    memory = FakeEpisodicMemory({"": [episode]})
    model = FakeOpenAIInstalledAgentModel([openai_text_response("direct-memory")])
    agent = create_retrieval_agent(
        model=model,
        reranker=DummyReranker(),
    )

    result, metrics = await agent.do_query(
        query_policy,
        QueryParam(query="", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["route"] == "RetrievalAgent"
    assert metrics["fallback_trigger_reason"] == "invalid_agent_request"
    assert metrics["agent_contract_error_code"] == "AGENT_CONTRACT_INVALID_REQUEST"


@pytest.mark.asyncio
async def test_retrieve_agent_attaches_all_skill_bundles_on_session_start(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode(uid="bundle-check")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel([openai_text_response("no-tool-calls")])
    agent = _build_skill(model, tmp_path=tmp_path)

    _, _ = await agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [call["file_name"] for call in model.client.files.calls] == [
        "retrieve_agent.md",
        "coq.md",
    ]
    assert len(model.client.responses.calls) == 1
    first_call = model.client.responses.calls[0]
    first_input = first_call["input"]
    assert isinstance(first_input, list)
    user_message = first_input[0]
    assert isinstance(user_message, dict)
    content = user_message["content"]
    assert content[:2] == [
        {"type": "input_file", "file_id": "file-1"},
        {"type": "input_file", "file_id": "file-2"},
    ]
    assert content[2]["type"] == "input_text"
    assert "hello" in str(content[2]["text"])


@pytest.mark.asyncio
async def test_retrieve_agent_fallback_records_provider_raw_error_response(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode(uid="provider-error")
    memory = FakeEpisodicMemory({"hello": [episode]})
    retrieve_agent = _build_skill(
        FakeOpenAIInstalledAgentModel(
            [
                AgentLanguageModelError(
                    "forced provider failure",
                    diagnostics={
                        "provider": "openai",
                        "operation": "responses.create",
                        "status_code": 400,
                        "response_body": '{"error":"bad request"}',
                    },
                )
            ]
        ),
        tmp_path=tmp_path,
    )

    result, metrics = await retrieve_agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["fallback_trigger_reason"] == "downstream_tool_failure"
    assert metrics["agent_contract_error_code"] == "AGENT_CONTRACT_DOWNSTREAM_FAILURE"
    assert metrics["provider_error_raw_response"] == '{"error":"bad request"}'
