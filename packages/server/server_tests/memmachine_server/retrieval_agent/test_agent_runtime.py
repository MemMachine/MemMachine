from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memmachine_server.common.episode_store import Episode
from memmachine_server.retrieval_agent.agents.runtime import validate_agent_result
from memmachine_server.retrieval_agent.agents.spec_loader import load_agent_spec
from memmachine_server.retrieval_agent.agents.tool_protocol import (
    parse_sub_agent_tool_call,
    parse_top_level_tool_call,
    sub_agent_tool_schemas,
    top_level_tool_schemas,
)
from memmachine_server.retrieval_agent.agents.types import (
    AgentContractError,
    AgentContractErrorCode,
    AgentRequestV1,
    AgentResultV1,
)


def _build_episode(uid: str = "ep-1") -> Episode:
    return Episode(
        uid=uid,
        content="hello",
        session_key="s1",
        created_at=datetime.now(tz=UTC),
        producer_id="test",
        producer_role="assistant",
    )


def test_contract_schema_requires_v1_version() -> None:
    with pytest.raises(ValidationError):
        AgentRequestV1.model_validate(
            {
                "version": "v2",
                "route_name": "retrieve-agent",
                "query": "hello",
                "limit": 1,
                "expand_context": 0,
                "score_threshold": 0.0,
            }
        )


def test_contract_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentResultV1.model_validate(
            {
                "version": "v1",
                "route_name": "retrieve-agent",
                "episodes": [_build_episode()],
                "perf_metrics": {},
                "unknown": "not-allowed",
            }
        )


def test_contract_schema_load_agent_spec_rejects_unknown_fields() -> None:
    with pytest.raises(AgentContractError) as exc_info:
        load_agent_spec(
            {
                "name": "retrieve-agent",
                "version": "v1",
                "description": "bootstrap",
                "unexpected": True,
            }
        )

    assert exc_info.value.code == AgentContractErrorCode.INVALID_SPEC.value


def test_contract_schema_load_agent_spec_requires_fields() -> None:
    with pytest.raises(AgentContractError) as exc_info:
        load_agent_spec({"name": "retrieve-agent", "version": "v1"})

    assert exc_info.value.code == AgentContractErrorCode.INVALID_SPEC.value


def test_normalize_result_single_attempt_succeeds() -> None:
    normalizer_calls = {"count": 0}
    episode = _build_episode()

    def normalizer(_raw_result: object) -> object:
        normalizer_calls["count"] += 1
        return ([episode], {"selected_agent_name": "DirectMemorySearch"})

    result = validate_agent_result(
        raw_result={"not": "a-result"},
        route_name="retrieve-agent",
        normalizer=normalizer,
    )

    assert normalizer_calls["count"] == 1
    assert result.version == "v1"
    assert result.route_name == "retrieve-agent"
    assert result.episodes == [episode]
    assert result.perf_metrics["selected_agent_name"] == "DirectMemorySearch"


def test_invalid_result_after_normalize_raises_stable_error() -> None:
    normalizer_calls = {"count": 0}

    def normalizer(_raw_result: object) -> object:
        normalizer_calls["count"] += 1
        return {"version": "v1", "route_name": "retrieve-agent", "episodes": []}

    with pytest.raises(AgentContractError) as exc_info:
        validate_agent_result(
            raw_result={"not": "a-result"},
            route_name="retrieve-agent",
            normalizer=normalizer,
        )

    assert normalizer_calls["count"] == 1
    assert exc_info.value.code == AgentContractErrorCode.INVALID_OUTPUT.value
    assert (
        exc_info.value.payload.fallback_trigger_reason == "invalid_after_normalization"
    )


def test_invalid_result_without_normalizer_raises_error_code() -> None:
    with pytest.raises(AgentContractError) as exc_info:
        validate_agent_result(
            raw_result={"not": "a-result"},
            route_name="retrieve-agent",
            normalizer=None,
        )

    assert exc_info.value.code == AgentContractErrorCode.INVALID_OUTPUT.value
    assert exc_info.value.payload.fallback_trigger_reason == "invalid_agent_output"


def test_top_level_schema_exposes_only_memmachine_search() -> None:
    schemas = top_level_tool_schemas(["memmachine_search"], ["coq"])
    assert len(schemas) == 1
    search_schema = schemas[0]
    assert search_schema["name"] == "memmachine_search"
    properties = search_schema["parameters"]["properties"]
    assert "query" in properties
    assert "rationale" in properties


def test_sub_agent_schema_exposes_only_memmachine_search() -> None:
    schemas = sub_agent_tool_schemas(["memmachine_search"])
    assert len(schemas) == 1
    assert schemas[0]["name"] == "memmachine_search"


def test_legacy_top_level_tool_names_are_rejected() -> None:
    with pytest.raises(AgentContractError) as exc_info:
        parse_top_level_tool_call(
            tool_name="spawn_sub_agent",
            arguments={"query": "hello"},
        )

    assert exc_info.value.payload.fallback_trigger_reason == "invalid_tool_call"


def test_legacy_sub_agent_tool_names_are_rejected() -> None:
    with pytest.raises(AgentContractError) as exc_info:
        parse_sub_agent_tool_call(
            tool_name="return_sub_agent_result",
            arguments={"summary": "hello"},
        )

    assert exc_info.value.payload.fallback_trigger_reason == "invalid_tool_call"
