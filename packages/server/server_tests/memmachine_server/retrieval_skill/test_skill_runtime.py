from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memmachine_server.common.episode_store import Episode
from memmachine_server.retrieval_skill.skills.runtime import validate_skill_result
from memmachine_server.retrieval_skill.skills.spec_loader import load_skill_spec
from memmachine_server.retrieval_skill.skills.tool_protocol import (
    sub_skill_tool_schemas,
    top_level_tool_schemas,
)
from memmachine_server.retrieval_skill.skills.types import (
    SkillContractError,
    SkillContractErrorCode,
    SkillRequestV1,
    SkillResultV1,
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
        SkillRequestV1.model_validate(
            {
                "version": "v2",
                "route_name": "retrieve-skill",
                "query": "hello",
                "limit": 1,
                "expand_context": 0,
                "score_threshold": 0.0,
            }
        )


def test_contract_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SkillResultV1.model_validate(
            {
                "version": "v1",
                "route_name": "retrieve-skill",
                "episodes": [_build_episode()],
                "perf_metrics": {},
                "unknown": "not-allowed",
            }
        )


def test_contract_schema_load_skill_spec_rejects_unknown_fields() -> None:
    with pytest.raises(SkillContractError) as exc_info:
        load_skill_spec(
            {
                "name": "retrieve-skill",
                "version": "v1",
                "description": "bootstrap",
                "unexpected": True,
            }
        )

    assert exc_info.value.code == SkillContractErrorCode.INVALID_SPEC.value


def test_contract_schema_load_skill_spec_requires_fields() -> None:
    with pytest.raises(SkillContractError) as exc_info:
        load_skill_spec({"name": "retrieve-skill", "version": "v1"})

    assert exc_info.value.code == SkillContractErrorCode.INVALID_SPEC.value


def test_normalize_result_single_attempt_succeeds() -> None:
    normalizer_calls = {"count": 0}
    episode = _build_episode()

    def normalizer(_raw_result: object) -> object:
        normalizer_calls["count"] += 1
        return ([episode], {"selected_skill_name": "MemMachineSkill"})

    result = validate_skill_result(
        raw_result={"not": "a-result"},
        route_name="retrieve-skill",
        normalizer=normalizer,
    )

    assert normalizer_calls["count"] == 1
    assert result.version == "v1"
    assert result.route_name == "retrieve-skill"
    assert result.episodes == [episode]
    assert result.perf_metrics["selected_skill_name"] == "MemMachineSkill"


def test_invalid_result_after_normalize_raises_stable_error() -> None:
    normalizer_calls = {"count": 0}

    def normalizer(_raw_result: object) -> object:
        normalizer_calls["count"] += 1
        return {"version": "v1", "route_name": "retrieve-skill", "episodes": []}

    with pytest.raises(SkillContractError) as exc_info:
        validate_skill_result(
            raw_result={"not": "a-result"},
            route_name="retrieve-skill",
            normalizer=normalizer,
        )

    assert normalizer_calls["count"] == 1
    assert exc_info.value.code == SkillContractErrorCode.INVALID_OUTPUT.value
    assert (
        exc_info.value.payload.fallback_trigger_reason == "invalid_after_normalization"
    )


def test_invalid_result_without_normalizer_raises_error_code() -> None:
    with pytest.raises(SkillContractError) as exc_info:
        validate_skill_result(
            raw_result={"not": "a-result"},
            route_name="retrieve-skill",
            normalizer=None,
        )

    assert exc_info.value.code == SkillContractErrorCode.INVALID_OUTPUT.value
    assert exc_info.value.payload.fallback_trigger_reason == "invalid_skill_output"


def test_top_level_spawn_sub_skill_schema_uses_enum_constraints() -> None:
    schemas = top_level_tool_schemas(["spawn_sub_skill"], ["coq"])
    assert len(schemas) == 1
    spawn_schema = schemas[0]
    properties = spawn_schema["parameters"]["properties"]
    skill_name_schema = properties["skill_name"]
    assert skill_name_schema["enum"] == ["coq"]


def test_sub_skill_return_schema_supports_structured_fields() -> None:
    schemas = sub_skill_tool_schemas(["return_sub_skill_result"])
    assert len(schemas) == 1
    return_schema = schemas[0]
    properties = return_schema["parameters"]["properties"]
    assert "summary" in properties
    assert "is_sufficient" in properties
    assert "new_query" in properties
    assert "answer_candidate" in properties
    assert "generated_sub_queries" in properties
