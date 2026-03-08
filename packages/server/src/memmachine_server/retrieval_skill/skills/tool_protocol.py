"""Tool protocol contracts for markdown-driven retrieval orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from memmachine_server.retrieval_skill.skills.types import (
    SkillContractError,
    SkillContractErrorCode,
    SkillContractErrorPayload,
)

TOP_LEVEL_TOOL_NAMES = (
    "spawn_sub_skill",
    "direct_memory_search",
    "return_final",
)
CANONICAL_SUB_SKILL_NAMES = ("coq",)
SUB_SKILL_TOOL_NAMES = (
    "memmachine_search",
    "return_sub_skill_result",
)


class TopLevelToolAction(BaseModel):
    """Validated top-level tool action emitted by LLM tool calls."""

    # Allow extra fields so the runtime can ignore over-specified arguments
    # instead of hard-failing into fallback.
    model_config = ConfigDict(extra="ignore")

    action: Literal["spawn_sub_skill", "direct_memory_search", "return_final"]
    skill_name: str | None = None
    query: str | None = None
    final_response: str | None = None
    rationale: str = ""
    # Optional top-level sufficiency signal fields for metrics/evaluation traces.
    is_sufficient: bool | None = None
    confidence_score: float | None = None
    reason_code: str | None = None
    reason_note: str | None = None
    related_episode_indices: list[int] | None = None
    selected_episode_indices: list[int] | None = None
    stage_results: list[dict[str, object]] | None = None
    sub_queries: list[str] | None = None


class SubSkillToolAction(BaseModel):
    """Validated sub-skill tool action emitted by LLM tool calls."""

    # Permit optional structured summary fields so the runtime can consume
    # either legacy `summary` strings or first-class structured payload values.
    model_config = ConfigDict(extra="ignore")

    action: Literal["memmachine_search", "return_sub_skill_result"]
    query: str | None = None
    summary: str = ""
    is_sufficient: bool | None = None
    evidence_indices: list[int] | None = None
    new_query: str | None = None
    confidence_score: float | None = None
    reason_code: str | None = None
    reason_note: str | None = None
    answer_candidate: str | None = None
    stage_results: list[dict[str, object]] | None = None
    generated_sub_queries: list[str] | None = None
    sub_queries: list[str] | None = None
    related_episode_indices: list[int] | None = None
    selected_episode_indices: list[int] | None = None


def top_level_tool_schemas(
    allowed_tools: list[str],
    available_sub_skills: list[str] | None = None,
) -> list[dict[str, object]]:
    """Return function-call schemas for allowed top-level orchestration tools."""
    names = set(allowed_tools) if allowed_tools else set(TOP_LEVEL_TOOL_NAMES)
    sub_skills = available_sub_skills or list(CANONICAL_SUB_SKILL_NAMES)
    schemas: list[dict[str, object]] = []
    if "spawn_sub_skill" in names:
        schemas.append(
            {
                "type": "function",
                "name": "spawn_sub_skill",
                "description": "Spawn a sub-skill run with a query payload.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "enum": sub_skills,
                        },
                        "query": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["skill_name"],
                    "additionalProperties": False,
                },
            }
        )
    if "direct_memory_search" in names:
        schemas.append(
            {
                "type": "function",
                "name": "direct_memory_search",
                "description": "Run top-level direct MemMachine memory search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "return_final" in names:
        schemas.append(
            {
                "type": "function",
                "name": "return_final",
                "description": "Finalize top-level orchestration response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_response": {"type": "string"},
                        "rationale": {"type": "string"},
                        "is_sufficient": {"type": "boolean"},
                        "confidence_score": {"type": "number"},
                        "reason_code": {"type": "string"},
                        "reason_note": {"type": "string"},
                        "related_episode_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "selected_episode_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "stage_results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "stage_result": {"type": "string"},
                                    "confidence_score": {"type": "number"},
                                    "reason_note": {"type": "string"},
                                },
                                "required": ["query", "stage_result"],
                                "additionalProperties": True,
                            },
                        },
                        "sub_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    return schemas


def sub_skill_tool_schemas(allowed_tools: list[str]) -> list[dict[str, object]]:
    """Return function-call schemas for allowed sub-skill tools."""
    names = set(allowed_tools) if allowed_tools else set(SUB_SKILL_TOOL_NAMES)
    schemas: list[dict[str, object]] = []
    if "memmachine_search" in names:
        schemas.append(
            {
                "type": "function",
                "name": "memmachine_search",
                "description": "Search MemMachine memory with the provided query.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "return_sub_skill_result" in names:
        schemas.append(
            {
                "type": "function",
                "name": "return_sub_skill_result",
                "description": "Finalize sub-skill result payload.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "is_sufficient": {"type": "boolean"},
                        "evidence_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "new_query": {"type": "string"},
                        "confidence_score": {"type": "number"},
                        "reason_code": {"type": "string"},
                        "reason_note": {"type": "string"},
                        "answer_candidate": {"type": "string"},
                        "stage_results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "stage_result": {"type": "string"},
                                    "confidence_score": {"type": "number"},
                                    "reason_note": {"type": "string"},
                                },
                                "required": ["query", "stage_result"],
                                "additionalProperties": True,
                            },
                        },
                        "generated_sub_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "sub_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "related_episode_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "selected_episode_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    return schemas


def _invalid_tool_call_error(*, where: str, why: str) -> SkillContractError:
    return SkillContractError(
        code=SkillContractErrorCode.INVALID_OUTPUT,
        payload=SkillContractErrorPayload(
            what_failed="Tool call validation failed",
            why=why,
            how_to_fix="Emit only allowed function calls with valid arguments.",
            where=where,
            fallback_trigger_reason="invalid_tool_call",
        ),
    )


def parse_top_level_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, object],
) -> TopLevelToolAction:
    """Validate and normalize a top-level tool call."""
    if tool_name not in TOP_LEVEL_TOOL_NAMES:
        raise _invalid_tool_call_error(
            where="skills.tool_protocol.parse_top_level_tool_call",
            why=f"Unsupported top-level tool name: {tool_name}",
        )

    payload = dict(arguments)
    payload["action"] = tool_name
    try:
        return TopLevelToolAction.model_validate(payload)
    except ValidationError as err:
        raise _invalid_tool_call_error(
            where="skills.tool_protocol.parse_top_level_tool_call",
            why=str(err),
        ) from err


def parse_sub_skill_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, object],
) -> SubSkillToolAction:
    """Validate and normalize a sub-skill tool call."""
    if tool_name not in SUB_SKILL_TOOL_NAMES:
        raise _invalid_tool_call_error(
            where="skills.tool_protocol.parse_sub_skill_tool_call",
            why=f"Unsupported sub-skill tool name: {tool_name}",
        )

    payload = dict(arguments)
    payload["action"] = tool_name
    try:
        return SubSkillToolAction.model_validate(payload)
    except ValidationError as err:
        raise _invalid_tool_call_error(
            where="skills.tool_protocol.parse_sub_skill_tool_call",
            why=str(err),
        ) from err
