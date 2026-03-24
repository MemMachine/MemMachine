"""Tool protocol contracts for memmachine_search-only retrieval sessions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from memmachine_server.retrieval_agent.agents.types import (
    AgentContractError,
    AgentContractErrorCode,
    AgentContractErrorPayload,
)

TOP_LEVEL_TOOL_NAMES = ("memmachine_search",)
CANONICAL_SUB_AGENT_NAMES = ("coq",)
SUB_AGENT_TOOL_NAMES = ("memmachine_search",)


class TopLevelToolAction(BaseModel):
    """Validated top-level tool action emitted by the live session."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["memmachine_search"]
    query: str | None = None
    rationale: str = ""


class SubAgentToolAction(BaseModel):
    """Validated sub-agent action emitted by the live session."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["memmachine_search"]
    query: str | None = None


def _memmachine_search_schema() -> dict[str, object]:
    return {
        "type": "function",
        "name": "memmachine_search",
        "description": "Search MemMachine memory with the provided query.",
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


def top_level_tool_schemas(
    allowed_tools: list[str],
    available_sub_agents: list[str] | None = None,
) -> list[dict[str, object]]:
    """Return function-call schemas for the top-level retrieval session."""
    _ = available_sub_agents
    names = set(allowed_tools) if allowed_tools else set(TOP_LEVEL_TOOL_NAMES)
    return [_memmachine_search_schema()] if "memmachine_search" in names else []


def sub_agent_tool_schemas(allowed_tools: list[str]) -> list[dict[str, object]]:
    """Return function-call schemas for sub-agent live sessions."""
    names = set(allowed_tools) if allowed_tools else set(SUB_AGENT_TOOL_NAMES)
    return [_memmachine_search_schema()] if "memmachine_search" in names else []


def _invalid_tool_call_error(*, where: str, why: str) -> AgentContractError:
    return AgentContractError(
        code=AgentContractErrorCode.INVALID_OUTPUT,
        payload=AgentContractErrorPayload(
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
            where="agents.tool_protocol.parse_top_level_tool_call",
            why=f"Unsupported top-level tool name: {tool_name}",
        )

    payload = dict(arguments)
    payload["action"] = tool_name
    try:
        return TopLevelToolAction.model_validate(payload)
    except ValidationError as err:
        raise _invalid_tool_call_error(
            where="agents.tool_protocol.parse_top_level_tool_call",
            why=str(err),
        ) from err


def parse_sub_agent_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, object],
) -> SubAgentToolAction:
    """Validate and normalize a sub-agent tool call."""
    if tool_name not in SUB_AGENT_TOOL_NAMES:
        raise _invalid_tool_call_error(
            where="agents.tool_protocol.parse_sub_agent_tool_call",
            why=f"Unsupported sub-agent tool name: {tool_name}",
        )

    payload = dict(arguments)
    payload["action"] = tool_name
    try:
        return SubAgentToolAction.model_validate(payload)
    except ValidationError as err:
        raise _invalid_tool_call_error(
            where="agents.tool_protocol.parse_sub_agent_tool_call",
            why=str(err),
        ) from err
