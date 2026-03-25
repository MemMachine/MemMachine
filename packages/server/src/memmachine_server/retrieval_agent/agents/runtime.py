"""Runtime helpers for strict retrieval-agent contract validation."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from memmachine_server.retrieval_agent.agents.types import (
    AGENT_CONTRACT_VERSION_V1,
    AgentContractError,
    AgentContractErrorCode,
    AgentContractErrorPayload,
    AgentRequestV1,
    AgentResultV1,
)
from memmachine_server.retrieval_agent.common.agent_api import QueryParam

AgentResultNormalizer = Callable[[object], object]


def build_agent_request(query: QueryParam, *, route_name: str) -> AgentRequestV1:
    """Build and validate a strict v1 agent request from retrieval query inputs."""
    try:
        return AgentRequestV1(
            version=AGENT_CONTRACT_VERSION_V1,
            route_name=route_name,
            query=query.query,
            limit=query.limit,
            expand_context=query.expand_context,
            score_threshold=query.score_threshold,
        )
    except ValidationError as err:
        raise AgentContractError(
            code=AgentContractErrorCode.INVALID_REQUEST,
            payload=AgentContractErrorPayload(
                what_failed="Agent request validation failed",
                why=str(err),
                how_to_fix="Provide required request fields with valid values.",
                where="agents.runtime.build_agent_request",
                fallback_trigger_reason="invalid_agent_request",
            ),
            validation_error=err,
        ) from err


def _normalize_result_from_tuple(raw_result: object, *, route_name: str) -> object:
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        retrieval_result, perf_metrics = raw_result
        normalized: dict[str, object] = {
            "version": AGENT_CONTRACT_VERSION_V1,
            "route_name": route_name,
            "perf_metrics": perf_metrics,
            "fallback_trigger_reason": None,
        }
        model_dump = getattr(retrieval_result, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="json")
            if isinstance(payload, dict):
                normalized.update(payload)
                return normalized
        if isinstance(retrieval_result, dict):
            normalized.update(retrieval_result)
            return normalized
    return raw_result


def validate_agent_result(
    raw_result: object,
    *,
    route_name: str,
    normalizer: AgentResultNormalizer | None = None,
) -> AgentResultV1:
    """Validate output against the v1 agent result contract."""
    try:
        return AgentResultV1.model_validate(raw_result)
    except ValidationError as first_err:
        if normalizer is None:
            raise AgentContractError(
                code=AgentContractErrorCode.INVALID_OUTPUT,
                payload=AgentContractErrorPayload(
                    what_failed="Agent result validation failed",
                    why=str(first_err),
                    how_to_fix="Return a strict v1 agent result object.",
                    where="agents.runtime.validate_agent_result",
                    fallback_trigger_reason="invalid_agent_output",
                ),
                validation_error=first_err,
            ) from first_err

    try:
        normalized_result = normalizer(raw_result)
    except Exception as err:
        raise AgentContractError(
            code=AgentContractErrorCode.NORMALIZATION_FAILED,
            payload=AgentContractErrorPayload(
                what_failed="Agent output normalization failed",
                why=str(err),
                how_to_fix=(
                    "Fix the normalization function to produce a valid "
                    "AgentResultV1 payload."
                ),
                where="agents.runtime.validate_agent_result",
                fallback_trigger_reason="normalization_exception",
            ),
        ) from err

    normalized_result = _normalize_result_from_tuple(
        normalized_result, route_name=route_name
    )
    try:
        return AgentResultV1.model_validate(normalized_result)
    except ValidationError as second_err:
        raise AgentContractError(
            code=AgentContractErrorCode.INVALID_OUTPUT,
            payload=AgentContractErrorPayload(
                what_failed="Agent result remained invalid after normalization",
                why=str(second_err),
                how_to_fix=(
                    "Ensure the single normalization pass returns a strict v1 "
                    "result object."
                ),
                where="agents.runtime.validate_agent_result",
                fallback_trigger_reason="invalid_after_normalization",
            ),
            validation_error=second_err,
        ) from second_err


def fallback_for_downstream_error(*, where: str, error: Exception) -> AgentContractError:
    """Map downstream execution failures to a stable retrieval-agent contract error."""
    return AgentContractError(
        code=AgentContractErrorCode.DOWNSTREAM_FAILURE,
        payload=AgentContractErrorPayload(
            what_failed="Agent execution failed in downstream tool",
            why=str(error),
            how_to_fix=(
                "Inspect downstream tool logs and return valid result "
                "contracts."
            ),
            where=where,
            fallback_trigger_reason="downstream_tool_failure",
        ),
    )


__all__ = [
    "AgentResultNormalizer",
    "build_agent_request",
    "fallback_for_downstream_error",
    "validate_agent_result",
]
