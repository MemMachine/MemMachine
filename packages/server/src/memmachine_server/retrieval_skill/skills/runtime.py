"""Runtime helpers for strict skill contract validation."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from memmachine_server.retrieval_skill.common.skill_api import QueryParam
from memmachine_server.retrieval_skill.skills.types import (
    SKILL_CONTRACT_VERSION_V1,
    SkillContractError,
    SkillContractErrorCode,
    SkillContractErrorPayload,
    SkillRequestV1,
    SkillResultV1,
)

SkillResultNormalizer = Callable[[object], object]


def build_skill_request(query: QueryParam, *, route_name: str) -> SkillRequestV1:
    """Build and validate a strict v1 skill request from retrieval query inputs."""
    try:
        return SkillRequestV1(
            version=SKILL_CONTRACT_VERSION_V1,
            route_name=route_name,
            query=query.query,
            limit=query.limit,
            expand_context=query.expand_context,
            score_threshold=query.score_threshold,
        )
    except ValidationError as err:
        raise SkillContractError(
            code=SkillContractErrorCode.INVALID_REQUEST,
            payload=SkillContractErrorPayload(
                what_failed="Skill request validation failed",
                why=str(err),
                how_to_fix="Provide required request fields with valid values.",
                where="skills.runtime.build_skill_request",
                fallback_trigger_reason="invalid_skill_request",
            ),
            validation_error=err,
        ) from err


def _normalize_result_from_tuple(raw_result: object, *, route_name: str) -> object:
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        episodes, perf_metrics = raw_result
        return {
            "version": SKILL_CONTRACT_VERSION_V1,
            "route_name": route_name,
            "episodes": episodes,
            "perf_metrics": perf_metrics,
            "fallback_trigger_reason": None,
        }
    return raw_result


def validate_skill_result(
    raw_result: object,
    *,
    route_name: str,
    normalizer: SkillResultNormalizer | None = None,
) -> SkillResultV1:
    """Validate output against v1 result contract with a single normalization pass."""
    try:
        return SkillResultV1.model_validate(raw_result)
    except ValidationError as first_err:
        if normalizer is None:
            raise SkillContractError(
                code=SkillContractErrorCode.INVALID_OUTPUT,
                payload=SkillContractErrorPayload(
                    what_failed="Skill result validation failed",
                    why=str(first_err),
                    how_to_fix="Return a strict v1 skill result object.",
                    where="skills.runtime.validate_skill_result",
                    fallback_trigger_reason="invalid_skill_output",
                ),
                validation_error=first_err,
            ) from first_err

    try:
        normalized_result = normalizer(raw_result)
    except Exception as err:
        raise SkillContractError(
            code=SkillContractErrorCode.NORMALIZATION_FAILED,
            payload=SkillContractErrorPayload(
                what_failed="Skill output normalization failed",
                why=str(err),
                how_to_fix=(
                    "Fix the normalization function to produce a valid "
                    "SkillResultV1 payload."
                ),
                where="skills.runtime.validate_skill_result",
                fallback_trigger_reason="normalization_exception",
            ),
        ) from err

    normalized_result = _normalize_result_from_tuple(
        normalized_result, route_name=route_name
    )
    try:
        return SkillResultV1.model_validate(normalized_result)
    except ValidationError as second_err:
        raise SkillContractError(
            code=SkillContractErrorCode.INVALID_OUTPUT,
            payload=SkillContractErrorPayload(
                what_failed="Skill result remained invalid after normalization",
                why=str(second_err),
                how_to_fix=(
                    "Ensure the single normalization pass returns a strict v1 "
                    "result object."
                ),
                where="skills.runtime.validate_skill_result",
                fallback_trigger_reason="invalid_after_normalization",
            ),
            validation_error=second_err,
        ) from second_err


def fallback_for_downstream_error(*, where: str, error: Exception) -> SkillContractError:
    """Map downstream execution failures to a stable skill-contract error."""
    return SkillContractError(
        code=SkillContractErrorCode.DOWNSTREAM_FAILURE,
        payload=SkillContractErrorPayload(
            what_failed="Skill execution failed in downstream tool",
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
    "SkillResultNormalizer",
    "build_skill_request",
    "fallback_for_downstream_error",
    "validate_skill_result",
]
