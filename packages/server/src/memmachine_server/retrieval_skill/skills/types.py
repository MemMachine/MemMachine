"""Canonical contracts for skill-style retrieval runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memmachine_server.common.episode_store import Episode

SKILL_CONTRACT_VERSION_V1: Literal["v1"] = "v1"


class SkillContractErrorCode(StrEnum):
    """Stable error codes emitted by skill runtime contracts."""

    INVALID_REQUEST = "SKILL_CONTRACT_INVALID_REQUEST"
    INVALID_SPEC = "SKILL_CONTRACT_INVALID_SPEC"
    INVALID_OUTPUT = "SKILL_CONTRACT_INVALID_OUTPUT"
    NORMALIZATION_FAILED = "SKILL_CONTRACT_NORMALIZATION_FAILED"
    DOWNSTREAM_FAILURE = "SKILL_CONTRACT_DOWNSTREAM_FAILURE"


class SkillContractErrorPayload(BaseModel):
    """Developer-facing contract error metadata."""

    model_config = ConfigDict(extra="forbid")

    what_failed: str
    why: str
    how_to_fix: str
    where: str
    fallback_trigger_reason: str


class SkillContractError(RuntimeError):
    """Raised when a skill runtime contract is violated."""

    def __init__(
        self,
        *,
        code: SkillContractErrorCode,
        payload: SkillContractErrorPayload,
        validation_error: ValidationError | None = None,
    ) -> None:
        """Initialize with a stable code and developer-facing payload."""
        self.code = code.value
        self.payload = payload
        self.validation_error = validation_error
        super().__init__(
            f"{self.code}: {payload.what_failed}. Why: {payload.why}. "
            f"How to fix: {payload.how_to_fix}. Where: {payload.where}. "
            f"Fallback trigger reason: {payload.fallback_trigger_reason}."
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable error object for diagnostics."""
        return {"code": self.code, **self.payload.model_dump()}


class SkillSpecV1(BaseModel):
    """Strict v1 specification for skill bootstrap metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: Literal["v1"] = SKILL_CONTRACT_VERSION_V1
    kind: Literal["inline", "top-level", "sub-skill"] = "inline"
    description: str = Field(min_length=1)
    route_name: str = Field(default="retrieve-skill", min_length=1)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_return_len: int = Field(default=10000, ge=1)
    max_steps: int = Field(default=8, ge=1, le=50)
    fallback_hook: str = Field(default="direct-memory-search", min_length=1)
    allowed_actions: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    policy_markdown: str | None = None


class SkillRequestV1(BaseModel):
    """Strict v1 request contract at skill runtime boundaries."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"] = SKILL_CONTRACT_VERSION_V1
    route_name: str = Field(default="retrieve-skill", min_length=1)
    query: str = Field(min_length=1)
    limit: int = 0
    expand_context: int = 0
    score_threshold: float = -float("inf")


class SkillResultV1(BaseModel):
    """Strict v1 result contract for retrieval skill execution."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"] = SKILL_CONTRACT_VERSION_V1
    route_name: str = Field(default="retrieve-skill", min_length=1)
    episodes: list[Episode]
    perf_metrics: dict[str, object]
    fallback_trigger_reason: str | None = None


__all__ = [
    "SKILL_CONTRACT_VERSION_V1",
    "SkillContractError",
    "SkillContractErrorCode",
    "SkillContractErrorPayload",
    "SkillRequestV1",
    "SkillResultV1",
    "SkillSpecV1",
]
