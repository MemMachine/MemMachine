"""Fallback policy decisions for retrieve-skill reliability guardrails."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class FallbackTrigger(StrEnum):
    """Supported fallback trigger categories."""

    LOW_CONFIDENCE = "low_confidence_route"
    SUB_SKILL_EXCEPTION = "sub_skill_exception"
    SUB_SKILL_TIMEOUT = "sub_skill_timeout"
    GLOBAL_TIMEOUT = "global_timeout"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    MAX_HOPS_EXCEEDED = "max_hops_exceeded"
    MAX_BRANCHES_EXCEEDED = "max_branches_exceeded"


class FallbackPolicyDecision(BaseModel):
    """Policy decision for retry-vs-fallback behavior."""

    model_config = ConfigDict(extra="forbid")

    trigger: FallbackTrigger
    action: Literal["retry", "fallback"]
    fallback_trigger_reason: str


RETRYABLE_TRIGGERS: set[FallbackTrigger] = {
    FallbackTrigger.SUB_SKILL_TIMEOUT,
    FallbackTrigger.GLOBAL_TIMEOUT,
    FallbackTrigger.MAX_STEPS_EXCEEDED,
    FallbackTrigger.MAX_HOPS_EXCEEDED,
    FallbackTrigger.MAX_BRANCHES_EXCEEDED,
}


def is_retryable_trigger(trigger: FallbackTrigger) -> bool:
    """Return whether this trigger is eligible for controlled retry."""
    return trigger in RETRYABLE_TRIGGERS


def retry_budget_remaining(*, retry_count: int, max_retries: int) -> bool:
    """Return whether another retry can still be used."""
    return retry_count < max_retries


def fallback_reason_for(trigger: FallbackTrigger) -> str:
    """Return stable fallback reason string exposed in metrics payloads."""
    return trigger.value


def decide_fallback_action(
    *,
    trigger: FallbackTrigger,
    retry_count: int,
    max_retries: int = 1,
) -> FallbackPolicyDecision:
    """Return retry/fallback action for a trigger and current retry count."""
    should_retry = is_retryable_trigger(trigger) and retry_budget_remaining(
        retry_count=retry_count,
        max_retries=max_retries,
    )
    return FallbackPolicyDecision(
        trigger=trigger,
        action="retry" if should_retry else "fallback",
        fallback_trigger_reason=fallback_reason_for(trigger),
    )


__all__ = [
    "RETRYABLE_TRIGGERS",
    "FallbackPolicyDecision",
    "FallbackTrigger",
    "decide_fallback_action",
    "fallback_reason_for",
    "is_retryable_trigger",
    "retry_budget_remaining",
]
