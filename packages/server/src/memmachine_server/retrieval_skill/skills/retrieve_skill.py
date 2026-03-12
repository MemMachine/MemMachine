"""Top-level markdown-guided retrieval orchestration runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.language_model import (
    ProviderSkillBundle,
    SkillLanguageModel,
    SkillLanguageModelError,
    SkillSessionLimitError,
    SkillSessionModelProtocol,
    SkillToolCallFormatError,
    SkillToolNotFoundError,
    materialize_provider_skill_bundle,
)
from memmachine_server.retrieval_skill.common.skill_api import (
    QueryParam,
    QueryPolicy,
    SkillToolBase,
    SkillToolBaseParam,
)
from memmachine_server.retrieval_skill.skills.runtime import (
    build_skill_request,
    fallback_for_downstream_error,
)
from memmachine_server.retrieval_skill.skills.session_state import (
    TopLevelSkillSessionState,
)
from memmachine_server.retrieval_skill.skills.spec_loader import load_skill_spec
from memmachine_server.retrieval_skill.skills.tool_protocol import (
    CANONICAL_SUB_SKILL_NAMES,
    TOP_LEVEL_TOOL_NAMES,
    parse_top_level_tool_call,
    top_level_tool_schemas,
)
from memmachine_server.retrieval_skill.skills.types import (
    SkillContractError,
    SkillContractErrorCode,
    SkillContractErrorPayload,
)

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVE_SKILL_SPEC = (
    Path(__file__).resolve().parent / "specs" / "top_level" / "retrieve_skill.md"
)
DEFAULT_SUB_SKILL_SPEC_ROOT = Path(__file__).resolve().parent / "specs" / "sub_skills"


class RetrieveSkill(SkillToolBase):
    """Top-level orchestrator that enforces markdown-driven action contracts."""

    def __init__(self, param: SkillToolBaseParam) -> None:
        """Initialize top-level markdown policy, memory tool, and sub-skill runner."""
        super().__init__(param)
        if self._model is None:
            raise ValueError("RetrieveSkill requires a language model.")

        self._extra_params = param.extra_params or {}
        raw_spec = self._extra_params.get("skill_spec", DEFAULT_RETRIEVE_SKILL_SPEC)
        self._spec = load_skill_spec(raw_spec)
        self._max_guardrail_retries = 1
        self._max_hops = int(self._extra_params.get("max_hops", 4))
        self._max_branches = int(self._extra_params.get("max_branches", 4))
        self._global_timeout_seconds = int(
            self._extra_params.get("global_timeout_seconds", self._spec.timeout_seconds)
        )
        self._max_combined_calls = int(self._extra_params.get("max_combined_calls", 10))
        self._sub_skill_timeout_seconds = int(
            self._extra_params.get("sub_skill_timeout_seconds", 120)
        )
        self._sub_skill_episode_line_cap = int(
            self._extra_params.get("sub_skill_episode_line_cap", 120)
        )
        self._native_skill_bundle_root = self._extra_params.get(
            "native_skill_bundle_root"
        )
        raw_stage_threshold = self._extra_params.get(
            "stage_result_confidence_threshold",
            0.9,
        )
        if isinstance(raw_stage_threshold, int | float) and not isinstance(
            raw_stage_threshold, bool
        ):
            self._stage_result_confidence_threshold = float(raw_stage_threshold)
        else:
            self._stage_result_confidence_threshold = 0.9
        raw_available_sub_skills = self._extra_params.get(
            "available_sub_skills",
            list(CANONICAL_SUB_SKILL_NAMES),
        )
        if not isinstance(raw_available_sub_skills, list) or any(
            not isinstance(skill_name, str) for skill_name in raw_available_sub_skills
        ):
            raise ValueError(
                "RetrieveSkill extra_params['available_sub_skills'] must be a list[str]."
            )
        invalid_sub_skills = [
            skill_name
            for skill_name in raw_available_sub_skills
            if skill_name not in CANONICAL_SUB_SKILL_NAMES
        ]
        if invalid_sub_skills:
            raise ValueError(
                "RetrieveSkill extra_params['available_sub_skills'] must only "
                f"contain {list(CANONICAL_SUB_SKILL_NAMES)}. "
                f"Invalid: {invalid_sub_skills}"
            )
        self._available_sub_skills = list(dict.fromkeys(raw_available_sub_skills))
        if not self._available_sub_skills:
            raise ValueError(
                "RetrieveSkill extra_params['available_sub_skills'] cannot be empty."
            )
        fallback_name = self._extra_params.get("fallback_tool_name", "MemMachineSkill")
        self._memory_tool = self._find_child_tool(fallback_name)
        if self._memory_tool is None:
            raise ValueError(
                f"RetrieveSkill requires a fallback child tool named '{fallback_name}'."
            )

        raw_sub_skill_root = self._extra_params.get("sub_skill_spec_root")
        if raw_sub_skill_root is None:
            self._sub_skill_spec_root = DEFAULT_SUB_SKILL_SPEC_ROOT
        else:
            self._sub_skill_spec_root = Path(str(raw_sub_skill_root))

        raw_session_model = self._extra_params.get("skill_session_model")
        if raw_session_model is None:
            self._session_model = (
                SkillLanguageModel.from_openai_responses_language_model(self._model)
            )
        elif hasattr(raw_session_model, "run_live_session"):
            self._session_model = cast(SkillSessionModelProtocol, raw_session_model)
        else:
            raise ValueError(
                "RetrieveSkill extra_params['skill_session_model'] must implement "
                "run_live_session(...)."
            )

    @property
    def skill_name(self) -> str:
        return "RetrieveSkill"

    @property
    def skill_description(self) -> str:
        return (
            "Top-level retrieval skill orchestrator driven by markdown policy "
            "tool contracts and explicit fallback reasons."
        )

    @property
    def accuracy_score(self) -> int:
        return 8

    @property
    def token_cost(self) -> int:
        return 7

    @property
    def time_cost(self) -> int:
        return 7

    def _find_child_tool(self, tool_name: str) -> SkillToolBase | None:
        for tool in self._children_tools:
            if tool.skill_name == tool_name:
                return tool
        return None

    def _native_top_level_skill_bundles(self) -> list[ProviderSkillBundle]:
        markdown = self._spec.policy_markdown or self._spec.description
        bundles = [
            materialize_provider_skill_bundle(
                name=self._spec.name,
                description=self._spec.description,
                skill_markdown=markdown,
                bundle_root=(
                    str(self._native_skill_bundle_root)
                    if self._native_skill_bundle_root is not None
                    else None
                ),
            )
        ]

        # Attach top-level and all decomposition bundles in one session.
        for sub_skill_name in self._available_sub_skills:
            sub_skill_spec = load_skill_spec(
                self._sub_skill_spec_root / f"{sub_skill_name}.md"
            )
            if sub_skill_spec.kind != "sub-skill":
                raise ValueError(
                    "Sub-skill bundle spec must declare kind='sub-skill': "
                    f"{sub_skill_name}"
                )
            sub_skill_markdown = (
                sub_skill_spec.policy_markdown or sub_skill_spec.description
            )
            bundles.append(
                materialize_provider_skill_bundle(
                    name=sub_skill_spec.name,
                    description=sub_skill_spec.description,
                    skill_markdown=sub_skill_markdown,
                    bundle_root=(
                        str(self._native_skill_bundle_root)
                        if self._native_skill_bundle_root is not None
                        else None
                    ),
                )
            )
        return bundles

    def _new_session_state(self, query: QueryParam) -> TopLevelSkillSessionState:
        session = TopLevelSkillSessionState.new(
            route_name=self._spec.route_name,
            policy_name=self._spec.name,
            query=query.query,
        )
        session.record_event(
            actor="top-level",
            event_type="session_started",
            detail="Top-level retrieve-skill session initialized.",
        )
        return session

    def _augment_metrics_with_session_state(
        self,
        *,
        metrics: dict[str, object],
        session: TopLevelSkillSessionState,
    ) -> dict[str, object]:
        metrics["orchestrator_step_count"] = session.current_step
        metrics["orchestrator_event_count"] = len(session.events)
        metrics["orchestrator_tool_call_count"] = len(session.tool_calls)
        metrics["orchestrator_sub_skill_count"] = len(session.sub_skill_runs)
        metrics["orchestrator_episode_count"] = len(session.merged_episodes)
        metrics["orchestrator_completed"] = session.completed
        metrics["orchestrator_policy_kind"] = self._spec.kind
        metrics["orchestrator_state_snapshot"] = session.prompt_snapshot()
        metrics["orchestrator_final_response"] = session.final_response
        metrics["orchestrator_trace"] = session.full_trace_snapshot()
        metrics["orchestrator_sub_skill_runs"] = [
            run.model_dump(mode="json") for run in session.sub_skill_runs
        ]
        return metrics

    @staticmethod
    def _serialize_diagnostic_object(
        payload: object,
        *,
        max_chars: int = 20000,
    ) -> str:
        try:
            serialized = json.dumps(payload, default=str)
        except Exception:
            serialized = repr(payload)
        if len(serialized) <= max_chars:
            return serialized
        return f"{serialized[:max_chars]}...[truncated]"

    def _record_error_diagnostics(
        self,
        *,
        session: TopLevelSkillSessionState,
        aggregated_metrics: dict[str, Any],
        err: Exception,
        context: str,
        mapped_contract_error: SkillContractError | None = None,
    ) -> None:
        diagnostics: dict[str, object] = {
            "context": context,
            "error_type": type(err).__name__,
            "error_message": str(err),
        }
        if isinstance(err, SkillLanguageModelError):
            provider_diag = getattr(err, "diagnostics", None)
            if isinstance(provider_diag, dict) and provider_diag:
                diagnostics["provider_diagnostics"] = provider_diag
                response_body = provider_diag.get("response_body")
                if isinstance(response_body, str) and response_body:
                    aggregated_metrics["provider_error_raw_response"] = response_body
        if isinstance(err, SkillContractError):
            diagnostics["contract_error"] = err.to_dict()
            aggregated_metrics["skill_contract_error_payload"] = err.to_dict()
        if mapped_contract_error is not None:
            diagnostics["mapped_contract_error"] = mapped_contract_error.to_dict()
            aggregated_metrics["skill_contract_error_payload"] = (
                mapped_contract_error.to_dict()
            )
        aggregated_metrics["error_diagnostics"] = diagnostics
        session.record_event(
            actor="top-level",
            event_type="error_diagnostics",
            detail=(
                f"{context}: {type(err).__name__}; "
                f"diagnostics={self._serialize_diagnostic_object(diagnostics, max_chars=1200)}"
            ),
        )

    def _contract_error(self, *, why: str, fallback_reason: str) -> SkillContractError:
        return SkillContractError(
            code=SkillContractErrorCode.INVALID_OUTPUT,
            payload=SkillContractErrorPayload(
                what_failed="Top-level tool-call contract validation failed",
                why=why,
                how_to_fix=(
                    "Emit only allowed memmachine_search calls with valid "
                    "arguments, then finish with plain assistant text."
                ),
                where="skills.retrieve_skill.do_query",
                fallback_trigger_reason=fallback_reason,
            ),
        )

    def _raise_contract_error(self, *, why: str, fallback_reason: str) -> None:
        raise self._contract_error(why=why, fallback_reason=fallback_reason)

    def _parse_function_call(
        self,
        *,
        function_call: object,
    ) -> tuple[str, dict[str, object]]:
        if not isinstance(function_call, dict):
            raise self._contract_error(
                why="Each top-level function call must be an object.",
                fallback_reason="invalid_tool_call",
            )
        fn = function_call.get("function")
        if not isinstance(fn, dict):
            raise self._contract_error(
                why="Top-level function call missing function object.",
                fallback_reason="invalid_tool_call",
            )
        raw_name = fn.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise self._contract_error(
                why="Top-level function call must provide function.name.",
                fallback_reason="invalid_tool_call",
            )
        raw_arguments = fn.get("arguments", {})
        if not isinstance(raw_arguments, dict):
            raise self._contract_error(
                why="Top-level function call arguments must be an object.",
                fallback_reason="invalid_tool_call",
            )
        return raw_name.strip(), cast(dict[str, object], raw_arguments)

    def _query_with_override(self, query: QueryParam, text: str) -> QueryParam:
        next_query = query.model_copy()
        next_query.query = text
        return next_query

    @staticmethod
    def _sanitize_tool_raw_result(raw: object) -> dict[str, object] | None:
        if not isinstance(raw, dict):
            return None
        sanitized: dict[str, object] = {}
        for key, value in raw.items():
            if key in {
                "episodes",
                "episodes_human_readable",
                "episodes_text",
                "episode_lines",
            }:
                continue
            if isinstance(value, str | int | float | bool) or value is None:
                sanitized[key] = value
                continue
            if isinstance(value, dict):
                sanitized[key] = value
                continue
            if isinstance(value, list):
                sanitized[key] = value
                continue
        return sanitized or None

    @staticmethod
    def _summary_payload(summary: str) -> dict[str, object] | None:
        stripped = summary.strip()
        if not stripped:
            return None
        candidates: list[str] = [stripped]
        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        candidates.extend(block for block in fenced_blocks if block.strip())
        if "{" in stripped and "}" in stripped:
            first_open = stripped.find("{")
            last_close = stripped.rfind("}")
            if first_open != -1 and last_close > first_open:
                candidates.append(stripped[first_open : last_close + 1])

        for candidate in candidates:
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                continue
            try:
                parsed = json.loads(candidate_stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                wrapped_v1 = parsed.get("v1")
                if isinstance(wrapped_v1, dict):
                    return cast(dict[str, object], wrapped_v1)
                return cast(dict[str, object], parsed)
        return None

    @staticmethod
    def _normalize_episode_indices(raw: object) -> list[int]:
        if not isinstance(raw, list):
            return []
        normalized: list[int] = []
        seen: set[int] = set()
        for item in raw:
            if isinstance(item, bool):
                continue
            value: int | None = None
            if isinstance(item, int):
                value = item
            elif isinstance(item, float) and item.is_integer():
                value = int(item)
            if value is None or value < 0 or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    @staticmethod
    def _normalize_string_list(
        raw: object,
        *,
        max_items: int = 32,
        max_len: int = 400,
    ) -> list[str]:
        if not isinstance(raw, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value:
                continue
            clipped = value[:max_len]
            if clipped in seen:
                continue
            seen.add(clipped)
            normalized.append(clipped)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_stage_results(self, raw: object) -> list[dict[str, object]]:
        if not isinstance(raw, list):
            return []
        normalized: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            raw_query = item.get("query")
            raw_stage_result = item.get("stage_result")
            if not isinstance(raw_query, str) or not isinstance(raw_stage_result, str):
                continue
            query_text = raw_query.strip()[:400]
            stage_text = raw_stage_result.strip()[:1000]
            if not query_text or not stage_text:
                continue
            key = (query_text, stage_text)
            if key in seen:
                continue
            seen.add(key)
            record: dict[str, object] = {
                "query": query_text,
                "stage_result": stage_text,
            }
            raw_confidence = item.get("confidence_score")
            if isinstance(raw_confidence, int | float) and not isinstance(
                raw_confidence, bool
            ):
                confidence = max(0.0, min(1.0, float(raw_confidence)))
                record["confidence_score"] = confidence
            raw_reason_note = item.get("reason_note")
            if isinstance(raw_reason_note, str) and raw_reason_note.strip():
                record["reason_note"] = raw_reason_note.strip()[:500]
            normalized.append(record)
            if len(normalized) >= 32:
                break
        return normalized

    @staticmethod
    def _append_unique_strings(target: list[str], incoming: list[str]) -> None:
        seen = set(target)
        for value in incoming:
            if value in seen:
                continue
            target.append(value)
            seen.add(value)

    @staticmethod
    def _append_unique_stage_results(
        target: list[dict[str, object]],
        incoming: list[dict[str, object]],
    ) -> None:
        seen: set[tuple[str, str]] = set()
        for item in target:
            query = item.get("query")
            stage_result = item.get("stage_result")
            if isinstance(query, str) and isinstance(stage_result, str):
                seen.add((query, stage_result))
        for item in incoming:
            query = item.get("query")
            stage_result = item.get("stage_result")
            if not isinstance(query, str) or not isinstance(stage_result, str):
                continue
            key = (query, stage_result)
            if key in seen:
                continue
            target.append(item)
            seen.add(key)

    def _stage_result_gate_passes(
        self,
        *,
        is_sufficient: bool,
        confidence_score: float | None,
    ) -> bool:
        if not is_sufficient:
            return False
        if confidence_score is None:
            return False
        return confidence_score >= self._stage_result_confidence_threshold

    def _record_sub_skill_summary_metrics(
        self,
        *,
        session: TopLevelSkillSessionState,
        aggregated_metrics: dict[str, Any],
        skill_name: str,
        summary: str,
    ) -> dict[str, object] | None:
        stripped = summary.strip()
        if not stripped:
            return None

        payload = self._summary_payload(stripped)
        records = aggregated_metrics.get("sub_skill_summaries")
        if not isinstance(records, list):
            records = []
            aggregated_metrics["sub_skill_summaries"] = records
        record: dict[str, object] = {
            "skill_name": skill_name,
            "summary": stripped[:1000],
        }
        if isinstance(payload, dict):
            record["summary_payload"] = payload
        records.append(record)

        if isinstance(payload, dict):
            self._apply_sub_skill_sufficiency_signal(
                session=session,
                aggregated_metrics=aggregated_metrics,
                skill_name=skill_name,
                summary_payload=payload,
            )
        return payload

    def _apply_sub_skill_sufficiency_signal(  # noqa: C901
        self,
        *,
        session: TopLevelSkillSessionState,
        aggregated_metrics: dict[str, Any],
        skill_name: str,
        summary_payload: dict[str, object],
    ) -> None:
        is_sufficient = summary_payload.get("is_sufficient")
        if not isinstance(is_sufficient, bool):
            return

        aggregated_metrics["sufficiency_signal_seen"] = True
        aggregated_metrics["latest_sufficiency_signal_skill"] = skill_name
        aggregated_metrics["latest_sufficiency_signal"] = is_sufficient
        if is_sufficient:
            aggregated_metrics["evidence_sufficient"] = True
        elif not bool(aggregated_metrics.get("evidence_sufficient", False)):
            aggregated_metrics["evidence_sufficient"] = False

        confidence_score = summary_payload.get("confidence_score")
        normalized_confidence: float | None = None
        if isinstance(confidence_score, int | float) and not isinstance(
            confidence_score, bool
        ):
            score = float(confidence_score)
            normalized_confidence = score
            aggregated_metrics["latest_sufficiency_confidence_score"] = score
            if is_sufficient:
                aggregated_metrics["sufficiency_confidence_score"] = score

        new_query = summary_payload.get("new_query")
        if isinstance(new_query, str) and new_query.strip():
            aggregated_metrics["latest_missing_query"] = new_query.strip()

        reason_code = summary_payload.get("reason_code")
        if isinstance(reason_code, str) and reason_code.strip():
            aggregated_metrics["latest_sufficiency_reason_code"] = reason_code
        reason_note = summary_payload.get("reason_note")
        if isinstance(reason_note, str) and reason_note.strip():
            aggregated_metrics["latest_sufficiency_reason_note"] = reason_note
        answer_candidate = summary_payload.get("answer_candidate")
        normalized_answer_candidate: str | None = None
        if isinstance(answer_candidate, str) and answer_candidate.strip():
            candidate = answer_candidate.strip()
            normalized_answer_candidate = candidate
            aggregated_metrics["latest_answer_candidate"] = candidate
            if is_sufficient:
                aggregated_metrics["answer_candidate"] = candidate

        evidence_indices = self._normalize_episode_indices(
            summary_payload.get("evidence_indices")
        )
        if evidence_indices:
            aggregated_metrics["latest_evidence_indices"] = evidence_indices
            if is_sufficient:
                aggregated_metrics["evidence_indices"] = evidence_indices

        related_episode_indices = self._normalize_episode_indices(
            summary_payload.get("related_episode_indices")
        )
        if related_episode_indices:
            aggregated_metrics["latest_related_episode_indices"] = (
                related_episode_indices
            )

        selected_episode_indices = self._normalize_episode_indices(
            summary_payload.get("selected_episode_indices")
        )
        if selected_episode_indices:
            aggregated_metrics["latest_selected_episode_indices"] = (
                selected_episode_indices
            )
            if is_sufficient:
                aggregated_metrics["selected_episode_indices"] = (
                    selected_episode_indices
                )

        if skill_name == "coq":
            stage_results = self._normalize_stage_results(
                summary_payload.get("stage_results")
            )
            if (
                not stage_results
                and normalized_answer_candidate is not None
                and isinstance(new_query, str)
                and new_query.strip()
                and self._stage_result_gate_passes(
                    is_sufficient=is_sufficient,
                    confidence_score=normalized_confidence,
                )
            ):
                inferred_stage_result: dict[str, object] = {
                    "query": new_query.strip(),
                    "stage_result": normalized_answer_candidate,
                }
                if normalized_confidence is not None:
                    inferred_stage_result["confidence_score"] = normalized_confidence
                stage_results = [inferred_stage_result]

            if stage_results:
                aggregated_metrics["latest_stage_results"] = stage_results
                if self._stage_result_gate_passes(
                    is_sufficient=is_sufficient,
                    confidence_score=normalized_confidence,
                ):
                    existing_stage_results = aggregated_metrics.get("stage_results")
                    if not isinstance(existing_stage_results, list):
                        existing_stage_results = []
                        aggregated_metrics["stage_results"] = existing_stage_results
                    self._append_unique_stage_results(
                        existing_stage_results,
                        stage_results,
                    )

        generated_sub_queries = self._normalize_string_list(
            summary_payload.get("generated_sub_queries")
            or summary_payload.get("sub_queries")
        )
        if generated_sub_queries:
            aggregated_metrics["latest_stage_sub_queries"] = generated_sub_queries
            if self._stage_result_gate_passes(
                is_sufficient=is_sufficient,
                confidence_score=normalized_confidence,
            ):
                existing_sub_queries = aggregated_metrics.get("stage_sub_queries")
                if not isinstance(existing_sub_queries, list):
                    existing_sub_queries = []
                    aggregated_metrics["stage_sub_queries"] = existing_sub_queries
                self._append_unique_strings(
                    existing_sub_queries,
                    generated_sub_queries,
                )

        session.record_event(
            actor="top-level",
            event_type="sub_skill_sufficiency_signal",
            detail=(
                f"skill={skill_name}; "
                f"is_sufficient={is_sufficient}; "
                f"reason_code={reason_code or 'n/a'}; "
                f"confidence={aggregated_metrics.get('latest_sufficiency_confidence_score', 'n/a')}"
            ),
        )

    def _record_branch_metrics(
        self,
        *,
        aggregated_metrics: dict[str, Any],
        branch_total: int,
        branch_success_count: int,
        branch_failure_count: int,
        branch_retry_count: int,
    ) -> None:
        aggregated_metrics["branch_total"] = (
            int(aggregated_metrics.get("branch_total", 0)) + branch_total
        )
        aggregated_metrics["branch_success_count"] = (
            int(aggregated_metrics.get("branch_success_count", 0))
            + branch_success_count
        )
        aggregated_metrics["branch_failure_count"] = (
            int(aggregated_metrics.get("branch_failure_count", 0))
            + branch_failure_count
        )
        aggregated_metrics["branch_retry_count"] = (
            int(aggregated_metrics.get("branch_retry_count", 0)) + branch_retry_count
        )

    @staticmethod
    def _selected_skill_name_for_skill(skill_name: str) -> str:
        if skill_name == "coq":
            return "ChainOfQuerySkill"
        return "MemMachineSkill"

    @staticmethod
    def _response_looks_inconclusive(response: str) -> bool:
        normalized = response.strip().lower()
        if not normalized:
            return True
        return any(
            marker in normalized
            for marker in (
                "i don't know",
                "i don't know",
                "i do not know",
                "unknown",
                "unclear",
                "insufficient",
                "not enough information",
                "cannot determine",
                "can't determine",
                "not mentioned",
            )
        )

    def _build_stage_result_memory_episodes(
        self,
        *,
        query: QueryParam,
        stage_results: list[dict[str, object]],
        sub_queries: list[str],
    ) -> list[Episode]:
        if not stage_results and not sub_queries:
            return []

        created_at = datetime.now(tz=UTC)
        session_key = query.memory.session_key
        episodes: list[Episode] = []
        for index, item in enumerate(stage_results, start=1):
            stage_query = str(item.get("query") or "").strip()
            stage_result = str(item.get("stage_result") or "").strip()
            if not stage_query or not stage_result:
                continue
            confidence_text = ""
            confidence_raw = item.get("confidence_score")
            if isinstance(confidence_raw, int | float) and not isinstance(
                confidence_raw, bool
            ):
                confidence_text = f" (confidence={float(confidence_raw):.2f})"
            reason_text = ""
            reason_note = item.get("reason_note")
            if isinstance(reason_note, str) and reason_note.strip():
                reason_text = f"\nReason: {reason_note.strip()}"
            content = (
                f"[StageResult {index}] Query: {stage_query}\n"
                f"Answer: {stage_result}{confidence_text}{reason_text}"
            )
            uid_seed = f"stage:{stage_query}:{stage_result}:{index}"
            uid = (
                f"stage-{index}-"
                f"{hashlib.sha1(uid_seed.encode('utf-8')).hexdigest()[:12]}"
            )
            episodes.append(
                Episode(
                    uid=uid,
                    content=content,
                    session_key=session_key,
                    created_at=created_at,
                    producer_id="retrieve-skill-stage-result",
                    producer_role="assistant",
                )
            )

        for index, sub_query in enumerate(sub_queries, start=1):
            normalized_sub_query = sub_query.strip()
            if not normalized_sub_query:
                continue
            content = f"[SubQuery {index}] {normalized_sub_query}"
            uid_seed = f"subquery:{normalized_sub_query}:{index}"
            uid = (
                f"subquery-{index}-"
                f"{hashlib.sha1(uid_seed.encode('utf-8')).hexdigest()[:12]}"
            )
            episodes.append(
                Episode(
                    uid=uid,
                    content=content,
                    session_key=session_key,
                    created_at=created_at,
                    producer_id="retrieve-skill-stage-result",
                    producer_role="assistant",
                )
            )

        if query.limit > 0:
            return episodes[: query.limit]
        return episodes

    async def _finalize_episodes(
        self,
        *,
        query: QueryParam,
        episodes: list[Episode],
    ) -> tuple[list[Episode], bool]:
        reranked = await self._do_rerank(query, episodes)
        rerank_applied = bool(
            self._reranker is not None
            and query.limit > 0
            and len(episodes) > query.limit
        )
        return reranked, rerank_applied

    async def _fallback_with_reason(
        self,
        *,
        policy: QueryPolicy,
        query: QueryParam,
        session: TopLevelSkillSessionState,
        reason: str,
        code: str,
        aggregated_metrics: dict[str, Any] | None = None,
    ) -> tuple[list[Episode], dict[str, object]]:
        session.next_step()
        fallback_episodes, fallback_metrics = await self._memory_tool.do_query(
            policy, query
        )
        fallback_arguments: dict[str, object] = {
            "query": query.query,
            "rationale": f"runtime_fallback:{reason}",
        }
        fallback_raw_result: dict[str, object] = {
            "query": query.query,
            "episodes_returned": len(fallback_episodes),
            "fallback_reason": reason,
        }
        if aggregated_metrics is not None:
            raw_error = aggregated_metrics.get("error_diagnostics")
            if isinstance(raw_error, dict):
                fallback_raw_result["error_diagnostics"] = raw_error
        session.record_tool_call(
            tool_name="memmachine_search",
            arguments=fallback_arguments,
            status="success",
            result_summary=f"episodes={len(fallback_episodes)}",
            raw_result=fallback_raw_result,
        )
        session.record_event(
            actor="top-level",
            event_type="fallback_applied",
            detail=f"Fallback executed with reason={reason}.",
        )
        session.merge_episodes(fallback_episodes)
        session.finalize()
        metrics: dict[str, object] = {}
        if aggregated_metrics is not None:
            metrics.update(cast(dict[str, object], aggregated_metrics))
        metrics.update(fallback_metrics)
        metrics["skill"] = self.skill_name
        metrics["route"] = self.skill_name
        selected_skill = metrics.get("selected_skill")
        if not isinstance(selected_skill, str) or not selected_skill.strip():
            selected_skill = "direct_memory"
            metrics["selected_skill"] = selected_skill
        metrics["selected_skill_name"] = self._selected_skill_name_for_skill(
            selected_skill
        )
        metrics["fallback_trigger_reason"] = reason
        metrics["skill_contract_error_code"] = code
        metrics["top_level_session_invocation_count"] = 1
        metrics.setdefault("llm_call_count", 0)
        metrics.setdefault("input_token", 0)
        metrics.setdefault("output_token", 0)
        metrics.setdefault("top_level_sufficiency_signal_seen", False)
        metrics.setdefault("top_level_is_sufficient", False)
        final_episodes, rerank_applied = await self._finalize_episodes(
            query=query,
            episodes=session.merged_episodes,
        )
        metrics["rerank_applied"] = rerank_applied
        metrics["final_episode_count"] = len(final_episodes)
        return final_episodes, self._augment_metrics_with_session_state(
            metrics=metrics,
            session=session,
        )

    async def do_query(  # noqa: C901
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, object]]:
        route_name = self._spec.route_name
        session = self._new_session_state(query)
        aggregated_metrics: dict[str, Any] = {}
        combined_call_count = 0
        global_started = time.monotonic()
        aggregated_metrics["session_call_budget_limit"] = self._max_combined_calls
        aggregated_metrics["session_call_budget_consumed"] = 0

        def _global_timeout_exceeded() -> bool:
            elapsed = time.monotonic() - global_started
            return elapsed > float(self._global_timeout_seconds)

        def _consume_call_budget(*, delta: int, source: str) -> None:
            nonlocal combined_call_count
            if delta <= 0:
                return
            next_count = combined_call_count + delta
            if next_count > self._max_combined_calls:
                self._raise_contract_error(
                    why=(
                        "Combined sub-skill/tool call budget exceeded: "
                        f"next={next_count} > limit={self._max_combined_calls}; "
                        f"source={source}."
                    ),
                    fallback_reason="session_call_budget_exceeded",
                )
            combined_call_count = next_count
            aggregated_metrics["session_call_budget_consumed"] = combined_call_count

        try:
            _ = build_skill_request(query, route_name=route_name)
            allowed_tools = set(self._spec.allowed_tools or TOP_LEVEL_TOOL_NAMES)
            async def _execute_memmachine_search(
                arguments: dict[str, object],
            ) -> dict[str, object]:
                session.next_step()
                if _global_timeout_exceeded():
                    self._raise_contract_error(
                        why="Global timeout exceeded during memmachine_search.",
                        fallback_reason="global_timeout",
                    )
                action = parse_top_level_tool_call(
                    tool_name="memmachine_search",
                    arguments=arguments,
                )
                _consume_call_budget(delta=1, source="memmachine_search")
                if action.action not in allowed_tools:
                    self._raise_contract_error(
                        why=(
                            "Tool 'memmachine_search' not allowed by top-level "
                            "markdown policy."
                        ),
                        fallback_reason="invalid_tool_call",
                    )

                direct_query = action.query or query.query
                tool_started = time.perf_counter()
                episodes, perf_metrics = await self._memory_tool.do_query(
                    policy,
                    self._query_with_override(query, direct_query),
                )
                tool_elapsed_seconds = time.perf_counter() - tool_started
                aggregated_metrics.update(
                    self._update_perf_metrics(perf_metrics, aggregated_metrics)
                )
                session.merge_episodes(episodes)
                direct_arguments: dict[str, object] = {
                    "query": direct_query,
                    "rationale": action.rationale,
                }
                episode_lines = [
                    line
                    for line in episodes_to_string(episodes).splitlines()
                    if line.strip()
                ]
                response_payload: dict[str, object] = {
                    "episodes_returned": len(episodes),
                    "query": direct_query,
                    "episodes_human_readable": episode_lines,
                    "wall_time_seconds": tool_elapsed_seconds,
                }
                raw_reported_memory_time = perf_metrics.get("memory_retrieval_time")
                if isinstance(raw_reported_memory_time, int | float) and not isinstance(
                    raw_reported_memory_time,
                    bool,
                ):
                    response_payload["reported_memory_retrieval_time"] = float(
                        raw_reported_memory_time
                    )
                raw_search_latency_breakdown = perf_metrics.get(
                    "memory_search_latency_seconds"
                )
                if isinstance(raw_search_latency_breakdown, list):
                    response_payload["memory_search_latency_seconds"] = [
                        float(item)
                        for item in raw_search_latency_breakdown
                        if isinstance(item, int | float) and not isinstance(item, bool)
                    ]
                session.record_tool_call(
                    tool_name="memmachine_search",
                    arguments=direct_arguments,
                    status="success",
                    result_summary=f"episodes={len(episodes)}",
                    raw_result=self._sanitize_tool_raw_result(response_payload),
                )
                session.record_event(
                    actor="top-level",
                    event_type="memmachine_search_completed",
                    detail=f"step={session.current_step}; episodes={len(episodes)}",
                )
                return response_payload
            tool_registry: dict[str, Any] = {}
            if "memmachine_search" in allowed_tools:
                tool_registry["memmachine_search"] = _execute_memmachine_search

            session_result = await self._session_model.run_live_session(
                system_prompt=(
                    "Use the attached retrieval skill and available tools to complete "
                    "the user request."
                ),
                user_prompt=(
                    f"query: {query.query}\n"
                    f"state: {session.prompt_snapshot()}\n"
                    "use memmachine_search for retrieval and answer in plain text "
                    "when finished"
                ),
                tools=top_level_tool_schemas(
                    self._spec.allowed_tools,
                    self._available_sub_skills,
                ),
                tool_registry=tool_registry,
                max_turns=self._spec.max_steps,
                timeout_seconds=float(self._global_timeout_seconds),
                provider_skill_bundles=self._native_top_level_skill_bundles(),
            )
            aggregated_metrics["llm_time"] = float(
                aggregated_metrics.get("llm_time", 0.0)
            ) + float(session_result.llm_time_seconds)
            self._update_perf_metrics(
                {
                    "llm_call_count": int(session_result.turn_count),
                    "input_token": int(session_result.llm_input_tokens),
                    "output_token": int(session_result.llm_output_tokens),
                    "top_level_input_token": int(session_result.llm_input_tokens),
                    "top_level_output_token": int(session_result.llm_output_tokens),
                    "top_level_llm_call_count": int(session_result.turn_count),
                },
                aggregated_metrics,
            )
            if session_result.normalization_warnings:
                aggregated_metrics["top_level_normalization_warnings"] = list(
                    session_result.normalization_warnings
                )

            if not session.completed:
                if not session.tool_calls:
                    # Preserve previous default behavior when no tool is emitted.
                    session.next_step()
                    episodes, perf_metrics = await self._memory_tool.do_query(
                        policy, query
                    )
                    aggregated_metrics = self._update_perf_metrics(
                        perf_metrics,
                        aggregated_metrics,
                    )
                    session.merge_episodes(episodes)
                    session.record_tool_call(
                        tool_name="memmachine_search",
                        arguments={"query": query.query},
                        status="success",
                        result_summary=(
                            "No memmachine_search call emitted; direct search "
                            "executed. "
                            f"episodes={len(episodes)}"
                        ),
                        raw_result={
                            "query": query.query,
                            "episodes_returned": len(episodes),
                        },
                    )
                    session.record_event(
                        actor="top-level",
                        event_type="default_memmachine_search",
                        detail=(
                            "No function calls returned; performed direct memory "
                            "search."
                        ),
                    )
                raw_final_response = session_result.final_response.strip()
                final_response = raw_final_response or "Top-level retrieval complete."
                top_level_is_sufficient = bool(raw_final_response) and not (
                    self._response_looks_inconclusive(raw_final_response)
                )
                aggregated_metrics["top_level_sufficiency_signal_seen"] = bool(
                    raw_final_response
                )
                aggregated_metrics["top_level_is_sufficient"] = top_level_is_sufficient
                aggregated_metrics["latest_sufficiency_signal"] = (
                    top_level_is_sufficient
                )
                aggregated_metrics["latest_sufficiency_signal_skill"] = "retrieve-skill"
                if top_level_is_sufficient:
                    aggregated_metrics["answer_candidate"] = raw_final_response
                    aggregated_metrics["latest_answer_candidate"] = raw_final_response
                session.record_event(
                    actor="top-level",
                    event_type="orchestration_completed",
                    detail=(
                        f"step={session.current_step}; "
                        f"is_sufficient={top_level_is_sufficient}"
                    ),
                )
                session.finalize(response=final_response)

            metrics: dict[str, object] = dict(aggregated_metrics)
            metrics["skill"] = self.skill_name
            metrics["route"] = self.skill_name
            metrics["skill_name"] = self._spec.name
            metrics["top_level_session_invocation_count"] = 1
            metrics["top_level_session_turn_count"] = session_result.turn_count
            metrics.setdefault("llm_call_count", 0)
            metrics.setdefault("input_token", 0)
            metrics.setdefault("output_token", 0)
            metrics.setdefault("branch_total", 0)
            metrics.setdefault("branch_success_count", 0)
            metrics.setdefault("branch_failure_count", 0)
            metrics.setdefault("branch_retry_count", 0)
            metrics.setdefault("top_level_sufficiency_signal_seen", False)
            metrics.setdefault("top_level_is_sufficient", False)
            selected_skill = metrics.get("selected_skill")
            if not isinstance(selected_skill, str) or not selected_skill.strip():
                selected_skill = "direct_memory"
                metrics["selected_skill"] = selected_skill
            metrics["selected_skill_name"] = self._selected_skill_name_for_skill(
                selected_skill
            )
            final_episodes, rerank_applied = await self._finalize_episodes(
                query=query,
                episodes=session.merged_episodes,
            )
            metrics["stage_result_memory_returned"] = False
            metrics["rerank_applied"] = rerank_applied
            metrics["final_episode_count"] = len(final_episodes)
            metrics = self._augment_metrics_with_session_state(
                metrics=metrics,
                session=session,
            )

        except SkillToolCallFormatError:
            contract_error = self._contract_error(
                why="Top-level tool-call payload shape invalid.",
                fallback_reason="invalid_tool_call",
            )
            self._record_error_diagnostics(
                session=session,
                aggregated_metrics=aggregated_metrics,
                err=contract_error,
                context="top_level_tool_call_format_error",
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=contract_error.payload.fallback_trigger_reason,
                code=contract_error.code,
                aggregated_metrics=aggregated_metrics,
            )
        except SkillToolNotFoundError:
            contract_error = self._contract_error(
                why="Top-level requested unsupported tool name.",
                fallback_reason="invalid_tool_call",
            )
            self._record_error_diagnostics(
                session=session,
                aggregated_metrics=aggregated_metrics,
                err=contract_error,
                context="top_level_tool_not_found",
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=contract_error.payload.fallback_trigger_reason,
                code=contract_error.code,
                aggregated_metrics=aggregated_metrics,
            )
        except SkillSessionLimitError as err:
            reason = "max_steps_exceeded"
            if "timeout" in str(err).lower():
                reason = "global_timeout"
            contract_error = self._contract_error(
                why=f"Top-level live session exceeded configured guardrails: {err}",
                fallback_reason=reason,
            )
            self._record_error_diagnostics(
                session=session,
                aggregated_metrics=aggregated_metrics,
                err=err,
                context="top_level_session_limit",
                mapped_contract_error=contract_error,
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=contract_error.payload.fallback_trigger_reason,
                code=contract_error.code,
                aggregated_metrics=aggregated_metrics,
            )
        except SkillLanguageModelError as err:
            contract_error = self._contract_error(
                why=f"Top-level live session failed: {err}",
                fallback_reason="downstream_tool_failure",
            )
            self._record_error_diagnostics(
                session=session,
                aggregated_metrics=aggregated_metrics,
                err=err,
                context="top_level_session_language_model_error",
                mapped_contract_error=contract_error,
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=contract_error.payload.fallback_trigger_reason,
                code=contract_error.code,
                aggregated_metrics=aggregated_metrics,
            )
        except SkillContractError as err:
            logger.warning(
                "RetrieveSkill contract failure. reason=%s code=%s",
                err.payload.fallback_trigger_reason,
                err.code,
            )
            self._record_error_diagnostics(
                session=session,
                aggregated_metrics=aggregated_metrics,
                err=err,
                context="top_level_contract_error",
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=err.payload.fallback_trigger_reason,
                code=err.code,
                aggregated_metrics=aggregated_metrics,
            )
        except Exception as err:
            mapped = fallback_for_downstream_error(
                where="skills.retrieve_skill.do_query",
                error=err,
            )
            logger.exception(
                "RetrieveSkill downstream failure. reason=%s code=%s",
                mapped.payload.fallback_trigger_reason,
                mapped.code,
            )
            self._record_error_diagnostics(
                session=session,
                aggregated_metrics=aggregated_metrics,
                err=err,
                context="top_level_unhandled_exception",
                mapped_contract_error=mapped,
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=mapped.payload.fallback_trigger_reason,
                code=mapped.code,
                aggregated_metrics=aggregated_metrics,
            )
        else:
            return final_episodes, metrics
