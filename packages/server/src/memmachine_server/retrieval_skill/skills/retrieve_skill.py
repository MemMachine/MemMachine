"""Top-level markdown-guided retrieval orchestration runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.language_model import (
    SkillLanguageModel,
    SkillLanguageModelError,
    SkillSessionLimitError,
    SkillSessionModelProtocol,
    SkillToolCallFormatError,
    SkillToolNotFoundError,
)
from memmachine_server.retrieval_skill.common.skill_api import (
    QueryParam,
    QueryPolicy,
    SkillToolBase,
    SkillToolBaseParam,
)
from memmachine_server.retrieval_skill.skills.fallback_policy import (
    FallbackTrigger,
    decide_fallback_action,
)
from memmachine_server.retrieval_skill.skills.runtime import (
    build_skill_request,
    fallback_for_downstream_error,
)
from memmachine_server.retrieval_skill.skills.session_state import (
    TopLevelSkillSessionState,
)
from memmachine_server.retrieval_skill.skills.spec_loader import load_skill_spec
from memmachine_server.retrieval_skill.skills.sub_skill_runner import (
    SubSkillExecutionResult,
    SubSkillRunner,
)
from memmachine_server.retrieval_skill.skills.tool_protocol import (
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
        self._sub_skill_timeout_seconds = int(
            self._extra_params.get("sub_skill_timeout_seconds", 120)
        )
        self._split_parallel_cap = int(self._extra_params.get("split_parallel_cap", 5))
        self._split_branch_retry_limit = int(
            self._extra_params.get("split_branch_retry_limit", 1)
        )
        self._sub_skill_episode_line_cap = int(
            self._extra_params.get("sub_skill_episode_line_cap", 120)
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
        self._available_sub_skills = list(
            self._extra_params.get(
                "available_sub_skills",
                ["direct_memory", "coq", "split"],
            )
        )
        fallback_name = self._extra_params.get("fallback_tool_name", "MemMachineSkill")
        self._memory_tool = self._find_child_tool(fallback_name)
        if self._memory_tool is None:
            raise ValueError(
                f"RetrieveSkill requires a fallback child tool named '{fallback_name}'."
            )

        raw_sub_skill_root = self._extra_params.get("sub_skill_spec_root")
        sub_skill_root: Path | None = None
        if raw_sub_skill_root is not None:
            sub_skill_root = Path(str(raw_sub_skill_root))

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

        self._sub_skill_runner = SubSkillRunner(
            model=self._model,
            memory_tool=self._memory_tool,
            session_model=self._session_model,
            spec_root=sub_skill_root,
            split_parallel_cap=self._split_parallel_cap,
            split_branch_retry_limit=self._split_branch_retry_limit,
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

    def _contract_error(self, *, why: str, fallback_reason: str) -> SkillContractError:
        return SkillContractError(
            code=SkillContractErrorCode.INVALID_OUTPUT,
            payload=SkillContractErrorPayload(
                what_failed="Top-level tool-call contract validation failed",
                why=why,
                how_to_fix=(
                    "Emit only allowed tool calls with valid arguments and "
                    "finish with return_final."
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
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
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

        normalized_skill_name = skill_name.strip().replace("-", "_").lower()
        if normalized_skill_name == "coq":
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
        normalized = skill_name.strip().replace("-", "_").lower()
        if normalized == "coq":
            return "ChainOfQuerySkill"
        if normalized == "split":
            return "SplitSkill"
        return "MemMachineSkill"

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
        session.record_tool_call(
            tool_name="direct_memory_search",
            arguments=fallback_arguments,
            status="success",
            result_summary=f"episodes={len(fallback_episodes)}",
            raw_result={
                "query": query.query,
                "episodes_returned": len(fallback_episodes),
                "fallback_reason": reason,
            },
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
        guardrail_retry_count = 0
        hop_count = 0
        branch_count = 0
        global_started = time.monotonic()

        def _global_timeout_exceeded() -> bool:
            elapsed = time.monotonic() - global_started
            return elapsed > float(self._global_timeout_seconds)

        try:
            _ = build_skill_request(query, route_name=route_name)
            allowed_tools = set(self._spec.allowed_tools or TOP_LEVEL_TOOL_NAMES)

            async def _execute_spawn_sub_skill(  # noqa: C901
                arguments: dict[str, object],
            ) -> dict[str, object]:
                nonlocal hop_count, branch_count, guardrail_retry_count
                session.next_step()
                if _global_timeout_exceeded():
                    self._raise_contract_error(
                        why="Global timeout exceeded during sub-skill spawn.",
                        fallback_reason="global_timeout",
                    )

                action = parse_top_level_tool_call(
                    tool_name="spawn_sub_skill",
                    arguments=arguments,
                )
                if action.action not in allowed_tools:
                    self._raise_contract_error(
                        why=(
                            f"Tool '{action.action}' not allowed by top-level "
                            "markdown policy."
                        ),
                        fallback_reason="invalid_tool_call",
                    )
                if not action.skill_name:
                    self._raise_contract_error(
                        why="spawn_sub_skill requires skill_name.",
                        fallback_reason="invalid_tool_call",
                    )
                if action.skill_name not in self._available_sub_skills:
                    self._raise_contract_error(
                        why=(
                            "spawn_sub_skill skill_name not allowed: "
                            f"{action.skill_name}. Allowed: {self._available_sub_skills}"
                        ),
                        fallback_reason="invalid_tool_call",
                    )

                hop_count += 1
                branch_count += 1
                if hop_count > self._max_hops:
                    self._raise_contract_error(
                        why=f"Max hops exceeded: hops={hop_count} > {self._max_hops}.",
                        fallback_reason="max_hops_exceeded",
                    )
                if branch_count > self._max_branches:
                    self._raise_contract_error(
                        why=(
                            "Max branches exceeded: "
                            f"branches={branch_count} > {self._max_branches}."
                        ),
                        fallback_reason="max_branches_exceeded",
                    )

                sub_query = action.query or query.query

                async def _run_sub_skill_once() -> SubSkillExecutionResult:
                    nonlocal guardrail_retry_count
                    timeout_retry_count = 0
                    while True:
                        try:
                            return await asyncio.wait_for(
                                self._sub_skill_runner.run(
                                    skill_name=action.skill_name,
                                    policy=policy,
                                    query=self._query_with_override(query, sub_query),
                                ),
                                timeout=float(self._sub_skill_timeout_seconds),
                            )
                        except TimeoutError:
                            decision = decide_fallback_action(
                                trigger=FallbackTrigger.SUB_SKILL_TIMEOUT,
                                retry_count=timeout_retry_count,
                                max_retries=self._max_guardrail_retries,
                            )
                            if decision.action == "retry":
                                timeout_retry_count += 1
                                guardrail_retry_count += 1
                                session.record_event(
                                    actor="top-level",
                                    event_type="guardrail_retry",
                                    detail=(
                                        f"trigger={FallbackTrigger.SUB_SKILL_TIMEOUT.value}; "
                                        f"retry_count={guardrail_retry_count}; "
                                        f"skill={action.skill_name}"
                                    ),
                                )
                                continue
                            self._raise_contract_error(
                                why=(
                                    "Sub-skill timeout exceeded: "
                                    f"timeout={self._sub_skill_timeout_seconds}s; "
                                    f"skill={action.skill_name}"
                                ),
                                fallback_reason=decision.fallback_trigger_reason,
                            )
                        except Exception as err:
                            decision = decide_fallback_action(
                                trigger=FallbackTrigger.SUB_SKILL_EXCEPTION,
                                retry_count=guardrail_retry_count,
                                max_retries=self._max_guardrail_retries,
                            )
                            self._raise_contract_error(
                                why=(
                                    "Sub-skill execution raised exception: "
                                    f"{action.skill_name}: {err}"
                                ),
                                fallback_reason=decision.fallback_trigger_reason,
                            )

                sub_result = await _run_sub_skill_once()
                aggregated_metrics["llm_time"] = float(
                    aggregated_metrics.get("llm_time", 0.0)
                ) + float(sub_result.llm_time)
                self._update_perf_metrics(
                    {
                        "llm_call_count": sub_result.llm_call_count,
                        "input_token": sub_result.llm_input_tokens,
                        "output_token": sub_result.llm_output_tokens,
                        "memory_search_called": sub_result.memory_search_called,
                        "memory_retrieval_time": sub_result.memory_retrieval_time,
                    },
                    aggregated_metrics,
                )
                session.merge_episodes(sub_result.episodes)

                if sub_result.branch_total > 0:
                    self._record_branch_metrics(
                        aggregated_metrics=aggregated_metrics,
                        branch_total=sub_result.branch_total,
                        branch_success_count=sub_result.branch_success_count,
                        branch_failure_count=sub_result.branch_failure_count,
                        branch_retry_count=sub_result.branch_retry_count,
                    )

                if sub_result.status != "success":
                    self._raise_contract_error(
                        why=(
                            f"Sub-skill {sub_result.skill_name} returned status="
                            f"{sub_result.status}."
                        ),
                        fallback_reason=(
                            sub_result.fallback_trigger_reason
                            or FallbackTrigger.SUB_SKILL_EXCEPTION.value
                        ),
                    )

                summary_payload = self._record_sub_skill_summary_metrics(
                    session=session,
                    aggregated_metrics=aggregated_metrics,
                    skill_name=sub_result.skill_name,
                    summary=sub_result.summary,
                )

                session.record_sub_skill_run(
                    skill_name=sub_result.skill_name,
                    query=sub_result.query,
                    status=sub_result.status,
                    fallback_trigger_reason=sub_result.fallback_trigger_reason,
                    tool_calls=sub_result.tool_calls,
                    llm_call_count=sub_result.llm_call_count,
                    llm_input_tokens=sub_result.llm_input_tokens,
                    llm_output_tokens=sub_result.llm_output_tokens,
                    llm_time=sub_result.llm_time,
                    episodes_returned=len(sub_result.episodes),
                    branch_total=sub_result.branch_total,
                    branch_success_count=sub_result.branch_success_count,
                    branch_failure_count=sub_result.branch_failure_count,
                    branch_retry_count=sub_result.branch_retry_count,
                )
                episode_lines = [
                    line
                    for line in episodes_to_string(sub_result.episodes).splitlines()
                    if line.strip()
                ][: self._sub_skill_episode_line_cap]
                spawn_arguments: dict[str, object] = {
                    "skill_name": sub_result.skill_name,
                    "query": sub_query,
                    "rationale": action.rationale,
                }
                if sub_result.summary.strip():
                    spawn_arguments["summary"] = sub_result.summary
                if summary_payload is not None:
                    spawn_arguments["summary_payload"] = summary_payload
                response_payload: dict[str, object] = {
                    "skill_name": sub_result.skill_name,
                    "episodes_returned": len(sub_result.episodes),
                    "status": sub_result.status,
                    "branch_total": sub_result.branch_total,
                    "tool_call_count": len(sub_result.tool_calls),
                    "episodes_human_readable": episode_lines,
                }
                if sub_result.summary.strip():
                    response_payload["summary"] = sub_result.summary
                if summary_payload is not None:
                    response_payload["summary_payload"] = summary_payload
                session.record_tool_call(
                    tool_name="spawn_sub_skill",
                    arguments=spawn_arguments,
                    status=sub_result.status,
                    result_summary=(
                        f"sub_skill={sub_result.skill_name}; "
                        f"episodes={len(sub_result.episodes)}"
                    ),
                    raw_result=self._sanitize_tool_raw_result(response_payload),
                )
                session.record_event(
                    actor="top-level",
                    event_type="sub_skill_completed",
                    detail=(
                        f"step={session.current_step}; "
                        f"skill={sub_result.skill_name}; "
                        f"episodes={len(sub_result.episodes)}; "
                        f"branches={sub_result.branch_total}; "
                        f"branch_failures={sub_result.branch_failure_count}"
                    ),
                )
                return response_payload

            async def _execute_direct_memory_search(
                arguments: dict[str, object],
            ) -> dict[str, object]:
                session.next_step()
                if _global_timeout_exceeded():
                    self._raise_contract_error(
                        why="Global timeout exceeded during direct memory search.",
                        fallback_reason="global_timeout",
                    )
                action = parse_top_level_tool_call(
                    tool_name="direct_memory_search",
                    arguments=arguments,
                )
                if action.action not in allowed_tools:
                    self._raise_contract_error(
                        why=(
                            f"Tool '{action.action}' not allowed by top-level "
                            "markdown policy."
                        ),
                        fallback_reason="invalid_tool_call",
                    )

                direct_query = action.query or query.query
                episodes, perf_metrics = await self._memory_tool.do_query(
                    policy,
                    self._query_with_override(query, direct_query),
                )
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
                }
                session.record_tool_call(
                    tool_name="direct_memory_search",
                    arguments=direct_arguments,
                    status="success",
                    result_summary=f"episodes={len(episodes)}",
                    raw_result=self._sanitize_tool_raw_result(response_payload),
                )
                session.record_event(
                    actor="top-level",
                    event_type="direct_memory_completed",
                    detail=f"step={session.current_step}; episodes={len(episodes)}",
                )
                return response_payload

            async def _execute_return_final(  # noqa: C901
                arguments: dict[str, object],
            ) -> dict[str, object]:
                session.next_step()
                action = parse_top_level_tool_call(
                    tool_name="return_final",
                    arguments=arguments,
                )
                if action.action not in allowed_tools:
                    self._raise_contract_error(
                        why=(
                            f"Tool '{action.action}' not allowed by top-level "
                            "markdown policy."
                        ),
                        fallback_reason="invalid_tool_call",
                    )

                final_response = (
                    action.final_response.strip()
                    if action.final_response
                    else "Top-level retrieval complete."
                )
                top_level_is_sufficient: bool
                top_level_signal_source = "explicit"
                if isinstance(action.is_sufficient, bool):
                    top_level_is_sufficient = action.is_sufficient
                else:
                    inferred_sufficiency = aggregated_metrics.get("evidence_sufficient")
                    if isinstance(inferred_sufficiency, bool):
                        top_level_is_sufficient = inferred_sufficiency
                        top_level_signal_source = "inferred_from_sub_skills"
                    else:
                        top_level_is_sufficient = False
                        top_level_signal_source = "default_false"

                confidence_score = action.confidence_score
                normalized_confidence: float | None = None
                if isinstance(confidence_score, int | float) and not isinstance(
                    confidence_score, bool
                ):
                    normalized_confidence = float(confidence_score)

                related_episode_indices = self._normalize_episode_indices(
                    action.related_episode_indices
                )
                selected_episode_indices = self._normalize_episode_indices(
                    action.selected_episode_indices
                )
                provided_stage_results = self._normalize_stage_results(
                    action.stage_results
                )
                provided_sub_queries = self._normalize_string_list(action.sub_queries)
                if provided_stage_results:
                    aggregated_metrics["top_level_stage_results"] = (
                        provided_stage_results
                    )
                if provided_sub_queries:
                    aggregated_metrics["top_level_sub_queries"] = provided_sub_queries
                gate_passes = self._stage_result_gate_passes(
                    is_sufficient=top_level_is_sufficient,
                    confidence_score=normalized_confidence,
                )
                if gate_passes and not provided_stage_results:
                    inferred_stage_results = aggregated_metrics.get("stage_results")
                    if isinstance(inferred_stage_results, list):
                        normalized_inferred = self._normalize_stage_results(
                            inferred_stage_results
                        )
                        if normalized_inferred:
                            provided_stage_results = normalized_inferred
                            aggregated_metrics["top_level_stage_results"] = (
                                normalized_inferred
                            )
                if gate_passes and not provided_sub_queries:
                    inferred_sub_queries = aggregated_metrics.get("stage_sub_queries")
                    if isinstance(inferred_sub_queries, list):
                        normalized_inferred_sub_queries = self._normalize_string_list(
                            inferred_sub_queries
                        )
                        if normalized_inferred_sub_queries:
                            provided_sub_queries = normalized_inferred_sub_queries
                            aggregated_metrics["top_level_sub_queries"] = (
                                normalized_inferred_sub_queries
                            )
                if (
                    top_level_is_sufficient
                    and normalized_confidence is not None
                    and normalized_confidence >= self._stage_result_confidence_threshold
                    and not selected_episode_indices
                ):
                    # Keep backward-compatible return-all behavior when no
                    # explicit evidence selection is emitted.
                    pass
                aggregated_metrics["top_level_sufficiency_signal_seen"] = True
                aggregated_metrics["top_level_is_sufficient"] = top_level_is_sufficient
                aggregated_metrics["top_level_sufficiency_signal_source"] = (
                    top_level_signal_source
                )
                if normalized_confidence is not None:
                    aggregated_metrics["top_level_confidence_score"] = (
                        normalized_confidence
                    )
                if isinstance(action.reason_code, str) and action.reason_code.strip():
                    aggregated_metrics["top_level_reason_code"] = action.reason_code
                if isinstance(action.reason_note, str) and action.reason_note.strip():
                    aggregated_metrics["top_level_reason_note"] = action.reason_note
                if related_episode_indices:
                    aggregated_metrics["top_level_related_episode_indices"] = (
                        related_episode_indices
                    )
                if selected_episode_indices:
                    aggregated_metrics["top_level_selected_episode_indices"] = (
                        selected_episode_indices
                    )
                if provided_stage_results:
                    aggregated_metrics["top_level_stage_results"] = (
                        provided_stage_results
                    )
                if provided_sub_queries:
                    aggregated_metrics["top_level_sub_queries"] = provided_sub_queries
                aggregated_metrics["stage_result_confidence_threshold"] = (
                    self._stage_result_confidence_threshold
                )

                return_arguments: dict[str, object] = {
                    "final_response": final_response,
                    "rationale": action.rationale,
                    "is_sufficient": top_level_is_sufficient,
                }
                if normalized_confidence is not None:
                    return_arguments["confidence_score"] = normalized_confidence
                if isinstance(action.reason_code, str) and action.reason_code.strip():
                    return_arguments["reason_code"] = action.reason_code
                if isinstance(action.reason_note, str) and action.reason_note.strip():
                    return_arguments["reason_note"] = action.reason_note
                if related_episode_indices:
                    return_arguments["related_episode_indices"] = (
                        related_episode_indices
                    )
                if selected_episode_indices:
                    return_arguments["selected_episode_indices"] = (
                        selected_episode_indices
                    )
                if provided_stage_results:
                    return_arguments["stage_results"] = provided_stage_results
                if provided_sub_queries:
                    return_arguments["sub_queries"] = provided_sub_queries
                session.record_tool_call(
                    tool_name="return_final",
                    arguments=return_arguments,
                    status="success",
                    result_summary="orchestration finalized",
                    raw_result={
                        "final_response": final_response,
                        "is_sufficient": top_level_is_sufficient,
                        "confidence_score": normalized_confidence,
                        "reason_code": action.reason_code,
                        "reason_note": action.reason_note,
                        "related_episode_indices": related_episode_indices,
                        "selected_episode_indices": selected_episode_indices,
                        "stage_results": provided_stage_results,
                        "sub_queries": provided_sub_queries,
                    },
                )
                session.record_event(
                    actor="top-level",
                    event_type="orchestration_completed",
                    detail=(
                        f"step={session.current_step}; finalized by top-level policy; "
                        f"is_sufficient={top_level_is_sufficient}; "
                        f"confidence={normalized_confidence if normalized_confidence is not None else 'n/a'}"
                    ),
                )
                session.finalize(response=final_response)
                return {"final_response": final_response}

            tool_registry: dict[str, Any] = {}
            if "spawn_sub_skill" in allowed_tools:
                tool_registry["spawn_sub_skill"] = _execute_spawn_sub_skill
            if "direct_memory_search" in allowed_tools:
                tool_registry["direct_memory_search"] = _execute_direct_memory_search
            if "return_final" in allowed_tools:
                tool_registry["return_final"] = _execute_return_final

            session_result = await self._session_model.run_live_session(
                system_prompt=self._spec.policy_markdown or self._spec.description,
                user_prompt=(
                    f"query: {query.query}\n"
                    f"available_sub_skills: {', '.join(self._available_sub_skills)}\n"
                    f"state: {session.prompt_snapshot()}\n"
                    "choose tool calls to complete retrieval"
                ),
                tools=top_level_tool_schemas(self._spec.allowed_tools),
                tool_registry=tool_registry,
                max_turns=self._spec.max_steps,
                timeout_seconds=float(self._global_timeout_seconds),
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
                        tool_name="direct_memory_search",
                        arguments={"query": query.query},
                        status="success",
                        result_summary=(
                            "No top-level tool call emitted; direct search executed. "
                            f"episodes={len(episodes)}"
                        ),
                        raw_result={
                            "query": query.query,
                            "episodes_returned": len(episodes),
                        },
                    )
                    session.record_event(
                        actor="top-level",
                        event_type="default_direct_memory_search",
                        detail="No function calls returned; performed direct memory search.",
                    )
                final_response = (
                    session.final_response
                    or session_result.final_response.strip()
                    or "Top-level retrieval complete."
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
            metrics.setdefault(
                "top_level_is_sufficient",
                bool(metrics.get("evidence_sufficient", False)),
            )
            if "top_level_confidence_score" not in metrics:
                latest_conf = metrics.get("latest_sufficiency_confidence_score")
                if isinstance(latest_conf, int | float) and not isinstance(
                    latest_conf, bool
                ):
                    metrics["top_level_confidence_score"] = float(latest_conf)
            selected_skill = metrics.get("selected_skill")
            if not isinstance(selected_skill, str) or not selected_skill.strip():
                if session.sub_skill_runs:
                    selected_skill = session.sub_skill_runs[0].skill_name
                else:
                    selected_skill = "direct_memory"
                metrics["selected_skill"] = selected_skill
            metrics["selected_skill_name"] = self._selected_skill_name_for_skill(
                selected_skill
            )
            top_level_confidence_raw = metrics.get("top_level_confidence_score")
            top_level_confidence: float | None = None
            if isinstance(top_level_confidence_raw, int | float) and not isinstance(
                top_level_confidence_raw, bool
            ):
                top_level_confidence = float(top_level_confidence_raw)

            top_level_stage_results = self._normalize_stage_results(
                metrics.get("top_level_stage_results") or metrics.get("stage_results")
            )
            top_level_sub_queries = self._normalize_string_list(
                metrics.get("top_level_sub_queries") or metrics.get("stage_sub_queries")
            )
            stage_result_memory_episodes: list[Episode] = []
            if self._stage_result_gate_passes(
                is_sufficient=bool(metrics.get("top_level_is_sufficient", False)),
                confidence_score=top_level_confidence,
            ) and top_level_stage_results:
                stage_result_memory_episodes = self._build_stage_result_memory_episodes(
                    query=query,
                    stage_results=top_level_stage_results,
                    sub_queries=top_level_sub_queries,
                )

            if stage_result_memory_episodes:
                final_episodes = stage_result_memory_episodes
                rerank_applied = False
                metrics["stage_result_memory_returned"] = True
                metrics["returned_stage_result_count"] = len(top_level_stage_results)
                metrics["returned_sub_query_count"] = len(top_level_sub_queries)
            else:
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
            err = self._contract_error(
                why="Top-level tool-call payload shape invalid.",
                fallback_reason="invalid_tool_call",
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=err.payload.fallback_trigger_reason,
                code=err.code,
                aggregated_metrics=aggregated_metrics,
            )
        except SkillToolNotFoundError:
            err = self._contract_error(
                why="Top-level requested unsupported tool name.",
                fallback_reason="invalid_tool_call",
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=err.payload.fallback_trigger_reason,
                code=err.code,
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
