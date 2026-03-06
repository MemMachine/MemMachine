"""Sub-skill execution runtime for markdown-guided retrieve-skill orchestration."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.language_model import (
    ProviderSkillBundle,
    SkillLanguageModel,
    SkillLanguageModelError,
    SkillRunResult,
    SkillSessionLimitError,
    SkillSessionModelProtocol,
    SkillToolCallFormatError,
    SkillToolNotFoundError,
    materialize_provider_skill_bundle,
)
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.retrieval_skill.common.skill_api import (
    QueryParam,
    QueryPolicy,
    SkillToolBase,
)
from memmachine_server.retrieval_skill.skills.session_state import SkillToolCallRecord
from memmachine_server.retrieval_skill.skills.spec_loader import load_skill_spec
from memmachine_server.retrieval_skill.skills.tool_protocol import (
    parse_sub_skill_tool_call,
    sub_skill_tool_schemas,
)
from memmachine_server.retrieval_skill.skills.types import (
    SkillContractError,
    SkillContractErrorCode,
    SkillContractErrorPayload,
    SkillSpecV1,
)


class SubSkillExecutionResult(BaseModel):
    """Structured result returned from a sub-skill execution."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str
    query: str
    status: str
    summary: str = ""
    episodes: list[Episode] = Field(default_factory=list)
    tool_calls: list[SkillToolCallRecord] = Field(default_factory=list)
    fallback_trigger_reason: str | None = None
    llm_time: float = 0.0
    llm_call_count: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    memory_search_called: int = 0
    memory_retrieval_time: float = 0.0
    branch_total: int = 0
    branch_success_count: int = 0
    branch_failure_count: int = 0
    branch_retry_count: int = 0
    normalization_warnings: list[str] = Field(default_factory=list)


class SubSkillRunner:
    """Run one sub-skill using markdown policy and memory-search tool access."""

    def __init__(
        self,
        *,
        model: LanguageModel,
        memory_tool: SkillToolBase,
        session_model: SkillSessionModelProtocol | None = None,
        spec_root: Path | None = None,
        native_skill_bundle_root: str | None = None,
    ) -> None:
        """Initialize sub-skill runtime dependencies."""
        self._model = model
        self._memory_tool = memory_tool
        self._session_model = (
            session_model
            or SkillLanguageModel.from_openai_responses_language_model(model)
        )
        self._spec_root = spec_root or (
            Path(__file__).resolve().parent / "specs" / "sub_skills"
        )
        self._native_skill_bundle_root = native_skill_bundle_root

    @staticmethod
    def _normalize_query_for_cache(query: str) -> str:
        return " ".join(query.lower().split())

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", query.lower()))

    @staticmethod
    def _query_similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
        if not a_tokens or not b_tokens:
            return 0.0
        union = a_tokens | b_tokens
        if not union:
            return 0.0
        return len(a_tokens & b_tokens) / len(union)

    @staticmethod
    def _sanitize_raw_tool_result(raw: object) -> dict[str, object] | None:
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

    def _spec_path_for(self, skill_name: str) -> Path:
        candidates = [
            skill_name,
            skill_name.replace("-", "_"),
            skill_name.replace("_", "-"),
        ]
        for candidate in candidates:
            candidate_path = self._spec_root / f"{candidate}.md"
            if candidate_path.exists():
                return candidate_path
        return self._spec_root / f"{skill_name}.md"

    def _load_sub_skill_spec(self, skill_name: str) -> SkillSpecV1:
        spec = load_skill_spec(self._spec_path_for(skill_name))
        if spec.kind != "sub-skill":
            raise SkillContractError(
                code=SkillContractErrorCode.INVALID_SPEC,
                payload=SkillContractErrorPayload(
                    what_failed="Sub-skill spec kind mismatch",
                    why=f"Expected sub-skill, got {spec.kind}",
                    how_to_fix="Set kind: sub-skill in markdown frontmatter.",
                    where="skills.sub_skill_runner._load_sub_skill_spec",
                    fallback_trigger_reason="invalid_sub_skill_spec",
                ),
            )
        return spec

    def _native_skill_bundles(
        self,
        *,
        spec: SkillSpecV1,
    ) -> list[ProviderSkillBundle]:
        markdown = spec.policy_markdown or spec.description
        bundle = materialize_provider_skill_bundle(
            name=spec.name,
            description=spec.description,
            skill_markdown=markdown,
            bundle_root=self._native_skill_bundle_root,
        )
        return [bundle]

    def _raise_invalid_output(self, *, why: str, fallback_reason: str) -> None:
        raise SkillContractError(
            code=SkillContractErrorCode.INVALID_OUTPUT,
            payload=SkillContractErrorPayload(
                what_failed="Sub-skill function call payload invalid",
                why=why,
                how_to_fix="Return list[dict] function calls with valid arguments.",
                where="skills.sub_skill_runner.run",
                fallback_trigger_reason=fallback_reason,
            ),
        )

    @staticmethod
    def _format_language_model_error(err: SkillLanguageModelError) -> str:
        diagnostics = getattr(err, "diagnostics", None)
        if not isinstance(diagnostics, dict) or not diagnostics:
            return str(err)
        try:
            encoded = json.dumps(diagnostics, default=str)
        except Exception:
            encoded = repr(diagnostics)
        if len(encoded) > 4000:
            encoded = f"{encoded[:4000]}...[truncated]"
        return f"{err} diagnostics={encoded}"

    def _query_with_override(self, query: QueryParam, text: str) -> QueryParam:
        next_query = query.model_copy()
        next_query.query = text
        return next_query

    def _dedupe_episodes(self, episodes: list[Episode]) -> list[Episode]:
        seen: set[str] = set()
        deduped: list[Episode] = []
        for episode in episodes:
            if episode.uid in seen:
                continue
            seen.add(episode.uid)
            deduped.append(episode)
        return deduped

    @staticmethod
    def _metric_as_int(metrics: dict[str, object], key: str) -> int:
        value = metrics.get(key, 0)
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return 0

    @staticmethod
    def _metric_as_float(metrics: dict[str, object], key: str) -> float:
        value = metrics.get(key, 0.0)
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, int | float):
            return float(value)
        return 0.0

    def _tool_calls_from_live_result(  # noqa: C901
        self,
        *,
        live_result: SkillRunResult,
        query: QueryParam,
        memmachine_call_details: list[dict[str, object]] | None = None,
    ) -> list[SkillToolCallRecord]:
        records: list[SkillToolCallRecord] = []
        call_details = memmachine_call_details or []
        memmachine_call_index = 0
        for step_index, execution in enumerate(live_result.tool_executions, start=1):
            action = parse_sub_skill_tool_call(
                tool_name=execution.name,
                arguments=execution.arguments,
            )
            if action.action == "memmachine_search":
                output = execution.output
                episodes_returned = 0
                cached = False
                if isinstance(output, dict):
                    raw_count = output.get("episodes_returned", 0)
                    if isinstance(raw_count, int):
                        episodes_returned = raw_count
                    cached = bool(output.get("cached", False))
                call_arguments: dict[str, object] = {
                    "query": action.query or query.query
                }
                episodes_human_readable: list[str] = []
                if memmachine_call_index < len(call_details):
                    detail = call_details[memmachine_call_index]
                    memmachine_call_index += 1
                    raw_query = detail.get("query")
                    if isinstance(raw_query, str) and raw_query.strip():
                        call_arguments["query"] = raw_query
                    raw_episode_lines = detail.get("episodes_human_readable")
                    if isinstance(raw_episode_lines, list):
                        episodes_human_readable = [
                            line
                            for line in raw_episode_lines
                            if isinstance(line, str) and line.strip()
                        ]
                    raw_cached = detail.get("cached")
                    if isinstance(raw_cached, bool):
                        call_arguments["cached"] = raw_cached
                    cached_from_query = detail.get("cached_from_query")
                    if isinstance(cached_from_query, str) and cached_from_query.strip():
                        call_arguments["cached_from_query"] = cached_from_query
                    raw_wall_time = detail.get("wall_time_seconds")
                    if isinstance(raw_wall_time, int | float) and not isinstance(
                        raw_wall_time, bool
                    ):
                        call_arguments["wall_time_seconds"] = float(raw_wall_time)
                    raw_reported_time = detail.get("reported_memory_retrieval_time")
                    if isinstance(raw_reported_time, int | float) and not isinstance(
                        raw_reported_time, bool
                    ):
                        call_arguments["reported_memory_retrieval_time"] = float(
                            raw_reported_time
                        )
                    raw_breakdown = detail.get("memory_search_latency_seconds")
                    if isinstance(raw_breakdown, list):
                        call_arguments["memory_search_latency_seconds"] = [
                            float(item)
                            for item in raw_breakdown
                            if isinstance(item, int | float)
                            and not isinstance(item, bool)
                        ]
                call_arguments["episodes_human_readable"] = episodes_human_readable
                result_summary = f"episodes={episodes_returned}"
                if cached:
                    result_summary += "; cached=true"
                records.append(
                    SkillToolCallRecord(
                        step=step_index,
                        tool_name=action.action,
                        arguments=call_arguments,
                        status="success",
                        result_summary=result_summary,
                        raw_result=self._sanitize_raw_tool_result(execution.output),
                    )
                )
                continue
            if action.action == "return_sub_skill_result":
                records.append(
                    SkillToolCallRecord(
                        step=step_index,
                        tool_name=action.action,
                        arguments={"summary": action.summary},
                        status="success",
                        result_summary="summary recorded",
                        raw_result=self._sanitize_raw_tool_result(execution.output),
                    )
                )
                continue
            records.append(
                SkillToolCallRecord(
                    step=step_index,
                    tool_name=action.action,
                    arguments=execution.arguments,
                    status="ignored",
                    result_summary="unsupported action ignored",
                    raw_result=self._sanitize_raw_tool_result(execution.output),
                )
            )
        return records

    async def _run_standard_skill(  # noqa: C901
        self,
        *,
        skill_name: str,
        spec: SkillSpecV1,
        policy: QueryPolicy,
        query: QueryParam,
        user_prompt: str | None = None,
        max_tool_calls: int | None = None,
    ) -> SubSkillExecutionResult:
        bounded_max_steps = max(1, spec.max_steps)
        result = SubSkillExecutionResult(
            skill_name=skill_name,
            query=query.query,
            status="in_progress",
        )
        collected_episodes: list[Episode] = []
        memmachine_call_details: list[dict[str, object]] = []
        cached_query_results: list[dict[str, object]] = []
        summary_from_tool: str = ""
        memory_search_called = 0
        memory_retrieval_time = 0.0

        async def _tool_memmachine_search(  # noqa: C901
            arguments: dict[str, object],
        ) -> dict[str, object]:
            nonlocal memory_search_called, memory_retrieval_time
            next_query = str(arguments.get("query") or query.query)
            normalized_query = self._normalize_query_for_cache(next_query)
            query_tokens = self._query_tokens(next_query)
            matched_cache: dict[str, object] | None = None
            for cached in cached_query_results:
                cached_norm = cached.get("normalized_query")
                if isinstance(cached_norm, str) and cached_norm == normalized_query:
                    matched_cache = cached
                    break
            if matched_cache is None and len(query_tokens) >= 4:
                best_match: dict[str, object] | None = None
                best_score = 0.0
                for cached in cached_query_results:
                    cached_tokens = cached.get("query_tokens")
                    if not isinstance(cached_tokens, set):
                        continue
                    score = self._query_similarity(query_tokens, cached_tokens)
                    if score > best_score:
                        best_score = score
                        best_match = cached
                if best_match is not None and best_score >= 0.85:
                    matched_cache = best_match

            cached = matched_cache is not None
            cached_from_query: str | None = None
            if matched_cache is not None:
                episodes = matched_cache["episodes"]
                assert isinstance(episodes, list)
                cached_from_query = matched_cache.get("query")
                if not isinstance(cached_from_query, str):
                    cached_from_query = None
                tool_elapsed_seconds = 0.0
                reported_memory_retrieval_time = 0.0
                search_latency_breakdown: list[float] = []
            else:
                tool_started = time.perf_counter()
                next_param = self._query_with_override(query, next_query)
                episodes, memory_metrics = await self._memory_tool.do_query(
                    policy, next_param
                )
                tool_elapsed_seconds = time.perf_counter() - tool_started
                reported_memory_retrieval_time = self._metric_as_float(
                    memory_metrics,
                    "memory_retrieval_time",
                )
                raw_search_latency_breakdown = memory_metrics.get(
                    "memory_search_latency_seconds"
                )
                search_latency_breakdown = (
                    [
                        float(item)
                        for item in raw_search_latency_breakdown
                        if isinstance(item, int | float) and not isinstance(item, bool)
                    ]
                    if isinstance(raw_search_latency_breakdown, list)
                    else []
                )
                memory_search_called += self._metric_as_int(
                    memory_metrics,
                    "memory_search_called",
                )
                memory_retrieval_time += reported_memory_retrieval_time
                collected_episodes.extend(episodes)
                cached_query_results.append(
                    {
                        "query": next_query,
                        "normalized_query": normalized_query,
                        "query_tokens": query_tokens,
                        "episodes": episodes,
                    }
                )

            episode_lines = [
                line
                for line in episodes_to_string(episodes).splitlines()
                if line.strip()
            ]
            memmachine_call_details.append(
                {
                    "query": next_query,
                    "episodes_human_readable": episode_lines,
                    "cached": cached,
                    "cached_from_query": cached_from_query,
                    "wall_time_seconds": tool_elapsed_seconds,
                    "reported_memory_retrieval_time": reported_memory_retrieval_time,
                    "memory_search_latency_seconds": search_latency_breakdown,
                }
            )
            response: dict[str, object] = {
                "episodes_returned": len(episodes),
                "query": next_query,
                "cached": cached,
                "episodes_human_readable": episode_lines,
                "wall_time_seconds": tool_elapsed_seconds,
                "reported_memory_retrieval_time": reported_memory_retrieval_time,
                "memory_search_latency_seconds": search_latency_breakdown,
            }
            if cached_from_query:
                response["cached_from_query"] = cached_from_query
            return response

        async def _tool_return_sub_skill_result(
            arguments: dict[str, object],
        ) -> dict[str, object]:
            nonlocal summary_from_tool
            summary_from_tool = str(arguments.get("summary") or "").strip()
            return {"summary_recorded": bool(summary_from_tool)}

        try:
            live_result = await self._session_model.run_live_session(
                system_prompt=(
                    "Use the attached sub-skill and available tools to resolve "
                    "the user request."
                ),
                user_prompt=user_prompt or f"sub-skill query: {query.query}",
                tools=sub_skill_tool_schemas(spec.allowed_tools),
                tool_registry={
                    "memmachine_search": _tool_memmachine_search,
                    "return_sub_skill_result": _tool_return_sub_skill_result,
                },
                max_turns=bounded_max_steps,
                timeout_seconds=float(spec.timeout_seconds),
                provider_skill_bundles=self._native_skill_bundles(spec=spec),
            )
        except SkillToolCallFormatError:
            self._raise_invalid_output(
                why="Sub-skill tool call payload shape was invalid.",
                fallback_reason="invalid_sub_skill_output",
            )
        except SkillToolNotFoundError:
            self._raise_invalid_output(
                why="Sub-skill requested unsupported tool.",
                fallback_reason="invalid_sub_skill_output",
            )
        except SkillSessionLimitError:
            self._raise_invalid_output(
                why="Sub-skill exceeded configured max steps/time budget.",
                fallback_reason="sub_skill_max_steps_exceeded",
            )
        except SkillLanguageModelError as err:
            self._raise_invalid_output(
                why=(
                    "Sub-skill session runtime failed: "
                    f"{self._format_language_model_error(err)}"
                ),
                fallback_reason="invalid_sub_skill_output",
            )

        result.episodes = collected_episodes
        if not result.episodes and "memmachine_search" in spec.allowed_tools:
            # Preserve existing behavior when no explicit tool calls are emitted.
            episodes, memory_metrics = await self._memory_tool.do_query(policy, query)
            memory_search_called += self._metric_as_int(
                memory_metrics,
                "memory_search_called",
            )
            memory_retrieval_time += self._metric_as_float(
                memory_metrics,
                "memory_retrieval_time",
            )
            result.episodes = episodes

        result.tool_calls = self._tool_calls_from_live_result(
            live_result=live_result,
            query=query,
            memmachine_call_details=memmachine_call_details,
        )
        if max_tool_calls is not None and len(result.tool_calls) > max_tool_calls:
            self._raise_invalid_output(
                why=(
                    "Sub-skill tool-call budget exceeded: "
                    f"{len(result.tool_calls)} > {max_tool_calls}."
                ),
                fallback_reason="session_call_budget_exceeded",
            )
        result.llm_time = float(live_result.llm_time_seconds)
        result.llm_call_count = int(live_result.turn_count)
        result.llm_input_tokens = int(live_result.llm_input_tokens)
        result.llm_output_tokens = int(live_result.llm_output_tokens)
        result.memory_search_called = memory_search_called
        result.memory_retrieval_time = memory_retrieval_time
        result.summary = summary_from_tool or live_result.final_response.strip()
        result.episodes = self._dedupe_episodes(result.episodes)
        result.normalization_warnings = list(live_result.normalization_warnings)
        result.status = "success"
        return result

    async def run(
        self,
        *,
        skill_name: str,
        policy: QueryPolicy,
        query: QueryParam,
        max_tool_calls: int | None = None,
    ) -> SubSkillExecutionResult:
        """Execute one sub-skill and return merged episodes + tool-call records."""
        spec = self._load_sub_skill_spec(skill_name)
        return await self._run_standard_skill(
            skill_name=skill_name,
            spec=spec,
            policy=policy,
            query=query,
            max_tool_calls=max_tool_calls,
        )
