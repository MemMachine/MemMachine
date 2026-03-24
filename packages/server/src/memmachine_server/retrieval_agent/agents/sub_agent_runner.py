"""Sub-agent execution runtime for markdown-guided retrieval-agent orchestration."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import ClassVar

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
from memmachine_server.retrieval_agent.agents.memory_search import (
    run_direct_memory_search,
)
from memmachine_server.retrieval_agent.agents.session_state import AgentToolCallRecord
from memmachine_server.retrieval_agent.agents.spec_loader import load_agent_spec
from memmachine_server.retrieval_agent.agents.tool_protocol import (
    parse_sub_agent_tool_call,
    sub_agent_tool_schemas,
)
from memmachine_server.retrieval_agent.agents.types import (
    AgentContractError,
    AgentContractErrorCode,
    AgentContractErrorPayload,
    AgentSpecV1,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryParam,
    QueryPolicy,
)


class SubAgentExecutionResult(BaseModel):
    """Structured result returned from a sub-agent execution."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    query: str
    status: str
    summary: str = ""
    episodes: list[Episode] = Field(default_factory=list)
    tool_calls: list[AgentToolCallRecord] = Field(default_factory=list)
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


class SubAgentRunner:
    """Run one sub-agent using markdown policy and memory-search tool access."""

    _COQ_ARTIFACT_QUERY_PATTERN = re.compile(
        r"(?i)(?:\bSKILL\.md\b|\.planning/|/tmp/|coq-[a-f0-9]{8,}|/skills/specs/)"
    )
    _COQ_INSTRUCTIONAL_QUERY_PATTERN = re.compile(
        r"(?i)\b(decompose|step\s*\d+|identify\b.*\bthen\b|first\b.*\bthen\b)\b"
    )
    _COQ_TERMINAL_ATTRIBUTE_SUFFIX_PATTERNS: tuple[str, ...] = (
        r"(?:\b(?:place of death|date of death|death date|cause of death|died where|died in|died|death)\b[\s,]*)+$",
        r"(?:\b(?:place of birth|birth place|birthplace|born where|born in|born|birth)\b[\s,]*)+$",
        r"(?:\b(?:nationality|country(?: of origin)?|from which country|same country|same nationality)\b[\s,]*)+$",
        r"(?:\b(?:works? at|workplace|employer|work)\b[\s,]*)+$",
        r"(?:\b(?:graduated from|graduated|education)\b[\s,]*)+$",
        r"(?:\b(?:award(?:s)?(?: won)?|prize(?: won)?)\b[\s,]*)+$",
        r"(?:\b(?:earlier|later|older|younger|first)\b[\s,]*)+$",
    )
    _COQ_ANCHOR_ROLE_TOKENS: ClassVar[frozenset[str]] = frozenset(
        {
            "director",
            "composer",
            "creator",
            "performer",
            "author",
            "producer",
            "founder",
        }
    )
    _COQ_TERMINAL_RELATION_TOKENS: ClassVar[frozenset[str]] = frozenset(
        {
            "father",
            "mother",
            "spouse",
            "husband",
            "wife",
            "child",
            "children",
            "son",
            "daughter",
        }
    )

    def __init__(
        self,
        *,
        model: LanguageModel,
        session_model: SkillSessionModelProtocol | None = None,
        spec_root: Path | None = None,
        native_skill_bundle_root: str | None = None,
    ) -> None:
        """Initialize sub-agent runtime dependencies."""
        self._model = model
        self._session_model = (
            session_model
            or SkillLanguageModel.from_openai_responses_language_model(model)
        )
        self._spec_root = spec_root or (
            Path(__file__).resolve().parent / "specs" / "sub_agents"
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

    @classmethod
    def _looks_like_non_query_artifact(cls, raw_query: str) -> bool:
        normalized = raw_query.strip()
        if not normalized:
            return True
        alpha_numeric_count = len(re.findall(r"[A-Za-z0-9]", normalized))
        if alpha_numeric_count < 3:
            return True
        return bool(cls._COQ_ARTIFACT_QUERY_PATTERN.search(normalized))

    @classmethod
    def _strip_terminal_attribute_suffixes(cls, raw_query: str) -> str:
        candidate = raw_query.strip()
        if not candidate:
            return candidate
        for _ in range(4):
            updated = candidate
            for pattern in cls._COQ_TERMINAL_ATTRIBUTE_SUFFIX_PATTERNS:
                updated = re.sub(pattern, "", updated, flags=re.IGNORECASE)
            updated = re.sub(r"[\s,;:.!?-]+$", "", updated).strip()
            if updated == candidate:
                break
            candidate = updated
        return re.sub(r"\s+", " ", candidate).strip()

    @classmethod
    def _rewrite_first_coq_hop_query(
        cls,
        *,
        original_query: str,
        candidate_query: str,
    ) -> str:
        normalized_original = re.sub(r"\s+", " ", original_query.strip())
        normalized_candidate = re.sub(r"\s+", " ", candidate_query.strip())
        if not normalized_candidate:
            return normalized_original or normalized_candidate
        if cls._looks_like_non_query_artifact(normalized_candidate):
            return normalized_original or normalized_candidate
        if cls._COQ_INSTRUCTIONAL_QUERY_PATTERN.search(normalized_candidate):
            return normalized_candidate

        rewritten = re.sub(
            r"(?i)\bpaternal\s+(grandmother|grandfather)\b",
            "father",
            normalized_candidate,
        )
        rewritten = re.sub(
            r"(?i)\bmaternal\s+(grandmother|grandfather)\b",
            "mother",
            rewritten,
        )
        rewritten = re.sub(r"(?i)\bstepmother\b", "father", rewritten)
        rewritten = re.sub(r"(?i)\bstepfather\b", "mother", rewritten)
        rewritten = cls._strip_terminal_attribute_suffixes(rewritten)

        words = rewritten.split()
        if words:
            trailing = words[-1].lower()
            lowered_words = {word.lower() for word in words[:-1]}
            if (
                trailing in cls._COQ_TERMINAL_RELATION_TOKENS
                and lowered_words & cls._COQ_ANCHOR_ROLE_TOKENS
            ):
                rewritten = " ".join(words[:-1]).strip()

        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if len(cls._query_tokens(rewritten)) < 2:
            return normalized_original or normalized_candidate
        return rewritten

    @classmethod
    def _rewrite_sub_agent_query(
        cls,
        *,
        agent_name: str,
        original_query: str,
        candidate_query: str,
        search_index: int,
    ) -> str:
        normalized_candidate = re.sub(r"\s+", " ", candidate_query.strip())
        if agent_name != "coq":
            return normalized_candidate or original_query.strip()

        if search_index <= 0:
            return cls._rewrite_first_coq_hop_query(
                original_query=original_query,
                candidate_query=normalized_candidate or original_query,
            )
        if cls._looks_like_non_query_artifact(normalized_candidate):
            return original_query.strip() or normalized_candidate
        return normalized_candidate or original_query.strip()

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

    @classmethod
    def _structured_summary_payload(  # noqa: C901
        cls,
        *,
        arguments: dict[str, object],
        original_query: str,
        observed_sub_queries: list[str],
    ) -> dict[str, object] | None:
        payload: dict[str, object] = {}
        raw_is_sufficient = arguments.get("is_sufficient")
        if isinstance(raw_is_sufficient, bool):
            payload["is_sufficient"] = raw_is_sufficient

        evidence_indices = cls._normalize_episode_indices(arguments.get("evidence_indices"))
        if evidence_indices:
            payload["evidence_indices"] = evidence_indices

        raw_new_query = arguments.get("new_query")
        if isinstance(raw_new_query, str) and raw_new_query.strip():
            payload["new_query"] = raw_new_query.strip()

        raw_confidence = arguments.get("confidence_score")
        if isinstance(raw_confidence, int | float) and not isinstance(
            raw_confidence,
            bool,
        ):
            payload["confidence_score"] = max(0.0, min(1.0, float(raw_confidence)))

        for key in ("reason_code", "reason_note", "answer_candidate"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                payload[key] = value.strip()

        stage_results = arguments.get("stage_results")
        if isinstance(stage_results, list):
            payload["stage_results"] = stage_results

        related_indices = cls._normalize_episode_indices(
            arguments.get("related_episode_indices")
        )
        if related_indices:
            payload["related_episode_indices"] = related_indices

        selected_indices = cls._normalize_episode_indices(
            arguments.get("selected_episode_indices")
        )
        if selected_indices:
            payload["selected_episode_indices"] = selected_indices

        explicit_sub_queries = cls._normalize_string_list(
            arguments.get("generated_sub_queries") or arguments.get("sub_queries")
        )
        if explicit_sub_queries:
            payload["generated_sub_queries"] = explicit_sub_queries
        elif observed_sub_queries:
            payload["generated_sub_queries"] = observed_sub_queries

        if isinstance(payload.get("is_sufficient"), bool) and "new_query" not in payload:
            payload["new_query"] = original_query.strip()

        return payload or None

    def _spec_path_for(self, agent_name: str) -> Path:
        candidates = [
            agent_name,
            agent_name.replace("-", "_"),
            agent_name.replace("_", "-"),
        ]
        for candidate in candidates:
            candidate_path = self._spec_root / f"{candidate}.md"
            if candidate_path.exists():
                return candidate_path
        return self._spec_root / f"{agent_name}.md"

    def _load_sub_agent_spec(self, agent_name: str) -> AgentSpecV1:
        spec = load_agent_spec(self._spec_path_for(agent_name))
        if spec.kind != "sub-agent":
            raise AgentContractError(
                code=AgentContractErrorCode.INVALID_SPEC,
                payload=AgentContractErrorPayload(
                    what_failed="Sub-agent spec kind mismatch",
                    why=f"Expected sub-agent, got {spec.kind}",
                    how_to_fix="Set kind: sub-agent in markdown frontmatter.",
                    where="agents.sub_agent_runner._load_sub_agent_spec",
                    fallback_trigger_reason="invalid_sub_agent_spec",
                ),
            )
        return spec

    def _native_skill_bundles(
        self,
        *,
        spec: AgentSpecV1,
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
        raise AgentContractError(
            code=AgentContractErrorCode.INVALID_OUTPUT,
            payload=AgentContractErrorPayload(
                what_failed="Sub-agent function call payload invalid",
                why=why,
                how_to_fix="Return list[dict] function calls with valid arguments.",
                where="agents.sub_agent_runner.run",
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
    ) -> list[AgentToolCallRecord]:
        records: list[AgentToolCallRecord] = []
        call_details = memmachine_call_details or []
        memmachine_call_index = 0
        for step_index, execution in enumerate(live_result.tool_executions, start=1):
            action = parse_sub_agent_tool_call(
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
                    AgentToolCallRecord(
                        step=step_index,
                        tool_name=action.action,
                        arguments=call_arguments,
                        status="success",
                        result_summary=result_summary,
                        raw_result=self._sanitize_raw_tool_result(execution.output),
                    )
                )
                continue
            records.append(
                AgentToolCallRecord(
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
        agent_name: str,
        spec: AgentSpecV1,
        policy: QueryPolicy,
        query: QueryParam,
        user_prompt: str | None = None,
        max_tool_calls: int | None = None,
    ) -> SubAgentExecutionResult:
        bounded_max_steps = max(1, spec.max_steps)
        result = SubAgentExecutionResult(
            agent_name=agent_name,
            query=query.query,
            status="in_progress",
        )
        collected_episodes: list[Episode] = []
        memmachine_call_details: list[dict[str, object]] = []
        cached_query_results: list[dict[str, object]] = []
        memory_search_called = 0
        memory_retrieval_time = 0.0

        async def _tool_memmachine_search(  # noqa: C901
            arguments: dict[str, object],
        ) -> dict[str, object]:
            nonlocal memory_search_called, memory_retrieval_time
            raw_next_query = str(arguments.get("query") or query.query)
            search_index = len(memmachine_call_details)
            next_query = self._rewrite_sub_agent_query(
                agent_name=agent_name,
                original_query=query.query,
                candidate_query=raw_next_query,
                search_index=search_index,
            )
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
                _ = policy
                episodes, memory_metrics = await run_direct_memory_search(next_param)
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
                    "rewritten_from_query": (
                        raw_next_query if raw_next_query.strip() != next_query else None
                    ),
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
            if raw_next_query.strip() != next_query:
                response["rewritten_from_query"] = raw_next_query
            if cached_from_query:
                response["cached_from_query"] = cached_from_query
            return response

        try:
            live_result = await self._session_model.run_live_session(
                system_prompt=(
                    "Use the attached sub-agent and available tools to resolve "
                    "the user request."
                ),
                user_prompt=user_prompt or f"sub-agent query: {query.query}",
                tools=sub_agent_tool_schemas(spec.allowed_tools),
                tool_registry={
                    "memmachine_search": _tool_memmachine_search,
                },
                max_turns=bounded_max_steps,
                timeout_seconds=float(spec.timeout_seconds),
                provider_skill_bundles=self._native_skill_bundles(spec=spec),
            )
        except SkillToolCallFormatError:
            self._raise_invalid_output(
                why="Sub-skill tool call payload shape was invalid.",
                fallback_reason="invalid_sub_agent_output",
            )
        except SkillToolNotFoundError:
            self._raise_invalid_output(
                why="Sub-skill requested unsupported tool.",
                fallback_reason="invalid_sub_agent_output",
            )
        except SkillSessionLimitError:
            self._raise_invalid_output(
                why="Sub-skill exceeded configured max steps/time budget.",
                fallback_reason="sub_agent_max_steps_exceeded",
            )
        except SkillLanguageModelError as err:
            self._raise_invalid_output(
                why=(
                    "Sub-agent session runtime failed: "
                    f"{self._format_language_model_error(err)}"
                ),
                fallback_reason="invalid_sub_agent_output",
            )

        result.episodes = collected_episodes
        if not result.episodes and "memmachine_search" in spec.allowed_tools:
            # Preserve existing behavior when no explicit tool calls are emitted.
            _ = policy
            episodes, memory_metrics = await run_direct_memory_search(query)
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
                    "Sub-agent tool-call budget exceeded: "
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
        result.summary = live_result.final_response.strip()
        result.episodes = self._dedupe_episodes(result.episodes)
        result.normalization_warnings = list(live_result.normalization_warnings)
        result.status = "success"
        return result

    async def run(
        self,
        *,
        agent_name: str,
        policy: QueryPolicy,
        query: QueryParam,
        max_tool_calls: int | None = None,
    ) -> SubAgentExecutionResult:
        """Execute one sub-agent and return merged episodes + tool-call records."""
        spec = self._load_sub_agent_spec(agent_name)
        return await self._run_standard_skill(
            agent_name=agent_name,
            spec=spec,
            policy=policy,
            query=query,
            max_tool_calls=max_tool_calls,
        )
