"""Top-level markdown-guided retrieval orchestration runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from memmachine_common import SkillRunner, install_skill
from memmachine_common.skill_loop import SkillLoopContractError

from memmachine_server.common.configuration.retrieval_config import (
    RetrievalAgentConf,
    RetrievalAgentSessionProvider,
)
from memmachine_server.common.episode_store import Episode
from memmachine_server.common.language_model import (
    SkillLanguageModelError as AgentLanguageModelError,
)
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_agent.agents.memory_search import (
    DIRECT_MEMORY_SELECTED_AGENT,
    DIRECT_MEMORY_SELECTED_AGENT_NAME,
    run_direct_memory_search,
)
from memmachine_server.retrieval_agent.agents.runtime import (
    build_agent_request,
    fallback_for_downstream_error,
)
from memmachine_server.retrieval_agent.agents.session_state import (
    TopLevelAgentSessionState,
)
from memmachine_server.retrieval_agent.agents.spec_loader import load_agent_spec
from memmachine_server.retrieval_agent.agents.tool_protocol import (
    CANONICAL_SUB_AGENT_NAMES,
)
from memmachine_server.retrieval_agent.agents.types import (
    AgentContractError,
    AgentContractErrorCode,
    AgentContractErrorPayload,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryParam,
    QueryPolicy,
    RetrievalAgentParams,
    rerank_episodes,
)

DEFAULT_RETRIEVE_AGENT_SPEC = (
    Path(__file__).resolve().parent / "specs" / "top_level" / "retrieve_agent.md"
)
DEFAULT_SUB_AGENT_SPEC_ROOT = Path(__file__).resolve().parent / "specs" / "sub_agents"
DEFAULT_AGENT_INSTALL_ROOT = Path(__file__).resolve().parent / "specs"


class _RunnerMemoryProxy:
    """Episodic-memory proxy that enforces server retrieval-agent search budgets."""

    def __init__(
        self,
        *,
        memory: EpisodicMemory,
        max_combined_calls: int,
    ) -> None:
        self._memory = memory
        self._max_combined_calls = max_combined_calls
        self.call_count = 0

    @property
    def session_key(self) -> str:
        return self._memory.session_key

    async def query_memory(self, *args: object, **kwargs: object) -> object:
        next_count = self.call_count + 1
        if next_count > self._max_combined_calls:
            raise AgentContractError(
                code=AgentContractErrorCode.INVALID_OUTPUT,
                payload=AgentContractErrorPayload(
                    what_failed="Top-level tool-call contract validation failed",
                    why=(
                        "Combined tool-call budget exceeded: "
                        f"next={next_count} > limit={self._max_combined_calls}; "
                        "source=memmachine_search."
                    ),
                    how_to_fix=(
                        "Emit only allowed memmachine_search calls with valid "
                        "arguments, then finish with plain assistant text."
                    ),
                    where="agents.retrieve_agent.do_query",
                    fallback_trigger_reason="session_call_budget_exceeded",
                ),
            )
        self.call_count = next_count
        return await self._memory.query_memory(*args, **kwargs)


class RetrievalAgent:
    """Top-level orchestrator that enforces markdown-driven action contracts."""

    def __init__(self, param: RetrievalAgentParams) -> None:
        """Initialize top-level markdown policy and session runtime."""
        self._model = param.model
        if self._model is None:
            raise ValueError("RetrievalAgent requires a language model.")
        self._reranker = param.reranker
        self._extra_params = param.extra_params or {}
        raw_retrieval_conf = self._extra_params.get("retrieval_conf")
        if raw_retrieval_conf is not None and not isinstance(
            raw_retrieval_conf,
            RetrievalAgentConf,
        ):
            raise ValueError(
                "RetrievalAgent extra_params['retrieval_conf'] must be RetrievalAgentConf."
            )
        self._retrieval_conf = cast(RetrievalAgentConf | None, raw_retrieval_conf)
        raw_spec = self._extra_params.get("agent_spec", DEFAULT_RETRIEVE_AGENT_SPEC)
        self._spec = load_agent_spec(raw_spec)
        self._max_guardrail_retries = 1
        self._max_hops = int(self._extra_params.get("max_hops", 4))
        self._max_branches = int(self._extra_params.get("max_branches", 4))
        self._global_timeout_seconds = int(
            self._extra_params.get("global_timeout_seconds", self._spec.timeout_seconds)
        )
        self._max_combined_calls = int(self._extra_params.get("max_combined_calls", 10))
        self._sub_agent_timeout_seconds = int(
            self._extra_params.get("sub_agent_timeout_seconds", 120)
        )
        self._sub_agent_episode_line_cap = int(
            self._extra_params.get("sub_agent_episode_line_cap", 120)
        )
        raw_adaptive_search_limit = self._extra_params.get("adaptive_search_limit")
        if (
            isinstance(raw_adaptive_search_limit, dict)
            and isinstance(raw_adaptive_search_limit.get("initial"), int)
            and isinstance(raw_adaptive_search_limit.get("escalated"), int)
        ):
            self._adaptive_search_limit = {
                "initial": int(raw_adaptive_search_limit["initial"]),
                "escalated": int(raw_adaptive_search_limit["escalated"]),
            }
        else:
            self._adaptive_search_limit = None
        raw_max_episode_chars = self._extra_params.get("max_episode_chars")
        if isinstance(raw_max_episode_chars, int) and raw_max_episode_chars > 0:
            self._max_episode_chars = raw_max_episode_chars
        else:
            self._max_episode_chars = None
        self._early_exit_confidence = bool(
            self._extra_params.get("early_exit_confidence", False)
        )
        self._query_dedup = bool(self._extra_params.get("query_dedup", False))
        self._stage_result_mode = bool(
            self._extra_params.get("stage_result_mode", False)
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
        self._omit_episode_text_on_confident_stage_result = bool(
            self._extra_params.get(
                "omit_episode_text_on_confident_stage_result",
                False,
            )
        )
        raw_available_sub_agents = self._extra_params.get(
            "available_sub_agents",
            list(CANONICAL_SUB_AGENT_NAMES),
        )
        if not isinstance(raw_available_sub_agents, list) or any(
            not isinstance(agent_name, str) for agent_name in raw_available_sub_agents
        ):
            raise ValueError(
                "RetrievalAgent extra_params['available_sub_agents'] must be a list[str]."
            )
        invalid_sub_agents = [
            agent_name
            for agent_name in raw_available_sub_agents
            if agent_name not in CANONICAL_SUB_AGENT_NAMES
        ]
        if invalid_sub_agents:
            raise ValueError(
                "RetrievalAgent extra_params['available_sub_agents'] must only "
                f"contain {list(CANONICAL_SUB_AGENT_NAMES)}. "
                f"Invalid: {invalid_sub_agents}"
            )
        self._available_sub_agents = list(dict.fromkeys(raw_available_sub_agents))
        if not self._available_sub_agents:
            raise ValueError(
                "RetrievalAgent extra_params['available_sub_agents'] cannot be empty."
            )

        raw_sub_agent_root = self._extra_params.get("sub_agent_spec_root")
        if raw_sub_agent_root is None:
            self._sub_agent_spec_root = DEFAULT_SUB_AGENT_SPEC_ROOT
        else:
            self._sub_agent_spec_root = Path(str(raw_sub_agent_root))
        raw_agent_install_root = self._extra_params.get(
            "agent_install_root",
            DEFAULT_AGENT_INSTALL_ROOT,
        )
        self._agent_install_root = Path(str(raw_agent_install_root))
        raw_agent_install_cache_path = self._extra_params.get("agent_install_cache_path")
        self._agent_install_cache_path = (
            Path(str(raw_agent_install_cache_path))
            if raw_agent_install_cache_path is not None
            else Path(tempfile.mkdtemp(prefix="memmachine_server_agent_"))
            / "agent_cache.json"
        )
        self._installed_agent: object | None = None
        self._provider_client: object | None = None
        self._provider_name: str | None = None
        self._provider_model_name: str | None = None
        self._installed_agent_lock = asyncio.Lock()

    @property
    def agent_name(self) -> str:
        return "RetrievalAgent"

    @property
    def agent_description(self) -> str:
        return (
            "Top-level retrieval agent driven by markdown policy "
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

    def _new_session_state(self, query: QueryParam) -> TopLevelAgentSessionState:
        session = TopLevelAgentSessionState.new(
            route_name=self._spec.route_name,
            policy_name=self._spec.name,
            query=query.query,
        )
        session.record_event(
            actor="top-level",
            event_type="session_started",
            detail="Top-level retrieve-agent session initialized.",
        )
        return session

    def _resolve_provider_runtime(self) -> tuple[str, object, str]:
        if self._provider_name is not None:
            if self._provider_client is None or self._provider_model_name is None:
                raise RuntimeError("RetrievalAgent provider runtime cache is incomplete.")
            return (
                self._provider_name,
                self._provider_client,
                self._provider_model_name,
            )

        retrieval_conf = self._retrieval_conf
        if (
            retrieval_conf is not None
            and retrieval_conf.agent_session_provider
            == RetrievalAgentSessionProvider.ANTHROPIC
        ):
            try:
                import anthropic
            except ImportError as err:
                raise RuntimeError(
                    "Anthropic retrieval-agent runtime requires the 'anthropic' package."
                ) from err

            api_key = None
            if retrieval_conf.anthropic_api_key is not None:
                api_key = retrieval_conf.anthropic_api_key.get_secret_value().strip()
            if not api_key:
                raise ValueError(
                    "Anthropic retrieval-agent runtime requires anthropic_api_key."
                )
            client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=retrieval_conf.anthropic_base_url,
            )
            self._provider_name = "anthropic"
            self._provider_client = client
            self._provider_model_name = retrieval_conf.anthropic_model
            return (
                self._provider_name,
                self._provider_client,
                self._provider_model_name,
            )

        client = getattr(self._model, "client", None)
        model_name = getattr(self._model, "model_name", None)
        if (
            callable(getattr(getattr(client, "responses", None), "create", None))
            and isinstance(model_name, str)
            and model_name.strip()
        ):
            self._provider_name = "openai"
            self._provider_client = client
            self._provider_model_name = model_name
            return (
                self._provider_name,
                self._provider_client,
                self._provider_model_name,
            )

        raise TypeError(
            "RetrievalAgent requires an OpenAI language model exposing client/model_name "
            "or retrieval_conf.agent_session_provider=anthropic."
        )

    async def _ensure_installed_agent(self) -> object:
        if self._installed_agent is not None:
            return self._installed_agent

        async with self._installed_agent_lock:
            if self._installed_agent is not None:
                return self._installed_agent
            provider, client, _ = self._resolve_provider_runtime()
            install_kwargs: dict[str, object] = {
                "cache_path": self._agent_install_cache_path,
            }
            if provider == "openai":
                install_kwargs["openai_client"] = client
            else:
                install_kwargs["anthropic_client"] = client
            install_name_key = "skill" + "_name"
            install_kwargs[install_name_key] = "retrieve-agent"
            self._installed_agent = await install_skill(
                self._agent_install_root,
                provider,  # type: ignore[arg-type]
                **install_kwargs,
            )
        return self._installed_agent

    async def _build_runner(
        self,
        query: QueryParam,
        *,
        direct_memory: object | None = None,
    ) -> SkillRunner:
        provider, client, model_name = self._resolve_provider_runtime()
        installed_agent = await self._ensure_installed_agent()
        tool_choice = self._extra_params.get("tool_choice", "auto")
        score_threshold = (
            query.score_threshold if query.score_threshold != -float("inf") else None
        )
        return SkillRunner(
            cast(Any, installed_agent),
            client=client,
            model=model_name,
            provider=cast(Any, provider),
            search_mode="direct",
            direct_memory=direct_memory or query.memory,
            direct_search_extra_kwargs={
                "property_filter": query.property_filter,
                "mode": EpisodicMemory.QueryMode.BOTH,
            },
            max_turns=self._spec.max_steps,
            search_limit=query.limit if query.limit > 0 else 20,
            expand_context=query.expand_context,
            score_threshold=score_threshold,
            adaptive_search_limit=self._adaptive_search_limit,
            max_episode_chars=self._max_episode_chars,
            early_exit_confidence=self._early_exit_confidence,
            query_dedup=self._query_dedup,
            stage_result_mode=self._stage_result_mode,
            stage_result_confidence_threshold=self._stage_result_confidence_threshold,
            omit_episode_text_on_confident_stage_result=(
                self._omit_episode_text_on_confident_stage_result
            ),
            tool_choice=cast(str | dict[str, str], tool_choice),
        )

    @classmethod
    def _episodes_from_runner_raw_result(
        cls,
        *,
        query: QueryParam,
        raw_result: object,
    ) -> list[Episode]:
        if raw_result is None:
            return []
        payload = cls._as_dict_for_episodes(raw_result)
        long_term = cls._as_dict_for_episodes(payload.get("long_term_memory"))
        raw_episodes = long_term.get("episodes", [])
        if not isinstance(raw_episodes, list):
            return []
        episodes: list[Episode] = []
        for raw_episode in raw_episodes:
            episode = cls._as_dict_for_episodes(raw_episode)
            uid = episode.get("uid")
            content = episode.get("content")
            if not isinstance(uid, str) or not isinstance(content, str):
                continue
            episodes.append(
                Episode(
                    uid=uid,
                    content=content,
                    session_key=query.memory.session_key,
                    created_at=cast(
                        datetime,
                        episode.get("created_at") or datetime.now(tz=UTC),
                    ),
                    producer_id=cast(str | None, episode.get("producer_id")),
                    producer_role=cast(str | None, episode.get("producer_role")),
                    produced_for_id=cast(str | None, episode.get("produced_for_id")),
                    metadata=cast(dict[str, object] | None, episode.get("metadata")),
                )
            )
        return episodes

    @staticmethod
    def _as_dict_for_episodes(raw_value: object) -> dict[str, object]:
        if isinstance(raw_value, dict):
            return raw_value
        model_dump = getattr(raw_value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json")
            if isinstance(dumped, dict):
                return {str(key): value for key, value in dumped.items()}
        raw_dict = getattr(raw_value, "__dict__", None)
        if isinstance(raw_dict, dict):
            return {str(key): value for key, value in raw_dict.items()}
        return {}

    def _augment_metrics_with_session_state(
        self,
        *,
        metrics: dict[str, object],
        session: TopLevelAgentSessionState,
    ) -> dict[str, object]:
        metrics["orchestrator_step_count"] = session.current_step
        metrics["orchestrator_event_count"] = len(session.events)
        metrics["orchestrator_tool_call_count"] = len(session.tool_calls)
        metrics["orchestrator_sub_agent_count"] = len(session.sub_agent_runs)
        metrics["orchestrator_episode_count"] = len(session.merged_episodes)
        metrics["orchestrator_completed"] = session.completed
        metrics["orchestrator_policy_kind"] = self._spec.kind
        metrics["orchestrator_state_snapshot"] = session.prompt_snapshot()
        metrics["orchestrator_final_response"] = session.final_response
        metrics["orchestrator_trace"] = session.full_trace_snapshot()
        metrics["orchestrator_sub_agent_runs"] = [
            run.model_dump(mode="json") for run in session.sub_agent_runs
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
        session: TopLevelAgentSessionState,
        aggregated_metrics: dict[str, Any],
        err: Exception,
        context: str,
        mapped_contract_error: AgentContractError | None = None,
    ) -> None:
        diagnostics: dict[str, object] = {
            "context": context,
            "error_type": type(err).__name__,
            "error_message": str(err),
        }
        if isinstance(err, AgentLanguageModelError):
            provider_diag = getattr(err, "diagnostics", None)
            if isinstance(provider_diag, dict) and provider_diag:
                diagnostics["provider_diagnostics"] = provider_diag
                response_body = provider_diag.get("response_body")
                if isinstance(response_body, str) and response_body:
                    aggregated_metrics["provider_error_raw_response"] = response_body
        if isinstance(err, AgentContractError):
            diagnostics["contract_error"] = err.to_dict()
            aggregated_metrics["agent_contract_error_payload"] = err.to_dict()
        if mapped_contract_error is not None:
            diagnostics["mapped_contract_error"] = mapped_contract_error.to_dict()
            aggregated_metrics["agent_contract_error_payload"] = (
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

    def _contract_error(self, *, why: str, fallback_reason: str) -> AgentContractError:
        return AgentContractError(
            code=AgentContractErrorCode.INVALID_OUTPUT,
            payload=AgentContractErrorPayload(
                what_failed="Top-level tool-call contract validation failed",
                why=why,
                how_to_fix=(
                    "Emit only allowed memmachine_search calls with valid "
                    "arguments, then finish with plain assistant text."
                ),
                where="agents.retrieve_agent.do_query",
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

    def _record_sub_agent_summary_metrics(
        self,
        *,
        session: TopLevelAgentSessionState,
        aggregated_metrics: dict[str, Any],
        agent_name: str,
        summary: str,
    ) -> dict[str, object] | None:
        stripped = summary.strip()
        if not stripped:
            return None

        payload = self._summary_payload(stripped)
        records = aggregated_metrics.get("sub_agent_summaries")
        if not isinstance(records, list):
            records = []
            aggregated_metrics["sub_agent_summaries"] = records
        record: dict[str, object] = {
            "agent_name": agent_name,
            "summary": stripped[:1000],
        }
        if isinstance(payload, dict):
            record["summary_payload"] = payload
        records.append(record)

        if isinstance(payload, dict):
            self._apply_sub_agent_sufficiency_signal(
                session=session,
                aggregated_metrics=aggregated_metrics,
                agent_name=agent_name,
                summary_payload=payload,
            )
        return payload

    def _apply_sub_agent_sufficiency_signal(  # noqa: C901
        self,
        *,
        session: TopLevelAgentSessionState,
        aggregated_metrics: dict[str, Any],
        agent_name: str,
        summary_payload: dict[str, object],
    ) -> None:
        is_sufficient = summary_payload.get("is_sufficient")
        if not isinstance(is_sufficient, bool):
            return

        aggregated_metrics["sufficiency_signal_seen"] = True
        aggregated_metrics["latest_sufficiency_signal_agent"] = agent_name
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

        if agent_name == "coq":
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
            event_type="sub_agent_sufficiency_signal",
            detail=(
                f"agent={agent_name}; "
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
    def _selected_agent_name_for_agent(agent_name: str) -> str:
        if agent_name == "coq":
            return "ChainOfQueryAgent"
        return DIRECT_MEMORY_SELECTED_AGENT_NAME

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
                    producer_id="retrieve-agent-stage-result",
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
                    producer_id="retrieve-agent-stage-result",
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
        reranked = await rerank_episodes(
            query=query,
            episodes=episodes,
            reranker=self._reranker,
        )
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
        session: TopLevelAgentSessionState,
        reason: str,
        code: str,
        aggregated_metrics: dict[str, Any] | None = None,
    ) -> tuple[list[Episode], dict[str, object]]:
        session.next_step()
        _ = policy
        fallback_episodes, fallback_metrics = await run_direct_memory_search(query)
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
        metrics["agent"] = self.agent_name
        metrics["route"] = self.agent_name
        selected_agent = metrics.get("selected_agent")
        if not isinstance(selected_agent, str) or not selected_agent.strip():
            selected_agent = "direct_memory"
            metrics["selected_agent"] = selected_agent
        metrics["selected_agent_name"] = self._selected_agent_name_for_agent(
            selected_agent
        )
        metrics["fallback_trigger_reason"] = reason
        metrics["agent_contract_error_code"] = code
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

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, object]]:
        route_name = self._spec.route_name
        session = self._new_session_state(query)
        aggregated_metrics: dict[str, Any] = {
            "session_call_budget_limit": self._max_combined_calls,
            "session_call_budget_consumed": 0,
        }
        memory_proxy = _RunnerMemoryProxy(
            memory=query.memory,
            max_combined_calls=self._max_combined_calls,
        )

        try:
            _ = build_agent_request(query, route_name=route_name)
            runner = await self._build_runner(query, direct_memory=memory_proxy)
            partial_response_text = (
                "Partial result: memmachine_search loop reached max_turns."
            )
            retrieval_started = time.perf_counter()
            async with asyncio.timeout(float(self._global_timeout_seconds)):
                raw_final_response = (await runner.run(query.query)).strip()
                if runner.last_memory_search_called == 0:
                    session.record_event(
                        actor="top-level",
                        event_type="default_memmachine_search",
                        detail=(
                            "No function calls returned; performed direct memory search."
                        ),
                    )
                    await runner.search(query.query)
            retrieval_duration = time.perf_counter() - retrieval_started

            aggregated_metrics["session_call_budget_consumed"] = memory_proxy.call_count
            aggregated_metrics["memory_search_called"] = runner.last_memory_search_called
            aggregated_metrics["memory_search_latency_seconds"] = list(
                runner.last_memory_search_latency_seconds
            )
            aggregated_metrics["memory_retrieval_time"] = float(
                sum(runner.last_memory_search_latency_seconds)
            )
            aggregated_metrics["llm_time"] = max(
                retrieval_duration - float(aggregated_metrics["memory_retrieval_time"]),
                0.0,
            )
            aggregated_metrics["llm_call_count"] = runner.last_llm_call_count
            aggregated_metrics["input_token"] = (
                runner.last_initial_input_tokens + runner.last_follow_up_input_tokens
            )
            aggregated_metrics["output_token"] = (
                runner.last_initial_output_tokens + runner.last_follow_up_output_tokens
            )
            aggregated_metrics["top_level_session_invocation_count"] = 1
            aggregated_metrics["top_level_session_turn_count"] = (
                runner.last_llm_call_count
            )
            aggregated_metrics["top_level_llm_call_count"] = (
                runner.last_llm_call_count
            )
            aggregated_metrics["top_level_input_token"] = aggregated_metrics[
                "input_token"
            ]
            aggregated_metrics["top_level_output_token"] = aggregated_metrics[
                "output_token"
            ]
            aggregated_metrics["selected_route"] = DIRECT_MEMORY_SELECTED_AGENT
            aggregated_metrics["selected_agent"] = DIRECT_MEMORY_SELECTED_AGENT
            aggregated_metrics["selected_agent_name"] = DIRECT_MEMORY_SELECTED_AGENT_NAME
            if runner.last_stage_results:
                aggregated_metrics["stage_results"] = list(runner.last_stage_results)
                aggregated_metrics["latest_stage_results"] = list(
                    runner.last_stage_results
                )
            if runner.last_stage_sub_queries:
                aggregated_metrics["stage_sub_queries"] = list(
                    runner.last_stage_sub_queries
                )
                aggregated_metrics["latest_stage_sub_queries"] = list(
                    runner.last_stage_sub_queries
                )

            for payload, raw_result, elapsed_seconds in zip(
                runner.last_search_results,
                runner.last_raw_search_results,
                runner.last_memory_search_latency_seconds,
                strict=False,
            ):
                session.next_step()
                episodes = self._episodes_from_runner_raw_result(
                    query=query,
                    raw_result=raw_result,
                )
                session.merge_episodes(episodes)
                trace_payload: dict[str, object] = {
                    "episodes_returned": int(payload.get("count", 0)),
                    "query": payload.get("query", query.query),
                    "episodes_human_readable": str(
                        payload.get("episodes_text", "")
                    ).splitlines(),
                    "wall_time_seconds": elapsed_seconds,
                    "reported_memory_retrieval_time": elapsed_seconds,
                    "memory_search_latency_seconds": [elapsed_seconds],
                }
                session.record_tool_call(
                    tool_name="memmachine_search",
                    arguments={"query": payload.get("query", query.query)},
                    status="success",
                    result_summary=(
                        f"episodes={int(payload.get('count', 0))}"
                    ),
                    raw_result=self._sanitize_tool_raw_result(trace_payload),
                )
                session.record_event(
                    actor="top-level",
                    event_type="memmachine_search_completed",
                    detail=(
                        f"step={session.current_step}; "
                        f"episodes={int(payload.get('count', 0))}"
                    ),
                )

            final_response = raw_final_response or "Top-level retrieval complete."
            top_level_is_sufficient = (
                bool(raw_final_response)
                and raw_final_response != partial_response_text
                and not self._response_looks_inconclusive(raw_final_response)
            )
            aggregated_metrics["top_level_sufficiency_signal_seen"] = bool(
                raw_final_response
            )
            aggregated_metrics["top_level_is_sufficient"] = top_level_is_sufficient
            aggregated_metrics["latest_sufficiency_signal"] = top_level_is_sufficient
            aggregated_metrics["latest_sufficiency_signal_agent"] = "retrieve-agent"
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
            metrics["agent"] = self.agent_name
            metrics["route"] = self.agent_name
            metrics["agent_name"] = self._spec.name
            metrics.setdefault("branch_total", 0)
            metrics.setdefault("branch_success_count", 0)
            metrics.setdefault("branch_failure_count", 0)
            metrics.setdefault("branch_retry_count", 0)
            final_episodes, rerank_applied = await self._finalize_episodes(
                query=query,
                episodes=session.merged_episodes,
            )
            metrics["stage_result_memory_returned"] = False
            metrics["rerank_applied"] = rerank_applied
            metrics["final_episode_count"] = len(final_episodes)
            return final_episodes, self._augment_metrics_with_session_state(
                metrics=metrics,
                session=session,
            )

        except TimeoutError as err:
            contract_error = self._contract_error(
                why=f"Top-level installed agent session exceeded timeout: {err}",
                fallback_reason="global_timeout",
            )
            self._record_error_diagnostics(
                session=session,
                aggregated_metrics=aggregated_metrics,
                err=contract_error,
                context="top_level_timeout",
            )
            return await self._fallback_with_reason(
                policy=policy,
                query=query,
                session=session,
                reason=contract_error.payload.fallback_trigger_reason,
                code=contract_error.code,
                aggregated_metrics=aggregated_metrics,
            )
        except SkillLoopContractError as err:
            contract_error = self._contract_error(
                why=str(err),
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
        except AgentContractError as err:
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
                where="agents.retrieve_agent.do_query",
                error=err,
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
