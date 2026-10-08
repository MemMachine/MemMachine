"""Progressive retrieval with bounded rounds and a UUID-deduplicated pool."""

from __future__ import annotations

import re
import time
from typing import Annotated, Any, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from memmachine_server.common.configuration.retrieval_config import (
    ProgressiveRetrievalConf,
)
from memmachine_server.common.episode_store import Episode
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.retrieval_agent.agents.coq_agent import (
    COMBINED_SUFFICIENCY_AND_REWRITE_PROMPT,
)
from memmachine_server.retrieval_agent.agents.memmachine_retriever import (
    MemMachineAgent,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)


class _RetrievalDecision(BaseModel):
    """Strict model output; malformed decisions never establish sufficiency."""

    model_config = ConfigDict(strict=True, extra="forbid")

    is_sufficient: bool
    evidence_indices: list[Annotated[int, Field(ge=0)]]
    new_query: str
    confidence_score: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator("evidence_indices")
    @classmethod
    def unique_indices(cls, indices: list[int]) -> list[int]:
        if len(indices) != len(set(indices)):
            raise ValueError("Evidence indices must be unique")
        return indices


class ProgressiveQueryAgent(AgentToolBase):
    """Widen retrieval until selected evidence suffices or a bound is reached.

    The candidate ceiling bounds the admitted pool, including expanded context.
    It does not bound backend computation, raw results, or model token usage.
    """

    def __init__(
        self, param: AgentToolBaseParam, config: ProgressiveRetrievalConf
    ) -> None:
        """Require a model and one direct memory tool to avoid unbounded fanout."""
        super().__init__(param)
        if self._model is None:
            raise ValueError("Model is not set")
        if len(self._children_tools) != 1 or not isinstance(
            self._children_tools[0], MemMachineAgent
        ):
            raise ValueError("Progressive retrieval requires one MemMachineAgent child")
        self._config = config.model_copy(deep=True)

    @property
    def agent_name(self) -> str:
        return "ProgressiveQueryAgent"

    @property
    def agent_description(self) -> str:
        return "Progressively widen memory retrieval and return supporting evidence."

    @property
    def accuracy_score(self) -> int:
        return 10

    @property
    def token_cost(self) -> int:
        return 9

    @property
    def time_cost(self) -> int:
        return 10

    def _init_metrics(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "queries": [],
            "retrieval_limits": [],
            "rounds": [],
            "is_sufficient": [],
            "confidence_scores": [],
            "memory_retrieval_time": 0.0,
            "memory_search_called": 0,
            "llm_time": 0.0,
            "input_token": 0,
            "output_token": 0,
            "raw_candidates": 0,
            "assessed_sufficient": None,
        }

    @staticmethod
    def _parse_decision(
        response: str | None, candidate_count: int
    ) -> _RetrievalDecision | None:
        if not response:
            return None
        text = response.strip()
        fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.DOTALL)
        if fence is not None:
            text = fence.group(1)
        try:
            decision = _RetrievalDecision.model_validate_json(text)
        except ValidationError:
            return None
        if any(index >= candidate_count for index in decision.evidence_indices):
            return None
        return decision

    async def _retrieve(
        self, policy: QueryPolicy, query: QueryParam, metrics: dict[str, Any]
    ) -> tuple[list[Episode], dict[str, Any]]:
        start = time.perf_counter()
        episodes, _ = await self._children_tools[0].do_query(policy, query)
        elapsed = time.perf_counter() - start
        round_metrics = {
            "query": query.query,
            "retrieval_limit": query.limit,
            "raw_candidates": len(episodes),
            "memory_retrieval_time": elapsed,
            "memory_search_called": 1,
            "llm_time": 0.0,
            "input_token": 0,
            "output_token": 0,
        }
        metrics["rounds"].append(round_metrics)
        metrics["queries"].append(query.query)
        metrics["retrieval_limits"].append(query.limit)
        metrics["memory_retrieval_time"] += elapsed
        metrics["memory_search_called"] += 1
        metrics["raw_candidates"] += len(episodes)
        return episodes, round_metrics

    async def _check(
        self,
        query: QueryParam,
        pool: dict[UUID, Episode],
        metrics: dict[str, Any],
        round_metrics: dict[str, Any],
    ) -> _RetrievalDecision | None:
        context = "".join(
            f"[{index}] {episodes_to_string([episode])}"
            for index, episode in enumerate(pool.values())
        )
        prompt = COMBINED_SUFFICIENCY_AND_REWRITE_PROMPT.format(
            original_query=query.query,
            used_query="\n".join(metrics["queries"]),
            retrieved_episodes=context,
        )
        model = cast(LanguageModel, self._model)
        start = time.perf_counter()
        (
            response,
            _,
            input_token,
            output_token,
        ) = await model.generate_response_with_token_usage(user_prompt=prompt)
        usage = {
            "llm_time": time.perf_counter() - start,
            "input_token": input_token,
            "output_token": output_token,
        }
        round_metrics.update(usage)
        for key, value in usage.items():
            metrics[key] += value
        decision = self._parse_decision(response, len(pool))
        metrics["assessed_sufficient"] = (
            decision.is_sufficient if decision is not None else None
        )
        if decision is not None:
            metrics["is_sufficient"].append(decision.is_sufficient)
            metrics["confidence_scores"].append(decision.confidence_score)
        return decision

    def _stop_reason(
        self, decision: _RetrievalDecision, candidate_count: int, return_cap: int
    ) -> str | None:
        if decision.is_sufficient:
            if not decision.evidence_indices:
                return "invalid_decision"
            if decision.confidence_score >= self._config.confidence_threshold:
                if len(decision.evidence_indices) > return_cap:
                    return "return_limit"
                return "sufficient"
        if candidate_count >= self._config.max_candidates:
            return "candidate_limit"
        return None

    @staticmethod
    def _finish(
        pool: dict[UUID, Episode],
        evidence: list[Episode],
        return_cap: int,
        stop_reason: str,
        metrics: dict[str, Any],
    ) -> tuple[list[Episode], dict[str, Any]]:
        selected = {episode.uid for episode in evidence}
        result = evidence
        if stop_reason != "sufficient":
            result = [
                *evidence,
                *(episode for uid, episode in pool.items() if uid not in selected),
            ][:return_cap]
        metrics.update(
            stop_reason=stop_reason,
            returned_sufficient=True if stop_reason == "sufficient" else None,
            unique_candidates=len(pool),
            returned_count=len(result),
            evidence_uids=[str(episode.uid) for episode in evidence],
        )
        return result, metrics

    async def do_query(
        self, policy: QueryPolicy, query: QueryParam
    ) -> tuple[list[Episode], dict[str, Any]]:
        """Search with request-local state and preserve the original query scope."""
        metrics = self._init_metrics()
        pool: dict[UUID, Episode] = {}
        evidence: list[Episode] = []
        return_cap = self._config.max_candidates
        if query.limit > 0:
            return_cap = min(query.limit, return_cap)
        if not query.query.strip():
            return self._finish(pool, evidence, return_cap, "empty_query", metrics)
        current = query.model_copy(
            update={
                "limit": min(self._config.initial_limit, self._config.max_candidates)
            }
        )
        stop_reason = "round_limit"
        for _ in range(min(self._config.max_rounds, policy.max_attempts)):
            episodes, round_metrics = await self._retrieve(policy, current, metrics)
            previous_count = len(pool)
            for episode in episodes:
                if len(pool) >= self._config.max_candidates:
                    break
                pool.setdefault(episode.uid, episode)
            round_metrics["admitted_candidates"] = len(pool) - previous_count
            if len(pool) == previous_count:
                stop_reason = "stalled"
                break
            decision = await self._check(query, pool, metrics, round_metrics)
            if decision is None:
                stop_reason = "invalid_decision"
                break
            candidates = list(pool.values())
            evidence = [candidates[index] for index in decision.evidence_indices]
            reason = self._stop_reason(decision, len(pool), return_cap)
            if reason is not None:
                stop_reason = reason
                break
            current = query.model_copy(
                update={
                    "query": decision.new_query.strip() or query.query,
                    "limit": min(current.limit * 2, self._config.max_candidates),
                }
            )
        return self._finish(pool, evidence, return_cap, stop_reason, metrics)
