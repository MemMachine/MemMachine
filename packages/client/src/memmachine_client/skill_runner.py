"""Client-side provider session loop for installed MemMachine skills."""

from __future__ import annotations

import inspect
import json
import re
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Literal

from .skill import Skill

if TYPE_CHECKING:
    from .memory import Memory

ProviderName = Literal["anthropic", "openai"]
SearchMode = Literal["direct", "rest"]

_UNCERTAINTY_PATTERN = re.compile(
    r"(?i)\b(if|likely|probably|suggests?|inferred?|assum(?:e|ed|ption)|"
    r"traditional|uncertain|unknown|not explicit|no explicit|may be|might)\b"
)
_STAGE_RESULT_LINE_PATTERN = re.compile(
    r"^\[StageResult(?:\s+\d+)?\]\s*Query:\s*(?P<query>.+?)\s*\|\s*"
    r"Answer:\s*(?P<answer>.+?)"
    r"(?:\s*\|\s*Confidence:\s*(?P<confidence>[01](?:\.\d+)?))?"
    r"(?:\s*\|\s*Reason:\s*(?P<reason>.+))?$"
)
_SUBQUERY_LINE_PATTERN = re.compile(r"^\[SubQuery(?:\s+\d+)?\]\s*(?P<query>.+)$")
_STAGE_RESULT_GUIDANCE = (
    "If you continue after a memory search, first emit one compact line exactly as "
    "[StageResult] Query: <resolved hop> | Answer: <best current candidate> | "
    "Confidence: <0.00-1.00> | Reason: <short basis>. "
    "If another search is needed, also emit one line exactly as "
    "[SubQuery] <next query>. Keep both lines short. "
    "When you are finished, give the final answer plainly with no stage tags."
)

_CANONICAL_MEMMACHINE_SEARCH_TOOL = {
    "type": "function",
    "name": "memmachine_search",
    "description": "Search MemMachine memory with the provided query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The memory search query to run.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


class SkillRunner:
    """Attach an installed skill to a caller-owned provider client session."""

    def __init__(
        self,
        skill: Skill,
        *,
        client: object,
        model: str,
        provider: ProviderName | None = None,
        search_mode: SearchMode = "rest",
        rest_memory: Memory | None = None,
        direct_memory: object | None = None,
        max_turns: int = 10,
        search_limit: int = 20,
        expand_context: int = 0,
        score_threshold: float | None = None,
        adaptive_search_limit: dict[str, int] | None = None,
        max_episode_chars: int | None = None,
        early_exit_confidence: bool | None = None,
        query_dedup: bool = False,
        stage_result_mode: bool = False,
        stage_result_confidence_threshold: float = 0.85,
        omit_episode_text_on_confident_stage_result: bool = False,
        use_answer_prompt_template: bool = False,
        anthropic_max_tokens: int = 2048,
        tool_choice: str | dict[str, str] = "auto",
    ) -> None:
        """Configure provider loop state and the search backend dependencies."""
        self.skill = skill
        self.provider = provider or skill.provider
        if self.provider != skill.provider:
            raise ValueError(
                "SkillRunner provider must match the provider used to install the skill"
            )
        self._client = client
        self._model = model
        self._search_mode = search_mode
        self._rest_memory = rest_memory
        self._direct_memory = direct_memory
        self.max_turns = max_turns
        self.search_limit = search_limit
        self.expand_context = expand_context
        self.score_threshold = score_threshold
        self.adaptive_search_limit = adaptive_search_limit
        self.max_episode_chars = max_episode_chars
        self.early_exit_confidence = early_exit_confidence
        self.query_dedup = query_dedup
        self.stage_result_mode = stage_result_mode
        self.stage_result_confidence_threshold = stage_result_confidence_threshold
        self.omit_episode_text_on_confident_stage_result = (
            omit_episode_text_on_confident_stage_result
        )
        self.use_answer_prompt_template = use_answer_prompt_template
        self._anthropic_max_tokens = anthropic_max_tokens
        self._tool_choice = tool_choice
        self._anthropic_messages: list[dict[str, object]] | None = None
        self._anthropic_system_prompt: str | None = None
        self.last_search_results: list[dict[str, object]] = []
        self.last_memory_search_called = 0
        self.last_memory_search_latency_seconds: list[float] = []
        self.last_stage_results: list[dict[str, object]] = []
        self.last_stage_sub_queries: list[str] = []
        self.last_llm_call_count = 0
        self.last_tool_call_count = 0
        self.last_follow_up_input_tokens = 0
        self.last_follow_up_output_tokens = 0
        self._seen_queries: set[frozenset[str]] = set()

        self._validate_client()

    def fork(self) -> SkillRunner:
        """Return a fresh runner with the same configuration and isolated state."""
        adaptive_search_limit = (
            dict(self.adaptive_search_limit)
            if self.adaptive_search_limit is not None
            else None
        )
        tool_choice = (
            dict(self._tool_choice)
            if isinstance(self._tool_choice, dict)
            else self._tool_choice
        )
        return SkillRunner(
            self.skill,
            client=self._client,
            model=self._model,
            provider=self.provider,
            search_mode=self._search_mode,
            rest_memory=self._rest_memory,
            direct_memory=self._direct_memory,
            max_turns=self.max_turns,
            search_limit=self.search_limit,
            expand_context=self.expand_context,
            score_threshold=self.score_threshold,
            adaptive_search_limit=adaptive_search_limit,
            max_episode_chars=self.max_episode_chars,
            early_exit_confidence=self.early_exit_confidence,
            query_dedup=self.query_dedup,
            stage_result_mode=self.stage_result_mode,
            stage_result_confidence_threshold=self.stage_result_confidence_threshold,
            omit_episode_text_on_confident_stage_result=(
                self.omit_episode_text_on_confident_stage_result
            ),
            use_answer_prompt_template=self.use_answer_prompt_template,
            anthropic_max_tokens=self._anthropic_max_tokens,
            tool_choice=tool_choice,
        )

    def _validate_client(self) -> None:
        if self.provider == "openai":
            create = getattr(getattr(self._client, "responses", None), "create", None)
            if not callable(create):
                raise TypeError(
                    "OpenAI SkillRunner requires a client with responses.create()."
                )
            return

        create = getattr(getattr(self._client, "messages", None), "create", None)
        if not callable(create):
            raise TypeError(
                "Anthropic SkillRunner requires a client with messages.create()."
            )

    def skill_messages(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, object]]:
        """Return provider-native attachment blocks for the next user prompt."""
        if self.provider == "openai":
            return self._openai_skill_messages(prompt)
        return self._anthropic_skill_messages(prompt, system_prompt=system_prompt)

    def _openai_skill_messages(self, prompt: str) -> list[dict[str, object]]:
        prompt_text = self._augment_prompt(prompt)
        return [
            *(
                {"type": "input_file", "file_id": file_id}
                for file_id in self.skill.file_ids
            ),
            {"type": "input_text", "text": prompt_text},
        ]

    def _anthropic_skill_messages(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, object]]:
        prompt_text = self._augment_prompt(prompt)
        content = [
            *(
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": file_id},
                }
                for file_id in self.skill.file_ids
            ),
            {"type": "text", "text": prompt_text},
        ]
        self._anthropic_messages = [{"role": "user", "content": content}]
        self._anthropic_system_prompt = system_prompt
        return content

    def tools(self) -> list[dict[str, object]]:
        """Return the provider-specific memmachine_search tool schema."""
        if self.provider == "openai":
            return [dict(_CANONICAL_MEMMACHINE_SEARCH_TOOL)]
        return [
            {
                "name": _CANONICAL_MEMMACHINE_SEARCH_TOOL["name"],
                "description": _CANONICAL_MEMMACHINE_SEARCH_TOOL["description"],
                "input_schema": dict(_CANONICAL_MEMMACHINE_SEARCH_TOOL["parameters"]),
            }
        ]

    async def handle_tool_loop(self, response: object) -> str:
        """Execute the memmachine_search loop until a final text response is produced."""
        self.last_search_results = []
        self.last_memory_search_called = 0
        self.last_memory_search_latency_seconds = []
        self.last_stage_results = []
        self.last_stage_sub_queries = []
        self.last_llm_call_count = 1
        self.last_tool_call_count = 0
        self.last_follow_up_input_tokens = 0
        self.last_follow_up_output_tokens = 0
        self._seen_queries = set()
        if self.provider == "openai":
            return await self._handle_openai_tool_loop(response)
        return await self._handle_anthropic_tool_loop(response)

    async def _handle_openai_tool_loop(self, response: object) -> str:
        latest_text = self._openai_response_output_text(response)
        self._record_stage_progress(latest_text)
        response_id = self._openai_response_id(response)

        for turn_index in range(self.max_turns):
            function_calls = [
                item
                for item in self._openai_response_items(response)
                if str(item.get("type", "")) == "function_call"
            ]
            self.last_tool_call_count += len(function_calls)
            if not function_calls:
                return self._final_response_text(latest_text)
            if turn_index + 1 >= self.max_turns:
                return self._final_response_text(latest_text) or self._partial_response_text()

            outputs: list[dict[str, object]] = []
            for call in function_calls:
                call_id = call.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    continue
                result = await self._run_search(
                    self._extract_query_from_arguments(call.get("arguments"))
                )
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, default=str),
                    }
                )

            if (
                self.early_exit_confidence
                and self.last_memory_search_called >= 1
                and latest_text
                and not _UNCERTAINTY_PATTERN.search(latest_text)
            ):
                return latest_text.strip()

            request: dict[str, object] = {
                "model": self._model,
                "input": outputs,
                "tools": self.tools(),
                "tool_choice": self._tool_choice,
            }
            if response_id is not None:
                request["previous_response_id"] = response_id
            response = await self._client.responses.create(**request)
            self.last_llm_call_count += 1
            self._record_follow_up_usage(response)
            latest_text = self._openai_response_output_text(response)
            self._record_stage_progress(latest_text)
            response_id = self._openai_response_id(response)

        return self._final_response_text(latest_text) or self._partial_response_text()

    async def _handle_anthropic_tool_loop(self, response: object) -> str:
        if self._anthropic_messages is None:
            raise RuntimeError("Call skill_messages() before Anthropic handle_tool_loop().")

        latest_text = ""
        for turn_index in range(self.max_turns):
            content_blocks = self._anthropic_content_blocks(response)
            latest_text = self._anthropic_extract_text(content_blocks)
            self._record_stage_progress(latest_text)
            tool_uses = [
                block for block in content_blocks if block.get("type") == "tool_use"
            ]
            self.last_tool_call_count += len(tool_uses)
            if not tool_uses:
                return self._final_response_text(latest_text)
            if turn_index + 1 >= self.max_turns:
                return self._final_response_text(latest_text) or self._partial_response_text()

            self._anthropic_messages.append({"role": "assistant", "content": content_blocks})
            tool_results: list[dict[str, object]] = []
            for tool_use in tool_uses:
                call_id = tool_use.get("id")
                if not isinstance(call_id, str) or not call_id:
                    continue
                result = await self._run_search(
                    self._extract_query_from_arguments(tool_use.get("input"))
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": json.dumps(result, default=str),
                    }
                )
            self._anthropic_messages.append({"role": "user", "content": tool_results})

            if (
                self.early_exit_confidence
                and self.last_memory_search_called >= 1
                and latest_text
                and not _UNCERTAINTY_PATTERN.search(latest_text)
            ):
                return latest_text.strip()

            request: dict[str, object] = {
                "model": self._model,
                "messages": self._anthropic_messages,
                "tools": self.tools(),
                "max_tokens": self._anthropic_max_tokens,
            }
            if self._anthropic_system_prompt is not None:
                request["system"] = self._anthropic_system_prompt
            anthropic_tool_choice = self._anthropic_tool_choice()
            if anthropic_tool_choice is not None:
                request["tool_choice"] = anthropic_tool_choice
            response = await self._client.messages.create(**request)
            self.last_llm_call_count += 1
            self._record_follow_up_usage(response)

        return self._final_response_text(latest_text) or self._partial_response_text()

    def _anthropic_tool_choice(self) -> dict[str, str] | None:
        if isinstance(self._tool_choice, dict):
            return {
                str(key): str(value)
                for key, value in self._tool_choice.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        if self._tool_choice == "auto":
            return {"type": "auto"}
        if self._tool_choice == "required":
            return {"type": "any"}
        return None

    async def _run_search(self, query: str) -> dict[str, object]:  # noqa: C901
        if self.query_dedup:
            normalized_query = frozenset(re.findall(r"[a-z0-9]+", query.lower()))
            if normalized_query in self._seen_queries:
                for cached in self.last_search_results:
                    if cached.get("query") == query:
                        return self._tool_search_result(cached)
                return self._tool_search_result(
                    {
                    "query": query,
                    "episode_summary": [],
                    "episodes": [],
                    "episodes_text": "",
                    "count": 0,
                    }
                )
            self._seen_queries.add(normalized_query)

        if self._search_mode == "rest":
            if self._rest_memory is None:
                raise RuntimeError("rest_memory is required when search_mode='rest'.")
            search_start = time.perf_counter()
            result = self._rest_memory.search(
                query,
                limit=self._current_search_limit(),
                expand_context=self.expand_context,
                score_threshold=self.score_threshold,
                agent_mode=False,
            )
            if inspect.isawaitable(result):
                result = await result
            elapsed_seconds = time.perf_counter() - search_start
            payload = self._normalize_search_result(query=query, raw_result=result)
            self.last_search_results.append(payload)
            self.last_memory_search_called += 1
            self.last_memory_search_latency_seconds.append(elapsed_seconds)
            return self._tool_search_result(payload)

        if self._direct_memory is None:
            raise RuntimeError("direct_memory is required when search_mode='direct'.")
        query_memory = getattr(self._direct_memory, "query_memory", None)
        if not callable(query_memory):
            raise TypeError("direct_memory must provide an async query_memory() method.")
        search_start = time.perf_counter()
        result = query_memory(
            query,
            limit=self._current_search_limit(),
            expand_context=self.expand_context,
            score_threshold=self.score_threshold if self.score_threshold is not None else -float("inf"),
        )
        if inspect.isawaitable(result):
            result = await result
        elapsed_seconds = time.perf_counter() - search_start
        payload = self._normalize_search_result(query=query, raw_result=result)
        self.last_search_results.append(payload)
        self.last_memory_search_called += 1
        self.last_memory_search_latency_seconds.append(elapsed_seconds)
        return self._tool_search_result(payload)

    def _normalize_search_result(
        self,
        *,
        query: str,
        raw_result: object,
    ) -> dict[str, object]:
        payload = self._as_dict(raw_result)
        episodic_payload = self._as_dict(payload.get("content")).get(
            "episodic_memory", payload
        )
        episodic = self._as_dict(episodic_payload)
        short_term = self._as_dict(episodic.get("short_term_memory"))
        long_term = self._as_dict(episodic.get("long_term_memory"))

        episode_summary = [
            item
            for item in short_term.get("episode_summary", [])
            if isinstance(item, str) and item.strip()
        ]
        episodes = [
            *self._normalize_episodes(short_term.get("episodes", [])),
            *self._normalize_episodes(long_term.get("episodes", [])),
        ]
        episode_lines = [
            f"{index}. {self._episode_content(episode['content'])}"
            for index, episode in enumerate(episodes, start=1)
            if isinstance(episode.get("content"), str) and episode["content"].strip()
        ]
        stage_result_memory = self._stage_result_memory_lines()

        payload = {
            "query": query,
            "episode_summary": episode_summary,
            "episodes": episodes,
            "episodes_text": "\n".join([*episode_summary, *episode_lines]).strip(),
            "count": len(episodes),
        }
        if stage_result_memory:
            payload["stage_result_memory"] = stage_result_memory
            payload["stage_result_instructions"] = _STAGE_RESULT_GUIDANCE
        return payload

    def _tool_search_result(self, payload: dict[str, object]) -> dict[str, object]:
        compact_payload: dict[str, object] = {
            "query": payload.get("query", ""),
            "count": payload.get("count", 0),
        }
        stage_result_memory = payload.get("stage_result_memory")
        if isinstance(stage_result_memory, list) and stage_result_memory:
            compact_payload["stage_result_memory"] = stage_result_memory
        if not self._should_omit_tool_episode_text():
            compact_payload["episodes_text"] = payload.get("episodes_text", "")
        return compact_payload

    def _should_omit_tool_episode_text(self) -> bool:
        if not self.omit_episode_text_on_confident_stage_result:
            return False
        if not self.stage_result_mode:
            return False
        if not self.last_stage_results:
            return False
        latest_stage_result = self.last_stage_results[-1]
        confidence = latest_stage_result.get("confidence_score")
        if not isinstance(confidence, int | float) or isinstance(confidence, bool):
            return False
        return float(confidence) >= self.stage_result_confidence_threshold

    def _normalize_episodes(self, raw_episodes: object) -> list[dict[str, object]]:
        if not isinstance(raw_episodes, list):
            return []
        normalized: list[dict[str, object]] = []
        for raw_episode in raw_episodes:
            episode = self._as_dict(raw_episode)
            if not episode:
                continue
            if self.score_threshold is not None:
                score = float(episode.get("score", episode.get("relevance_score", 1.0)))
                if score < self.score_threshold:
                    continue
            normalized.append(episode)
        return normalized

    def _current_search_limit(self) -> int:
        if self.adaptive_search_limit is None:
            return self.search_limit
        if self.last_memory_search_called == 0:
            return self.adaptive_search_limit["initial"]
        return self.adaptive_search_limit["escalated"]

    def _episode_content(self, content: str) -> str:
        if self.max_episode_chars is None:
            return content
        return content[: self.max_episode_chars]

    def _augment_prompt(self, prompt: str) -> str:
        if not self.stage_result_mode:
            return prompt
        return f"{prompt}\n\n{_STAGE_RESULT_GUIDANCE}"

    def _record_follow_up_usage(self, response: object) -> None:
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        if usage is None:
            return
        usage_payload = self._as_dict(usage)
        self.last_follow_up_input_tokens += int(
            usage_payload.get("input_tokens", getattr(usage, "input_tokens", 0)) or 0
        )
        self.last_follow_up_output_tokens += int(
            usage_payload.get("output_tokens", getattr(usage, "output_tokens", 0)) or 0
        )

    def _record_stage_progress(self, latest_text: str) -> None:  # noqa: C901
        if not self.stage_result_mode or not latest_text.strip():
            return
        for raw_line in latest_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            stage_match = _STAGE_RESULT_LINE_PATTERN.match(line)
            if stage_match:
                query = stage_match.group("query").strip()
                answer = stage_match.group("answer").strip()
                if not query or not answer:
                    continue
                record: dict[str, object] = {
                    "query": query,
                    "stage_result": answer,
                }
                confidence_raw = stage_match.group("confidence")
                if confidence_raw is not None:
                    with suppress(ValueError):
                        record["confidence_score"] = float(confidence_raw)
                reason = stage_match.group("reason")
                if reason is not None and reason.strip():
                    record["reason_note"] = reason.strip()
                key = (query, answer)
                existing = {
                    (
                        str(item.get("query", "")),
                        str(item.get("stage_result", "")),
                    )
                    for item in self.last_stage_results
                }
                if key not in existing:
                    self.last_stage_results.append(record)
                continue

            subquery_match = _SUBQUERY_LINE_PATTERN.match(line)
            if subquery_match:
                subquery = subquery_match.group("query").strip()
                if subquery and subquery not in self.last_stage_sub_queries:
                    self.last_stage_sub_queries.append(subquery)

    def _stage_result_memory_lines(self) -> list[str]:
        if not self.stage_result_mode:
            return []
        lines: list[str] = []
        for index, item in enumerate(self.last_stage_results, start=1):
            query = str(item.get("query") or "").strip()
            answer = str(item.get("stage_result") or "").strip()
            if not query or not answer:
                continue
            line = f"[StageResult {index}] Query: {query} | Answer: {answer}"
            confidence = item.get("confidence_score")
            if isinstance(confidence, int | float) and not isinstance(confidence, bool):
                line += f" | Confidence: {float(confidence):.2f}"
            reason = item.get("reason_note")
            if isinstance(reason, str) and reason.strip():
                line += f" | Reason: {reason.strip()}"
            lines.append(line)
        for index, subquery in enumerate(self.last_stage_sub_queries, start=1):
            lines.append(f"[SubQuery {index}] {subquery}")
        return lines

    def _final_response_text(self, latest_text: str) -> str:
        stripped = latest_text.strip()
        if not self.stage_result_mode or not stripped:
            return stripped
        clean_lines = [
            line
            for line in stripped.splitlines()
            if not _STAGE_RESULT_LINE_PATTERN.match(line.strip())
            and not _SUBQUERY_LINE_PATTERN.match(line.strip())
        ]
        cleaned = "\n".join(line for line in clean_lines if line.strip()).strip()
        return cleaned or stripped

    @staticmethod
    def _as_dict(raw_value: object) -> dict[str, object]:
        if isinstance(raw_value, dict):
            return {str(key): value for key, value in raw_value.items()}
        model_dump = getattr(raw_value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json")
            if isinstance(dumped, dict):
                return {str(key): value for key, value in dumped.items()}
        raw_dict = getattr(raw_value, "__dict__", None)
        if isinstance(raw_dict, dict):
            return {str(key): value for key, value in raw_dict.items()}
        return {}

    @staticmethod
    def _extract_query_from_arguments(raw_arguments: object) -> str:
        if isinstance(raw_arguments, dict):
            query = raw_arguments.get("query")
            return query.strip() if isinstance(query, str) else ""
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return ""
            if isinstance(parsed, dict):
                query = parsed.get("query")
                return query.strip() if isinstance(query, str) else ""
        return ""

    @staticmethod
    def _partial_response_text() -> str:
        return "Partial result: memmachine_search loop reached max_turns."

    @staticmethod
    def _openai_response_id(response: object) -> str | None:
        if isinstance(response, dict):
            value = response.get("id")
            return value if isinstance(value, str) else None
        value = getattr(response, "id", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _openai_response_output_text(response: object) -> str:
        if isinstance(response, dict):
            value = response.get("output_text")
            return value if isinstance(value, str) else ""
        value = getattr(response, "output_text", "")
        return value if isinstance(value, str) else ""

    @classmethod
    def _openai_response_items(cls, response: object) -> list[dict[str, object]]:
        if isinstance(response, dict):
            items = response.get("output", [])
            return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

        items = getattr(response, "output", []) or []
        normalized: list[dict[str, object]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(item)
                continue
            if getattr(item, "type", None) == "function_call":
                normalized.append(
                    {
                        "type": "function_call",
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", {}),
                        "call_id": getattr(item, "call_id", None),
                    }
                )
        return normalized

    @classmethod
    def _anthropic_content_blocks(cls, response: object) -> list[dict[str, object]]:
        if isinstance(response, dict):
            blocks = response.get("content", [])
            raw_blocks = blocks if isinstance(blocks, list) else []
        else:
            raw_blocks = getattr(response, "content", []) or []

        normalized: list[dict[str, object]] = []
        for raw_block in raw_blocks:
            block = cls._normalize_anthropic_block(raw_block)
            if block is not None:
                normalized.append(block)
        return normalized

    @staticmethod
    def _normalize_anthropic_block(raw_block: object) -> dict[str, object] | None:
        if isinstance(raw_block, dict):
            return SkillRunner._normalize_anthropic_block_dict(raw_block)

        block_type = getattr(raw_block, "type", None)
        if block_type == "text":
            text = getattr(raw_block, "text", None)
            if isinstance(text, str):
                return {"type": "text", "text": text}
            return None
        if block_type == "tool_use":
            name = getattr(raw_block, "name", None)
            if not isinstance(name, str):
                return None
            block = {
                "type": "tool_use",
                "name": name,
                "input": getattr(raw_block, "input", {}),
            }
            raw_id = getattr(raw_block, "id", None)
            if isinstance(raw_id, str):
                block["id"] = raw_id
            return block
        return None

    @staticmethod
    def _normalize_anthropic_block_dict(
        raw_block: dict[str, object],
    ) -> dict[str, object] | None:
        block_type = raw_block.get("type")
        if block_type == "text":
            text = raw_block.get("text")
            if isinstance(text, str):
                return {"type": "text", "text": text}
            return None
        if block_type == "tool_use":
            name = raw_block.get("name")
            if not isinstance(name, str):
                return None
            block: dict[str, object] = {
                "type": "tool_use",
                "name": name,
                "input": raw_block.get("input", {}),
            }
            raw_id = raw_block.get("id")
            if isinstance(raw_id, str):
                block["id"] = raw_id
            return block
        return None

    @staticmethod
    def _anthropic_extract_text(content_blocks: list[dict[str, object]]) -> str:
        return "\n".join(
            block["text"]
            for block in content_blocks
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        ).strip()
