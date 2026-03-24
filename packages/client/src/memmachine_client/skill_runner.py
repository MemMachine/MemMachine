"""Client-side provider session loop for installed MemMachine skills."""
# ruff: noqa: SLF001

from __future__ import annotations

import inspect
import json
import re
import time
from typing import TYPE_CHECKING, Literal

from memmachine_common.skill_loop import (
    SkillLoopState,
    SkillLoopToolCall,
    SkillLoopToolResult,
    as_dict,
    augment_prompt,
    build_tool_search_result,
    continue_tool_loop,
    extract_query_from_arguments,
    final_response_text,
    normalize_search_result,
    normalize_tool_arguments,
)

from .skill import Skill

if TYPE_CHECKING:
    from .memory import Memory

ProviderName = Literal["anthropic", "openai"]
SearchMode = Literal["direct", "rest"]

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


class _OpenAIToolLoopTransport:
    """OpenAI Responses transport wrapper for the shared tool loop."""

    def __init__(self, runner: SkillRunner) -> None:
        self._runner = runner

    def response_text(self, response: object) -> str:
        return self._runner._openai_response_output_text(response)

    def tool_calls(self, response: object) -> list[SkillLoopToolCall]:
        calls: list[SkillLoopToolCall] = []
        for item in self._runner._openai_response_items(response):
            if str(item.get("type", "")) != "function_call":
                continue
            arguments = normalize_tool_arguments(item.get("arguments", {}))
            calls.append(
                SkillLoopToolCall(
                    name=str(item.get("name", "")),
                    query=extract_query_from_arguments(arguments),
                    call_id=item.get("call_id")
                    if isinstance(item.get("call_id"), str)
                    else None,
                    arguments=arguments,
                )
            )
        return calls

    async def continue_with_results(
        self,
        *,
        response: object,
        tool_results: list[SkillLoopToolResult],
    ) -> object:
        outputs: list[dict[str, object]] = []
        for result in tool_results:
            if not isinstance(result.call_id, str) or not result.call_id:
                continue
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": json.dumps(result.output, default=str),
                }
            )
        request: dict[str, object] = {
            "model": self._runner._model,
            "input": outputs,
            "tools": self._runner.tools(),
            "tool_choice": self._runner._tool_choice,
        }
        response_id = self._runner._openai_response_id(response)
        if response_id is not None:
            request["previous_response_id"] = response_id
        return await self._runner._client.responses.create(**request)

    def usage(self, response: object) -> tuple[int, int]:
        return self._runner._usage_tuple(response)


class _AnthropicToolLoopTransport:
    """Anthropic Messages transport wrapper for the shared tool loop."""

    def __init__(self, runner: SkillRunner) -> None:
        self._runner = runner

    def response_text(self, response: object) -> str:
        return self._runner._anthropic_extract_text(
            self._runner._anthropic_content_blocks(response)
        )

    def tool_calls(self, response: object) -> list[SkillLoopToolCall]:
        calls: list[SkillLoopToolCall] = []
        for block in self._runner._anthropic_content_blocks(response):
            if block.get("type") != "tool_use":
                continue
            arguments = normalize_tool_arguments(block.get("input"))
            calls.append(
                SkillLoopToolCall(
                    name=str(block.get("name", "")),
                    query=extract_query_from_arguments(arguments),
                    call_id=block.get("id") if isinstance(block.get("id"), str) else None,
                    arguments=arguments,
                )
            )
        return calls

    async def continue_with_results(
        self,
        *,
        response: object,
        tool_results: list[SkillLoopToolResult],
    ) -> object:
        if self._runner._anthropic_messages is None:
            raise RuntimeError(
                "Call skill_messages() before Anthropic handle_tool_loop()."
            )

        content_blocks = self._runner._anthropic_content_blocks(response)
        self._runner._anthropic_messages.append(
            {"role": "assistant", "content": content_blocks}
        )
        tool_outputs: list[dict[str, object]] = []
        for result in tool_results:
            if not isinstance(result.call_id, str) or not result.call_id:
                continue
            tool_outputs.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": json.dumps(result.output, default=str),
                }
            )
        self._runner._anthropic_messages.append(
            {"role": "user", "content": tool_outputs}
        )

        request: dict[str, object] = {
            "model": self._runner._model,
            "messages": self._runner._anthropic_messages,
            "tools": self._runner.tools(),
            "max_tokens": self._runner._anthropic_max_tokens,
        }
        if self._runner._anthropic_system_prompt is not None:
            request["system"] = self._runner._anthropic_system_prompt
        anthropic_tool_choice = self._runner._anthropic_tool_choice()
        if anthropic_tool_choice is not None:
            request["tool_choice"] = anthropic_tool_choice
        return await self._runner._client.messages.create(**request)

    def usage(self, response: object) -> tuple[int, int]:
        return self._runner._usage_tuple(response)


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
        direct_search_extra_kwargs: dict[str, object] | None = None,
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
        self._direct_search_extra_kwargs = (
            dict(direct_search_extra_kwargs)
            if direct_search_extra_kwargs is not None
            else {}
        )
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
        self.last_raw_search_results: list[object] = []
        self.last_memory_search_called = 0
        self.last_memory_search_latency_seconds: list[float] = []
        self.last_stage_results: list[dict[str, object]] = []
        self.last_stage_sub_queries: list[str] = []
        self.last_initial_input_tokens = 0
        self.last_initial_output_tokens = 0
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
            direct_search_extra_kwargs=dict(self._direct_search_extra_kwargs),
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
        self.last_raw_search_results = []
        self.last_memory_search_called = 0
        self.last_memory_search_latency_seconds = []
        self.last_stage_results = []
        self.last_stage_sub_queries = []
        self.last_llm_call_count = 1
        self.last_tool_call_count = 0
        self.last_follow_up_input_tokens = 0
        self.last_follow_up_output_tokens = 0
        self._seen_queries = set()
        state = SkillLoopState()
        transport = (
            _OpenAIToolLoopTransport(self)
            if self.provider == "openai"
            else _AnthropicToolLoopTransport(self)
        )
        result = await continue_tool_loop(
            initial_response=response,
            transport=transport,
            state=state,
            run_search=self._run_search,
            max_turns=self.max_turns,
            stage_result_mode=self.stage_result_mode,
            early_exit_confidence=bool(self.early_exit_confidence),
            partial_response_text=self._partial_response_text(),
        )
        self.last_stage_results = list(state.stage_results)
        self.last_stage_sub_queries = list(state.stage_sub_queries)
        self.last_llm_call_count = state.llm_call_count
        self.last_tool_call_count = state.tool_call_count
        self.last_follow_up_input_tokens = state.follow_up_input_tokens
        self.last_follow_up_output_tokens = state.follow_up_output_tokens
        self.last_memory_search_called = state.memory_search_called
        return result

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

    async def _run_search(  # noqa: C901
        self,
        query: str,
        state: SkillLoopState | None = None,
    ) -> dict[str, object]:
        sync_back = state is None
        if state is None:
            state = SkillLoopState(
                memory_search_called=self.last_memory_search_called,
                stage_results=list(self.last_stage_results),
                stage_sub_queries=list(self.last_stage_sub_queries),
            )
        if self.query_dedup:
            normalized_query = frozenset(re.findall(r"[a-z0-9]+", query.lower()))
            if normalized_query in self._seen_queries:
                for cached in self.last_search_results:
                    if cached.get("query") == query:
                        return self._tool_search_result(
                            cached,
                            stage_results=state.stage_results,
                        )
                return self._tool_search_result(
                    {
                        "query": query,
                        "episode_summary": [],
                        "episodes": [],
                        "episodes_text": "",
                        "count": 0,
                    },
                    stage_results=state.stage_results,
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
        else:
            if self._direct_memory is None:
                raise RuntimeError(
                    "direct_memory is required when search_mode='direct'."
                )
            query_memory = getattr(self._direct_memory, "query_memory", None)
            if not callable(query_memory):
                raise TypeError(
                    "direct_memory must provide an async query_memory() method."
                )
            search_start = time.perf_counter()
            result = query_memory(
                query=query,
                limit=self._current_search_limit(),
                expand_context=self.expand_context,
                score_threshold=(
                    self.score_threshold
                    if self.score_threshold is not None
                    else -float("inf")
                ),
                **self._direct_search_extra_kwargs,
            )
        if inspect.isawaitable(result):
            result = await result
        elapsed_seconds = time.perf_counter() - search_start
        self.last_raw_search_results.append(result)
        payload = self._normalize_search_result(
            query=query,
            raw_result=result,
            state=state,
        )
        self.last_search_results.append(payload)
        state.memory_search_called += 1
        self.last_memory_search_latency_seconds.append(elapsed_seconds)
        tool_result = self._tool_search_result(payload, stage_results=state.stage_results)
        if sync_back:
            self.last_memory_search_called = state.memory_search_called
            self.last_stage_results = list(state.stage_results)
            self.last_stage_sub_queries = list(state.stage_sub_queries)
        return tool_result

    async def search(self, query: str) -> dict[str, object]:
        """Run one search outside the provider loop and return the tool payload."""
        return await self._run_search(query)

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
    ) -> str:
        """Run the full installed-skill session, including the initial provider call."""
        self.last_initial_input_tokens = 0
        self.last_initial_output_tokens = 0
        if self.provider == "openai":
            request: dict[str, object] = {
                "model": self._model,
                "input": [{"role": "user", "content": self.skill_messages(prompt)}],
                "tools": self.tools(),
                "tool_choice": self._tool_choice,
            }
            if max_output_tokens is not None:
                request["max_output_tokens"] = max_output_tokens
            if top_p is not None:
                request["top_p"] = top_p
            response = await self._client.responses.create(**request)
        else:
            _ = self.skill_messages(prompt, system_prompt=system_prompt)
            if self._anthropic_messages is None:
                raise RuntimeError("Anthropic skill_messages() did not initialize messages.")
            request = {
                "model": self._model,
                "messages": self._anthropic_messages,
                "tools": self.tools(),
                "max_tokens": (
                    max_output_tokens
                    if max_output_tokens is not None
                    else self._anthropic_max_tokens
                ),
            }
            if system_prompt is not None:
                request["system"] = system_prompt
            anthropic_tool_choice = self._anthropic_tool_choice()
            if anthropic_tool_choice is not None:
                request["tool_choice"] = anthropic_tool_choice
            response = await self._client.messages.create(**request)
        self.last_initial_input_tokens, self.last_initial_output_tokens = (
            self._usage_tuple(response)
        )
        return await self.handle_tool_loop(response)

    def _normalize_search_result(
        self,
        *,
        query: str,
        raw_result: object,
        state: SkillLoopState | None = None,
    ) -> dict[str, object]:
        if state is None:
            state = SkillLoopState(
                stage_results=list(self.last_stage_results),
                stage_sub_queries=list(self.last_stage_sub_queries),
            )
        return normalize_search_result(
            query=query,
            raw_result=raw_result,
            score_threshold=self.score_threshold,
            max_episode_chars=self.max_episode_chars,
            stage_result_mode=self.stage_result_mode,
            stage_results=state.stage_results,
            stage_sub_queries=state.stage_sub_queries,
        )

    def _tool_search_result(
        self,
        payload: dict[str, object],
        *,
        stage_results: list[dict[str, object]],
    ) -> dict[str, object]:
        return build_tool_search_result(
            payload=payload,
            stage_result_mode=self.stage_result_mode,
            stage_results=stage_results,
            omit_episode_text_on_confident_stage_result=(
                self.omit_episode_text_on_confident_stage_result
            ),
            stage_result_confidence_threshold=self.stage_result_confidence_threshold,
        )

    def _current_search_limit(self) -> int:
        if self.adaptive_search_limit is None:
            return self.search_limit
        if self.last_memory_search_called == 0 and not self.last_search_results:
            return self.adaptive_search_limit["initial"]
        return self.adaptive_search_limit["escalated"]

    def _augment_prompt(self, prompt: str) -> str:
        return augment_prompt(prompt, stage_result_mode=self.stage_result_mode)

    def _final_response_text(self, latest_text: str) -> str:
        return final_response_text(
            latest_text,
            stage_result_mode=self.stage_result_mode,
        )

    @staticmethod
    def _as_dict(raw_value: object) -> dict[str, object]:
        return as_dict(raw_value)

    @staticmethod
    def _extract_query_from_arguments(raw_arguments: object) -> str:
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return ""
        return extract_query_from_arguments(raw_arguments)

    @staticmethod
    def _partial_response_text() -> str:
        return "Partial result: memmachine_search loop reached max_turns."

    def _usage_tuple(self, response: object) -> tuple[int, int]:
        usage = response.get("usage") if isinstance(response, dict) else getattr(
            response, "usage", None
        )
        if usage is None:
            return 0, 0
        usage_payload = self._as_dict(usage)
        return (
            int(
                usage_payload.get(
                    "input_tokens",
                    getattr(usage, "input_tokens", 0),
                )
                or 0
            ),
            int(
                usage_payload.get(
                    "output_tokens",
                    getattr(usage, "output_tokens", 0),
                )
                or 0
            ),
        )

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
