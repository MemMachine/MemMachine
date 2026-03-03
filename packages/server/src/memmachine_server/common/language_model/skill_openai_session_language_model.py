"""OpenAI live-session runtime for skill-style function-calling orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import uuid4

import json_repair
import openai
from pydantic import BaseModel, ConfigDict, Field, InstanceOf

ToolHandler = Callable[[dict[str, object]], object | Awaitable[object]]

logger = logging.getLogger(__name__)


class SkillLanguageModelError(RuntimeError):
    """Base error for skill session runtime failures."""


class SkillToolCallFormatError(SkillLanguageModelError):
    """Raised when a tool call payload cannot be parsed safely."""


class SkillToolNotFoundError(SkillLanguageModelError):
    """Raised when model requests a tool with no registered handler."""


class SkillSessionLimitError(SkillLanguageModelError):
    """Raised when max-turn or timeout guardrails are exceeded."""


class SkillToolExecution(BaseModel):
    """Tool execution transcript item."""

    model_config = ConfigDict(extra="forbid")

    call_id: str | None = None
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    output: object = None


class SkillRunResult(BaseModel):
    """Final result of a skill live-session run."""

    model_config = ConfigDict(extra="forbid")

    final_response: str
    raw_model_output: str = ""
    tool_executions: list[SkillToolExecution] = Field(default_factory=list)
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_time_seconds: float = 0.0
    turn_count: int = 0


class SkillOpenAISessionLanguageModelParams(BaseModel):
    """Configuration for OpenAI skill live-session runtime."""

    client: InstanceOf[openai.AsyncOpenAI]
    model: str = Field(min_length=1)
    max_retry_interval_seconds: int = Field(default=120, gt=0)
    reasoning_effort: str | None = None


class SkillSessionModelProtocol(Protocol):
    """Interface consumed by retrieve-skill and sub-skill runtimes."""

    async def run_live_session(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, object]],
        tool_registry: dict[str, ToolHandler],
        tool_choice: str | dict[str, str] = "auto",
        max_turns: int = 16,
        timeout_seconds: float | None = None,
    ) -> SkillRunResult: ...


class SkillLanguageModel:
    """OpenAI Responses function-calling live session runner."""

    def __init__(self, params: SkillOpenAISessionLanguageModelParams) -> None:
        """Initialize runtime with OpenAI Responses client settings."""
        self._client = params.client
        self._model = params.model
        self._max_retry_interval_seconds = params.max_retry_interval_seconds
        self._reasoning_effort = params.reasoning_effort

    @classmethod
    def from_openai_responses_language_model(
        cls,
        model: object,
    ) -> SkillLanguageModel:
        """Build from existing OpenAIResponsesLanguageModel instance."""
        from .openai_responses_language_model import OpenAIResponsesLanguageModel

        if not isinstance(model, OpenAIResponsesLanguageModel):
            raise TypeError(
                "SkillLanguageModel requires OpenAIResponsesLanguageModel "
                "for OpenAI live function-calling sessions."
            )
        return cls(
            SkillOpenAISessionLanguageModelParams(
                client=model.client,
                model=model.model_name,
                max_retry_interval_seconds=model.max_retry_interval_seconds,
                reasoning_effort=model.reasoning_effort,
            )
        )

    async def run_live_session(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, object]],
        tool_registry: dict[str, ToolHandler],
        tool_choice: str | dict[str, str] = "auto",
        max_turns: int = 16,
        timeout_seconds: float | None = None,
    ) -> SkillRunResult:
        """Run one live model session until no more function calls are emitted."""
        if max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")

        started = time.monotonic()
        turn_count = 0
        tool_executions: list[SkillToolExecution] = []
        input_tokens_total = 0
        output_tokens_total = 0
        llm_time_total = 0.0
        raw_model_output = ""
        last_response_id: str | None = None
        current_input: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        while True:
            if turn_count >= max_turns:
                raise SkillSessionLimitError("Skill session exceeded max_turns.")
            if (
                timeout_seconds is not None
                and (time.monotonic() - started) > timeout_seconds
            ):
                raise SkillSessionLimitError("Skill session exceeded timeout.")

            request: dict[str, object] = {
                "model": self._model,
                "input": current_input,
                "tools": tools,
                "tool_choice": tool_choice,
            }
            if self._reasoning_effort is not None:
                request["reasoning"] = {"effort": self._reasoning_effort}
            if last_response_id is not None:
                request["previous_response_id"] = last_response_id

            llm_call_started = time.monotonic()
            response = await self._call_responses_create_with_retry(
                max_attempts=3, **request
            )
            llm_time_total += time.monotonic() - llm_call_started
            turn_count += 1

            last_response_id = self._response_id(response)
            usage = self._response_usage(response)
            input_tokens_total += usage["input_tokens"]
            output_tokens_total += usage["output_tokens"]

            raw_model_output = self._response_output_text(response)
            response_items = self._response_items(response)
            function_calls = [
                item
                for item in response_items
                if str(item.get("type", "")) == "function_call"
            ]
            if not function_calls:
                return SkillRunResult(
                    final_response=raw_model_output.strip(),
                    raw_model_output=raw_model_output,
                    tool_executions=tool_executions,
                    llm_input_tokens=input_tokens_total,
                    llm_output_tokens=output_tokens_total,
                    llm_time_seconds=llm_time_total,
                    turn_count=turn_count,
                )

            executions = await self._execute_function_calls(
                function_calls=function_calls,
                tool_registry=tool_registry,
            )
            tool_executions.extend(executions)

            function_call_outputs: list[dict[str, object]] = []
            for item in executions:
                if item.call_id is None:
                    raise SkillToolCallFormatError(
                        "function_call item missing call_id for output callback."
                    )
                function_call_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": self._serialize_tool_output(item.output),
                    }
                )
            current_input.extend(function_call_outputs)

    async def _execute_function_calls(
        self,
        *,
        function_calls: object,
        tool_registry: dict[str, ToolHandler],
    ) -> list[SkillToolExecution]:
        if function_calls is None:
            return []
        if not isinstance(function_calls, list):
            raise SkillToolCallFormatError("function_calls must be a list")

        results: list[SkillToolExecution] = []
        for raw_call in function_calls:
            call_id, name, arguments = self._parse_tool_call(raw_call)
            handler = tool_registry.get(name)
            if handler is None:
                raise SkillToolNotFoundError(f"No handler registered for tool: {name}")
            output = handler(arguments)
            if asyncio.iscoroutine(output):
                output = await output
            results.append(
                SkillToolExecution(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    output=output,
                )
            )
        return results

    def _parse_tool_call(
        self,
        raw_call: object,
    ) -> tuple[str | None, str, dict[str, object]]:
        if not isinstance(raw_call, dict):
            raise SkillToolCallFormatError("Tool call entry must be an object.")
        if raw_call.get("type") != "function_call":
            raise SkillToolCallFormatError(
                "Tool call entry must have type=function_call."
            )

        name = raw_call.get("name")
        arguments = raw_call.get("arguments", {})
        call_id = raw_call.get("call_id")

        if not isinstance(name, str) or not name.strip():
            raise SkillToolCallFormatError("Tool call missing function name.")
        parsed_args = self._parse_arguments(arguments)
        return (
            str(call_id) if isinstance(call_id, str) else None,
            name.strip(),
            parsed_args,
        )

    def _parse_arguments(self, raw_arguments: object) -> dict[str, object]:
        if isinstance(raw_arguments, dict):
            return {str(key): value for key, value in raw_arguments.items()}
        if isinstance(raw_arguments, str):
            try:
                parsed = json_repair.loads(raw_arguments)
            except Exception as err:
                raise SkillToolCallFormatError(
                    "Failed to parse function call arguments."
                ) from err
            if not isinstance(parsed, dict):
                raise SkillToolCallFormatError(
                    "Function arguments must decode to object."
                )
            return {str(key): value for key, value in parsed.items()}
        raise SkillToolCallFormatError(
            "Function call arguments must be object or JSON string."
        )

    async def _call_responses_create_with_retry(
        self,
        *,
        max_attempts: int,
        **kwargs: object,
    ) -> object:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        call_uuid = uuid4()
        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._client.responses.create(**kwargs)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as err:
                if attempt >= max_attempts:
                    raise SkillLanguageModelError(
                        f"[call uuid: {call_uuid}] OpenAI responses.create failed "
                        f"after {attempt} attempts due to retryable {type(err).__name__}."
                    ) from err
                logger.info(
                    "[call uuid: %s] Retrying responses.create in %d second(s) "
                    "after attempt %d (%s).",
                    call_uuid,
                    sleep_seconds,
                    attempt,
                    type(err).__name__,
                )
                await asyncio.sleep(sleep_seconds)
                sleep_seconds = min(
                    sleep_seconds * 2,
                    self._max_retry_interval_seconds,
                )
            except openai.OpenAIError as err:
                raise SkillLanguageModelError(
                    f"[call uuid: {call_uuid}] OpenAI responses.create failed "
                    f"with non-retryable {type(err).__name__}."
                ) from err

        raise SkillLanguageModelError(
            "responses.create retry loop exited unexpectedly."
        )

    def _response_id(self, response: object) -> str | None:
        if isinstance(response, dict):
            value = response.get("id")
            return str(value) if value is not None else None
        value = getattr(response, "id", None)
        return str(value) if value is not None else None

    def _response_output_text(self, response: object) -> str:
        if isinstance(response, dict):
            value = response.get("output_text")
            return value if isinstance(value, str) else ""
        value = getattr(response, "output_text", "")
        return value if isinstance(value, str) else ""

    def _response_usage(self, response: object) -> dict[str, int]:
        if isinstance(response, dict):
            usage = response.get("usage", {}) or {}
            return {
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
            }
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0}
        return {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }

    def _response_items(self, response: object) -> list[dict[str, object]]:  # noqa: C901
        if isinstance(response, dict):
            items = response.get("output", [])
            if not isinstance(items, list):
                return []
            return [item for item in items if isinstance(item, dict)]

        items = getattr(response, "output", []) or []
        normalized: list[dict[str, object]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(item)
                continue

            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                normalized.append(
                    {
                        "type": "function_call",
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", {}),
                        "call_id": getattr(item, "call_id", None),
                    }
                )
                continue
            if item_type == "reasoning":
                dump_method = getattr(item, "model_dump", None)
                if callable(dump_method):
                    dumped = dump_method()
                    if isinstance(dumped, dict):
                        normalized.append(dumped)
                        continue
                item_dict: dict[str, object] = {"type": "reasoning"}
                item_id = getattr(item, "id", None)
                if item_id is not None:
                    item_dict["id"] = item_id
                summary = getattr(item, "summary", None)
                if summary is not None:
                    item_dict["summary"] = summary
                normalized.append(item_dict)

        return normalized

    def _serialize_tool_output(self, output: object) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, default=str)
