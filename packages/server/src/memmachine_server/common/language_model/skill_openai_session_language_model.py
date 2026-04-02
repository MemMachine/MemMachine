"""OpenAI live-session runtime for skill-style function-calling orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast
from uuid import uuid4

import httpx
import json_repair
import openai
from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from .provider_skill_bundle import ProviderSkillBundle

ToolHandler = Callable[[dict[str, object]], object | Awaitable[object]]

logger = logging.getLogger(__name__)


class SkillLanguageModelError(RuntimeError):
    """Base error for skill session runtime failures."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        """Store structured diagnostics for fallback metrics/debugging."""
        self.diagnostics = diagnostics or {}
        if self.diagnostics:
            serialized = SkillLanguageModel.serialize_object_for_diagnostics(
                self.diagnostics
            )
            message = f"{message} diagnostics={serialized}"
        super().__init__(message)


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
    normalization_warnings: list[str] = Field(default_factory=list)


class SkillOpenAISessionLanguageModelParams(BaseModel):
    """Configuration for OpenAI skill live-session runtime."""

    client: InstanceOf[openai.AsyncOpenAI]
    model: str = Field(min_length=1)
    max_retry_interval_seconds: int = Field(default=120, gt=0)
    reasoning_effort: str | None = None
    log_raw_output: bool = False
    native_skill_environment: str = Field(default="local")


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
        provider_skill_bundles: list[ProviderSkillBundle] | None = None,
    ) -> SkillRunResult: ...


class SkillLanguageModel:
    """OpenAI Responses function-calling live session runner."""

    _LOCAL_SHELL_ALLOWED_BASE_COMMANDS: ClassVar[set[str]] = {
        "cat",
        "echo",
        "find",
        "head",
        "ls",
        "pwd",
        "rg",
        "sed",
        "tail",
        "wc",
    }
    _LOCAL_SHELL_DISALLOWED_TOKENS: ClassVar[tuple[str, ...]] = (
        ";",
        "&&",
        "||",
        "|",
        ">",
        "<",
        "$(",
        "`",
    )

    def __init__(self, params: SkillOpenAISessionLanguageModelParams) -> None:
        """Initialize runtime with OpenAI Responses client settings."""
        self._client = params.client
        self._model = params.model
        self._max_retry_interval_seconds = params.max_retry_interval_seconds
        self._reasoning_effort = params.reasoning_effort
        self._log_raw_output = params.log_raw_output
        self._native_skill_environment = params.native_skill_environment

    @classmethod
    def from_openai_responses_language_model(
        cls,
        model: object,
        *,
        log_raw_output: bool = False,
        native_skill_environment: str = "local",
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
                log_raw_output=log_raw_output,
                native_skill_environment=native_skill_environment,
            )
        )

    async def run_live_session(  # noqa: C901
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, object]],
        tool_registry: dict[str, ToolHandler],
        tool_choice: str | dict[str, str] = "auto",
        max_turns: int = 16,
        timeout_seconds: float | None = None,
        provider_skill_bundles: list[ProviderSkillBundle] | None = None,
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
        normalization_warnings: list[str] = []
        last_response_id: str | None = None
        current_input: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resolved_tools = self._resolve_request_tools(
            tools=tools,
            provider_skill_bundles=provider_skill_bundles,
        )
        resolved_tool_registry = self._resolve_tool_registry(
            tool_registry=tool_registry,
            resolved_tools=resolved_tools,
        )

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
                "tools": resolved_tools,
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

            if self._log_raw_output:
                logger.debug(
                    "OpenAI skill session raw response turn=%d payload=%s",
                    turn_count,
                    self._serialize_response_for_logging(response),
                )

            last_response_id = self._response_id(response)
            usage, usage_warnings = self._response_usage(response)
            normalization_warnings.extend(usage_warnings)
            input_tokens_total += usage["input_tokens"]
            output_tokens_total += usage["output_tokens"]

            raw_model_output = self._response_output_text(response)
            response_items, response_item_warnings = self._response_items(response)
            normalization_warnings.extend(response_item_warnings)
            function_calls = [
                item
                for item in response_items
                if str(item.get("type", "")) == "function_call"
            ]
            shell_calls = [
                item for item in response_items if self._is_actionable_shell_call(item)
            ]
            if not function_calls and not shell_calls:
                return SkillRunResult(
                    final_response=raw_model_output.strip(),
                    raw_model_output=raw_model_output,
                    tool_executions=tool_executions,
                    llm_input_tokens=input_tokens_total,
                    llm_output_tokens=output_tokens_total,
                    llm_time_seconds=llm_time_total,
                    turn_count=turn_count,
                    normalization_warnings=normalization_warnings,
                )

            function_call_outputs: list[dict[str, object]] = []
            shell_call_outputs: list[dict[str, object]] = []

            if function_calls:
                function_executions = await self._execute_function_calls(
                    function_calls=function_calls,
                    tool_registry=resolved_tool_registry,
                )
                tool_executions.extend(function_executions)
                for item in function_executions:
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

            if shell_calls:
                shell_executions, shell_call_outputs = await self._execute_shell_calls(
                    shell_calls=shell_calls
                )
                tool_executions.extend(shell_executions)

            current_input = [*function_call_outputs, *shell_call_outputs]

    @staticmethod
    def _is_actionable_shell_call(item: dict[str, object]) -> bool:
        if str(item.get("type", "")) != "shell_call":
            return False
        action = SkillLanguageModel._as_dict(item.get("action"))
        if not action:
            return False
        commands = action.get("commands")
        return isinstance(commands, list) and len(commands) > 0

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
        call = self._as_dict(raw_call)
        if not call:
            raise SkillToolCallFormatError("Tool call entry must be an object.")
        if call.get("type") != "function_call":
            raise SkillToolCallFormatError(
                "Tool call entry must have type=function_call."
            )

        name = call.get("name")
        arguments = call.get("arguments", {})
        call_id = call.get("call_id")

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
        create = cast(Callable[..., Awaitable[object]], self._client.responses.create)
        for attempt in range(1, max_attempts + 1):
            try:
                return await create(**kwargs)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as err:
                if attempt >= max_attempts:
                    raise SkillLanguageModelError(
                        f"[call uuid: {call_uuid}] OpenAI responses.create failed "
                        f"after {attempt} attempts due to retryable {type(err).__name__}.",
                        diagnostics={
                            "provider": "openai",
                            "operation": "responses.create",
                            "attempt": attempt,
                            "error_type": type(err).__name__,
                            "error_message": str(err),
                            "request_payload": self._request_snapshot(kwargs),
                        },
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
            except TypeError:
                logger.warning(
                    "[call uuid: %s] OpenAI responses.create TypeError; using HTTP "
                    "fallback /responses.",
                    call_uuid,
                )
                return await self._call_responses_create_http_fallback(**kwargs)
            except openai.OpenAIError as err:
                if self._should_fallback_to_http(err):
                    return await self._call_responses_create_http_fallback(**kwargs)
                raise SkillLanguageModelError(
                    f"[call uuid: {call_uuid}] OpenAI responses.create failed "
                    f"with non-retryable {type(err).__name__}.",
                    diagnostics={
                        "provider": "openai",
                        "operation": "responses.create",
                        "error_type": type(err).__name__,
                        "error_message": str(err),
                        "request_payload": self._request_snapshot(kwargs),
                    },
                ) from err

        raise SkillLanguageModelError(
            "responses.create retry loop exited unexpectedly."
        )

    @staticmethod
    def _as_dict(raw_value: object) -> dict[str, object]:
        if isinstance(raw_value, dict):
            return {str(key): value for key, value in raw_value.items()}
        model_dump = getattr(raw_value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return {str(key): value for key, value in dumped.items()}
        raw_dict = getattr(raw_value, "__dict__", None)
        if isinstance(raw_dict, dict):
            return {str(key): value for key, value in raw_dict.items()}
        return {}

    @staticmethod
    def _coerce_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    def _response_id(self, response: object) -> str | None:
        payload = self._as_dict(response)
        value = payload.get("id")
        if value is not None:
            return str(value)
        value = getattr(response, "id", None)
        return str(value) if value is not None else None

    def _response_output_text(self, response: object) -> str:
        payload = self._as_dict(response)
        value = payload.get("output_text")
        if isinstance(value, str):
            return value
        value = getattr(response, "output_text", "")
        return value if isinstance(value, str) else ""

    def _response_usage(self, response: object) -> tuple[dict[str, int], list[str]]:
        warnings: list[str] = []
        payload = self._as_dict(response)
        if payload:
            usage = self._as_dict(payload.get("usage"))
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if not isinstance(input_tokens, int):
                warnings.append("usage_input_tokens_missing_or_invalid")
            if not isinstance(output_tokens, int):
                warnings.append("usage_output_tokens_missing_or_invalid")
            return (
                {
                    "input_tokens": self._coerce_int(input_tokens),
                    "output_tokens": self._coerce_int(output_tokens),
                },
                warnings,
            )
        usage = getattr(response, "usage", None)
        if usage is None:
            return (
                {"input_tokens": 0, "output_tokens": 0},
                ["usage_missing"],
            )
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if not isinstance(input_tokens, int):
            warnings.append("usage_input_tokens_missing_or_invalid")
        if not isinstance(output_tokens, int):
            warnings.append("usage_output_tokens_missing_or_invalid")
        return (
            {
                "input_tokens": self._coerce_int(input_tokens),
                "output_tokens": self._coerce_int(output_tokens),
            },
            warnings,
        )

    def _response_items(  # noqa: C901
        self,
        response: object,
    ) -> tuple[list[dict[str, object]], list[str]]:
        warnings: list[str] = []
        payload = self._as_dict(response)
        if payload:
            items = payload.get("output", [])
            if not isinstance(items, list):
                return [], ["response_output_missing_or_invalid"]
            normalized_items: list[dict[str, object]] = []
            for item in items:
                item_dict = self._as_dict(item)
                if item_dict:
                    item_type = str(item_dict.get("type", ""))
                    if item_type and item_type not in {
                        "function_call",
                        "reasoning",
                        "shell_call",
                        "message",
                    }:
                        warnings.append(f"unsupported_response_item_type:{item_type}")
                    normalized_items.append(item_dict)
            return normalized_items, warnings

        items = getattr(response, "output", []) or []
        normalized: list[dict[str, object]] = []
        for item in items:
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                if item_type and item_type not in {
                    "function_call",
                    "reasoning",
                    "shell_call",
                    "message",
                }:
                    warnings.append(f"unsupported_response_item_type:{item_type}")
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
                continue
            if item_type == "shell_call":
                dump_method = getattr(item, "model_dump", None)
                if callable(dump_method):
                    dumped = dump_method()
                    if isinstance(dumped, dict):
                        normalized.append(dumped)
                        continue
                action_obj = getattr(item, "action", None)
                action: dict[str, object] = {}
                if isinstance(action_obj, dict):
                    action = {str(key): value for key, value in action_obj.items()}
                else:
                    action_dump = getattr(action_obj, "model_dump", None)
                    if callable(action_dump):
                        dumped_action = action_dump()
                        if isinstance(dumped_action, dict):
                            action = dumped_action
                    if not action:
                        commands = getattr(action_obj, "commands", None)
                        timeout_ms = getattr(action_obj, "timeout_ms", None)
                        max_output_length = getattr(
                            action_obj, "max_output_length", None
                        )
                        if isinstance(commands, list):
                            action["commands"] = commands
                        if isinstance(timeout_ms, int):
                            action["timeout_ms"] = timeout_ms
                        if isinstance(max_output_length, int):
                            action["max_output_length"] = max_output_length
                        action["type"] = getattr(action_obj, "type", "exec")
                normalized.append(
                    {
                        "type": "shell_call",
                        "id": getattr(item, "id", None),
                        "call_id": getattr(item, "call_id", None),
                        "status": getattr(item, "status", None),
                        "action": action,
                    }
                )
                continue
            if item_type == "message":
                dump_method = getattr(item, "model_dump", None)
                if callable(dump_method):
                    dumped = dump_method()
                    if isinstance(dumped, dict):
                        normalized.append(dumped)
                        continue
                normalized.append({"type": "message"})
                continue
            if item_type:
                warnings.append(f"unsupported_response_item_type:{item_type}")

        return normalized, warnings

    def _serialize_tool_output(self, output: object) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, default=str)

    def _serialize_response_for_logging(self, response: object) -> str:
        if isinstance(response, dict):
            return json.dumps(response, default=str)
        dump_method = getattr(response, "model_dump", None)
        if callable(dump_method):
            try:
                dumped = dump_method()
                if isinstance(dumped, dict):
                    return json.dumps(dumped, default=str)
            except Exception:
                return repr(response)
        return repr(response)

    def _resolve_tool_registry(
        self,
        *,
        tool_registry: dict[str, ToolHandler],
        resolved_tools: list[dict[str, object]],
    ) -> dict[str, ToolHandler]:
        _ = resolved_tools
        return dict(tool_registry)

    async def _execute_shell_calls(
        self,
        *,
        shell_calls: list[dict[str, object]],
    ) -> tuple[list[SkillToolExecution], list[dict[str, object]]]:
        executions: list[SkillToolExecution] = []
        outputs: list[dict[str, object]] = []
        for raw_call in shell_calls:
            (
                call_id,
                commands,
                timeout_ms,
                max_output_length,
                cwd,
            ) = self._parse_shell_call(raw_call)
            command_results: list[dict[str, object]] = []
            for command in commands:
                result = await self._run_local_shell_command(
                    command=command,
                    timeout_ms=timeout_ms,
                    max_output_length=max_output_length,
                    cwd=cwd,
                )
                command_results.append(result)
            shell_output_item: dict[str, object] = {
                "type": "shell_call_output",
                "call_id": call_id,
                "output": command_results,
            }
            outputs.append(shell_output_item)
            executions.append(
                SkillToolExecution(
                    call_id=call_id,
                    name="shell_call",
                    arguments={
                        "commands": commands,
                        "timeout_ms": timeout_ms,
                        "max_output_length": max_output_length,
                        "cwd": cwd,
                    },
                    output={"output": command_results},
                )
            )
        return executions, outputs

    def _parse_shell_call(
        self,
        raw_call: object,
    ) -> tuple[str, list[str], int, int, str | None]:
        call = self._as_dict(raw_call)
        if not call:
            raise SkillToolCallFormatError("shell_call entry must be an object.")
        if call.get("type") != "shell_call":
            raise SkillToolCallFormatError("Tool call entry must have type=shell_call.")

        raw_call_id = call.get("call_id", call.get("id"))
        if not isinstance(raw_call_id, str) or not raw_call_id.strip():
            raise SkillToolCallFormatError("shell_call missing call identifier.")
        call_id = raw_call_id.strip()

        raw_action = self._as_dict(call.get("action"))
        if not raw_action:
            raise SkillToolCallFormatError("shell_call missing action object.")
        action_type = raw_action.get("type")
        if action_type != "exec":
            raise SkillToolCallFormatError(
                f"Unsupported shell action type: {action_type!r}"
            )

        raw_commands = raw_action.get("commands")
        if not isinstance(raw_commands, list):
            raise SkillToolCallFormatError(
                "shell_call action.commands must be an array."
            )
        commands = [
            command.strip()
            for command in raw_commands
            if isinstance(command, str) and command.strip()
        ]
        if not commands:
            raise SkillToolCallFormatError("shell_call has no executable commands.")

        timeout_ms_raw = raw_action.get("timeout_ms", 15_000)
        timeout_ms = timeout_ms_raw if isinstance(timeout_ms_raw, int) else 15_000
        timeout_ms = max(1_000, min(timeout_ms, 120_000))

        max_output_length_raw = raw_action.get("max_output_length", 6_000)
        max_output_length = (
            max_output_length_raw if isinstance(max_output_length_raw, int) else 6_000
        )
        max_output_length = max(512, min(max_output_length, 50_000))

        cwd: str | None = None
        raw_cwd = raw_action.get("cwd")
        if isinstance(raw_cwd, str) and raw_cwd.strip():
            cwd = raw_cwd.strip()

        return call_id, commands, timeout_ms, max_output_length, cwd

    async def _run_local_shell_command(
        self,
        *,
        command: str,
        timeout_ms: int,
        max_output_length: int,
        cwd: str | None,
    ) -> dict[str, object]:
        if not self._is_allowed_local_shell_command(command):
            return {
                "stdout": "",
                "stderr": (
                    "command not allowed by local shell policy: "
                    f"{command}. Allowed base commands: "
                    f"{sorted(self._LOCAL_SHELL_ALLOWED_BASE_COMMANDS)}"
                ),
                "outcome": {"type": "exit", "exit_code": 126},
            }

        resolved_cwd, invalid_cwd = self._resolve_local_shell_cwd(cwd)
        if invalid_cwd is not None:
            return {
                "stdout": "",
                "stderr": f"invalid cwd for local shell command: {invalid_cwd}",
                "outcome": {"type": "exit", "exit_code": 2},
            }

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(resolved_cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as err:
            return {
                "stdout": "",
                "stderr": (
                    f"failed to start local shell command ({type(err).__name__}): {err}"
                ),
                "outcome": {"type": "exit", "exit_code": 1},
            }

        timeout_seconds = timeout_ms / 1000
        timed_out = False
        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            timed_out = True
            process.kill()
            stdout_raw, stderr_raw = await process.communicate()

        stdout = stdout_raw.decode("utf-8", errors="replace")
        stderr = stderr_raw.decode("utf-8", errors="replace")
        if len(stdout) > max_output_length:
            stdout = f"{stdout[:max_output_length]}...[truncated]"
        if len(stderr) > max_output_length:
            stderr = f"{stderr[:max_output_length]}...[truncated]"

        if timed_out:
            return {
                "stdout": stdout,
                "stderr": stderr,
                "outcome": {"type": "timeout"},
            }
        return {
            "stdout": stdout,
            "stderr": stderr,
            "outcome": {"type": "exit", "exit_code": int(process.returncode or 0)},
        }

    def _is_allowed_local_shell_command(self, command: str) -> bool:
        if any(token in command for token in self._LOCAL_SHELL_DISALLOWED_TOKENS):
            return False
        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts:
            return False
        return parts[0] in self._LOCAL_SHELL_ALLOWED_BASE_COMMANDS

    @staticmethod
    def _resolve_local_shell_cwd(cwd: str | None) -> tuple[Path, str | None]:
        resolved_cwd = Path.cwd()
        if isinstance(cwd, str) and cwd.strip():
            candidate = Path(cwd.strip()).expanduser()
            if candidate.exists() and candidate.is_dir():
                return candidate, None
            return resolved_cwd, str(candidate)
        return resolved_cwd, None

    @staticmethod
    def serialize_object_for_diagnostics(
        payload: object,
        *,
        max_chars: int = 20000,
    ) -> str:
        """Serialize diagnostic payloads with hard truncation."""
        try:
            serialized = json.dumps(payload, default=str)
        except Exception:
            serialized = repr(payload)
        if len(serialized) <= max_chars:
            return serialized
        return f"{serialized[:max_chars]}...[truncated]"

    def _request_snapshot(self, request: dict[str, object] | dict[str, Any]) -> str:
        return self.serialize_object_for_diagnostics(request)

    def _resolve_request_tools(
        self,
        *,
        tools: list[dict[str, object]],
        provider_skill_bundles: list[ProviderSkillBundle] | None,
    ) -> list[dict[str, object]]:
        if not provider_skill_bundles:
            return list(tools)
        if self._native_skill_environment != "local":
            logger.warning(
                "OpenAI native skill environment '%s' is not supported in this "
                "runtime yet; skipping provider skill attachment.",
                self._native_skill_environment,
            )
            return list(tools)
        has_shell_tool = any(
            isinstance(tool, dict) and tool.get("type") == "shell" for tool in tools
        )
        if has_shell_tool:
            return list(tools)
        shell_tool: dict[str, object] = {
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [
                    {
                        "name": bundle.name,
                        "description": bundle.description,
                        "path": bundle.path,
                    }
                    for bundle in provider_skill_bundles
                ],
            },
        }
        return [shell_tool, *tools]

    @staticmethod
    def _should_fallback_to_http(err: openai.OpenAIError) -> bool:
        message = str(err).lower()
        return "unknown parameter" in message or "invalid type" in message

    async def _call_responses_create_http_fallback(self, **kwargs: object) -> object:
        api_key = getattr(self._client, "api_key", None)
        if not isinstance(api_key, str) or not api_key:
            raise SkillLanguageModelError(
                "OpenAI HTTP fallback requires client.api_key to be set."
            )
        base_url_raw = getattr(self._client, "base_url", "https://api.openai.com/v1/")
        base_url = str(base_url_raw).rstrip("/") + "/"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        request_snapshot = self._request_snapshot(kwargs)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
                response = await client.post(
                    "responses",
                    headers=headers,
                    json=kwargs,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as err:
            response = err.response
            raise SkillLanguageModelError(
                "OpenAI HTTP fallback /responses request failed.",
                diagnostics={
                    "provider": "openai",
                    "operation": "responses.http_fallback",
                    "error_type": type(err).__name__,
                    "status_code": response.status_code,
                    "response_body": self.serialize_object_for_diagnostics(
                        response.text
                    ),
                    "request_payload": request_snapshot,
                },
            ) from err
        except httpx.HTTPError as err:
            raise SkillLanguageModelError(
                "OpenAI HTTP fallback /responses request failed.",
                diagnostics={
                    "provider": "openai",
                    "operation": "responses.http_fallback",
                    "error_type": type(err).__name__,
                    "error_message": str(err),
                    "request_payload": request_snapshot,
                },
            ) from err
