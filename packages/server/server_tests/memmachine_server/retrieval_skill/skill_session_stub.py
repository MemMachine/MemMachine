from __future__ import annotations

import json
from typing import Any

from memmachine_server.common.language_model import (
    ProviderSkillBundle,
    SkillRunResult,
    SkillSessionLimitError,
    SkillToolCallFormatError,
    SkillToolExecution,
    SkillToolNotFoundError,
)
from memmachine_server.common.language_model.language_model import LanguageModel


class ScriptedSkillSessionModel:
    """Test-only session runtime that executes one scripted model turn."""

    def __init__(self, model: LanguageModel) -> None:
        self._model = model
        self._session_call_count = 0
        self.provider_skill_bundles_history: list[list[ProviderSkillBundle]] = []

    async def run_live_session(  # noqa: C901
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, object]],
        tool_registry: dict[str, Any],
        tool_choice: str | dict[str, str] = "auto",
        max_turns: int = 16,
        timeout_seconds: float | None = None,
        provider_skill_bundles: list[ProviderSkillBundle] | None = None,
    ) -> SkillRunResult:
        _ = max_turns, timeout_seconds
        self.provider_skill_bundles_history.append(list(provider_skill_bundles or []))
        output, function_calls = await self._model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_choice=tool_choice,
        )
        calls = function_calls or []
        if not isinstance(calls, list):
            raise SkillToolCallFormatError("function_calls must be a list")
        if len(calls) > max_turns:
            raise SkillSessionLimitError("Skill session exceeded max_turns.")

        call_index = self._session_call_count
        llm_time_seconds = 0.0
        raw_times = getattr(self._model, "session_llm_times", None)
        if isinstance(raw_times, list):
            if call_index < len(raw_times):
                raw_value = raw_times[call_index]
                if isinstance(raw_value, int | float):
                    llm_time_seconds = float(raw_value)
        elif isinstance(raw_times, int | float):
            llm_time_seconds = float(raw_times)

        llm_input_tokens = 0
        raw_input_tokens = getattr(self._model, "session_llm_input_tokens", None)
        if isinstance(raw_input_tokens, list):
            if call_index < len(raw_input_tokens):
                raw_value = raw_input_tokens[call_index]
                if isinstance(raw_value, int | float):
                    llm_input_tokens = int(raw_value)
        elif isinstance(raw_input_tokens, int | float):
            llm_input_tokens = int(raw_input_tokens)

        llm_output_tokens = 0
        raw_output_tokens = getattr(self._model, "session_llm_output_tokens", None)
        if isinstance(raw_output_tokens, list):
            if call_index < len(raw_output_tokens):
                raw_value = raw_output_tokens[call_index]
                if isinstance(raw_value, int | float):
                    llm_output_tokens = int(raw_value)
        elif isinstance(raw_output_tokens, int | float):
            llm_output_tokens = int(raw_output_tokens)

        self._session_call_count += 1

        executions: list[SkillToolExecution] = []
        for raw in calls:
            call_id, name, arguments = self._parse_call(raw)
            handler = tool_registry.get(name)
            if handler is None:
                raise SkillToolNotFoundError(f"No handler registered for tool: {name}")
            result = handler(arguments)
            if hasattr(result, "__await__"):
                result = await result
            executions.append(
                SkillToolExecution(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    output=result,
                )
            )

        return SkillRunResult(
            final_response=(output or "").strip(),
            raw_model_output=output or "",
            tool_executions=executions,
            llm_input_tokens=llm_input_tokens,
            llm_output_tokens=llm_output_tokens,
            llm_time_seconds=llm_time_seconds,
            turn_count=1,
        )

    def _parse_call(
        self,
        raw: object,
    ) -> tuple[str | None, str, dict[str, object]]:
        if not isinstance(raw, dict):
            raise SkillToolCallFormatError("Tool call entry must be an object.")

        call_id = raw.get("call_id")
        if raw.get("type") == "function_call":
            name = raw.get("name")
            arguments = raw.get("arguments", {})
        else:
            function = raw.get("function")
            if not isinstance(function, dict):
                raise SkillToolCallFormatError("Tool call missing function object.")
            name = function.get("name")
            arguments = function.get("arguments", {})

        if not isinstance(name, str) or not name.strip():
            raise SkillToolCallFormatError("Tool call missing function name.")
        parsed_arguments = self._parse_arguments(arguments)
        parsed_call_id = str(call_id) if isinstance(call_id, str) else None
        return parsed_call_id, name.strip(), parsed_arguments

    def _parse_arguments(self, raw_arguments: object) -> dict[str, object]:
        if isinstance(raw_arguments, dict):
            return {str(key): value for key, value in raw_arguments.items()}
        if isinstance(raw_arguments, str):
            parsed = json.loads(raw_arguments)
            if not isinstance(parsed, dict):
                raise SkillToolCallFormatError(
                    "Function arguments must decode to object."
                )
            return {str(key): value for key, value in parsed.items()}
        raise SkillToolCallFormatError(
            "Function call arguments must be object or JSON string."
        )
