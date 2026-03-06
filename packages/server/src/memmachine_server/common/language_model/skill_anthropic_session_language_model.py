"""Anthropic live-session runtime for skill-style function-calling orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

import httpx
import json_repair
from pydantic import BaseModel, Field, field_validator

from .provider_skill_bundle import ProviderSkillBundle
from .skill_openai_session_language_model import (
    SkillLanguageModelError,
    SkillRunResult,
    SkillSessionLimitError,
    SkillToolCallFormatError,
    SkillToolExecution,
    SkillToolNotFoundError,
)

ToolHandler = Callable[[dict[str, object]], object | Awaitable[object]]

logger = logging.getLogger(__name__)


class SkillAnthropicSessionLanguageModelParams(BaseModel):
    """Configuration for Anthropic skill live-session runtime."""

    client: object
    model: str = Field(min_length=1)
    max_retry_interval_seconds: int = Field(default=120, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    temperature: float | None = None
    log_raw_output: bool = False

    @field_validator("client")
    @classmethod
    def _validate_client(cls, value: object) -> object:
        messages = getattr(value, "messages", None)
        create = getattr(messages, "create", None)
        if not callable(create):
            raise TypeError(
                "SkillAnthropicSessionLanguageModel requires client.messages.create"
            )
        return value


class SkillAnthropicSessionLanguageModel:
    """Anthropic Messages function-calling live session runner."""

    def __init__(self, params: SkillAnthropicSessionLanguageModelParams) -> None:
        """Initialize runtime with Anthropic Messages client settings."""
        self._client = params.client
        self._model = params.model
        self._max_retry_interval_seconds = params.max_retry_interval_seconds
        self._max_output_tokens = params.max_output_tokens
        self._temperature = params.temperature
        self._log_raw_output = params.log_raw_output
        self._skill_id_cache: dict[str, str] = {}

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
        """Run one live model session until no more tool calls are emitted."""
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

        messages: list[dict[str, object]] = [
            {"role": "user", "content": user_prompt},
        ]
        anthropic_tools = self._to_anthropic_tools(tools)
        native_skill_refs: list[dict[str, object]] = []
        if provider_skill_bundles:
            native_skill_refs = await self._resolve_native_skill_refs(
                provider_skill_bundles
            )
            anthropic_tools.append(self._code_execution_tool())

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
                "system": system_prompt,
                "messages": messages,
                "tools": anthropic_tools,
                "max_tokens": self._max_output_tokens,
            }
            if native_skill_refs:
                request["container"] = {"skills": native_skill_refs}
                request["betas"] = ["skills-2025-10-02"]
            if self._temperature is not None:
                request["temperature"] = self._temperature
            # Keep the same function signature as OpenAI runtime. Anthropic tool
            # choice options differ, so ignore unsupported values for now.
            if tool_choice not in {"auto", "required"}:
                normalization_warnings.append(
                    "anthropic_unsupported_tool_choice_ignored"
                )

            llm_call_started = time.monotonic()
            response = await self._call_messages_create_with_retry(
                max_attempts=3,
                use_beta=bool(native_skill_refs),
                **request,
            )
            llm_time_total += time.monotonic() - llm_call_started
            turn_count += 1

            if self._log_raw_output:
                logger.debug(
                    "Anthropic skill session raw response turn=%d payload=%s",
                    turn_count,
                    self._serialize_response_for_logging(response),
                )

            usage, usage_warnings = self._response_usage(response)
            normalization_warnings.extend(usage_warnings)
            input_tokens_total += usage["input_tokens"]
            output_tokens_total += usage["output_tokens"]

            content_blocks, block_warnings = self._response_content_blocks(response)
            normalization_warnings.extend(block_warnings)
            raw_model_output = self._extract_text(content_blocks)
            tool_uses = [
                block for block in content_blocks if block.get("type") == "tool_use"
            ]
            if not tool_uses:
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

            messages.append({"role": "assistant", "content": content_blocks})
            executions = await self._execute_tool_uses(
                tool_uses=tool_uses,
                tool_registry=tool_registry,
            )
            tool_executions.extend(executions)

            tool_results: list[dict[str, object]] = []
            for item in executions:
                if item.call_id is None:
                    raise SkillToolCallFormatError(
                        "tool_use block missing id for tool_result callback."
                    )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": item.call_id,
                        "content": self._serialize_tool_output(item.output),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

    async def _execute_tool_uses(
        self,
        *,
        tool_uses: object,
        tool_registry: dict[str, ToolHandler],
    ) -> list[SkillToolExecution]:
        if not isinstance(tool_uses, list):
            raise SkillToolCallFormatError("tool_uses must be a list")
        results: list[SkillToolExecution] = []
        for raw_tool_use in tool_uses:
            call_id, name, arguments = self._parse_tool_use(raw_tool_use)
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

    def _parse_tool_use(
        self,
        raw_tool_use: object,
    ) -> tuple[str | None, str, dict[str, object]]:
        if not isinstance(raw_tool_use, dict):
            raise SkillToolCallFormatError("Tool call entry must be an object.")
        if raw_tool_use.get("type") != "tool_use":
            raise SkillToolCallFormatError("Tool call entry must have type=tool_use.")

        name = raw_tool_use.get("name")
        tool_input = raw_tool_use.get("input", {})
        call_id = raw_tool_use.get("id")
        if not isinstance(name, str) or not name.strip():
            raise SkillToolCallFormatError("Tool call missing function name.")
        parsed_args = self._parse_arguments(tool_input)
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
                    "Failed to parse tool_use input arguments."
                ) from err
            if not isinstance(parsed, dict):
                raise SkillToolCallFormatError("Tool input must decode to object.")
            return {str(key): value for key, value in parsed.items()}
        raise SkillToolCallFormatError(
            "Tool input arguments must be object or JSON string."
        )

    async def _call_messages_create_with_retry(
        self,
        *,
        max_attempts: int,
        use_beta: bool = False,
        **kwargs: object,
    ) -> object:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        create = self._resolve_messages_create(use_beta=use_beta)
        call_uuid = uuid4()
        sleep_seconds = 1
        request_snapshot = self._request_snapshot(kwargs)
        for attempt in range(1, max_attempts + 1):
            try:
                return await create(**kwargs)
            except Exception as err:
                retryable = self._is_retryable_error(err)
                if retryable and attempt < max_attempts:
                    logger.info(
                        "[call uuid: %s] Retrying Anthropic messages.create in %d "
                        "second(s) after attempt %d (%s).",
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
                    continue
                response = getattr(err, "response", None)
                status_code = getattr(response, "status_code", None)
                response_body = None
                if response is not None:
                    response_text = getattr(response, "text", None)
                    if isinstance(response_text, str) and response_text:
                        response_body = self._serialize_object_for_diagnostics(
                            response_text
                        )
                raise SkillLanguageModelError(
                    f"[call uuid: {call_uuid}] Anthropic messages.create failed "
                    f"with {type(err).__name__}.",
                    diagnostics={
                        "provider": "anthropic",
                        "operation": (
                            "beta.messages.create"
                            if use_beta
                            else "messages.create"
                        ),
                        "attempt": attempt,
                        "error_type": type(err).__name__,
                        "error_message": str(err),
                        "status_code": status_code,
                        "response_body": response_body,
                        "request_payload": request_snapshot,
                    },
                ) from err

        raise SkillLanguageModelError("messages.create retry loop exited unexpectedly.")

    def _resolve_messages_create(self, *, use_beta: bool) -> Callable[..., Awaitable]:
        if not use_beta:
            create = getattr(getattr(self._client, "messages", None), "create", None)
            if callable(create):
                return create
            raise SkillLanguageModelError(
                "Anthropic client.messages.create is unavailable."
            )

        beta = getattr(self._client, "beta", None)
        create = getattr(getattr(beta, "messages", None), "create", None)
        if callable(create):
            return create
        raise SkillLanguageModelError(
            "Anthropic beta.messages.create is required for native skills."
        )

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        retryable_error_names = {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "OverloadedError",
        }
        return type(error).__name__ in retryable_error_names

    def _response_usage(self, response: object) -> tuple[dict[str, int], list[str]]:
        warnings: list[str] = []
        usage_raw: object = None
        if isinstance(response, dict):
            usage_raw = response.get("usage")
        else:
            usage_raw = getattr(response, "usage", None)

        input_tokens = 0
        output_tokens = 0
        if isinstance(usage_raw, dict):
            raw_input = usage_raw.get("input_tokens")
            raw_output = usage_raw.get("output_tokens")
            if isinstance(raw_input, int):
                input_tokens = raw_input
            else:
                warnings.append("usage_input_tokens_missing_or_invalid")
            if isinstance(raw_output, int):
                output_tokens = raw_output
            else:
                warnings.append("usage_output_tokens_missing_or_invalid")
        elif usage_raw is not None:
            raw_input = getattr(usage_raw, "input_tokens", None)
            raw_output = getattr(usage_raw, "output_tokens", None)
            if isinstance(raw_input, int):
                input_tokens = raw_input
            else:
                warnings.append("usage_input_tokens_missing_or_invalid")
            if isinstance(raw_output, int):
                output_tokens = raw_output
            else:
                warnings.append("usage_output_tokens_missing_or_invalid")
        else:
            warnings.append("usage_missing")

        return (
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            warnings,
        )

    def _response_content_blocks(
        self,
        response: object,
    ) -> tuple[list[dict[str, object]], list[str]]:
        warnings: list[str] = []
        content_raw: object
        if isinstance(response, dict):
            content_raw = response.get("content", [])
        else:
            content_raw = getattr(response, "content", [])

        if not isinstance(content_raw, list):
            return [], ["response_content_missing_or_invalid"]

        normalized: list[dict[str, object]] = []
        for raw_block in content_raw:
            block, warning = self._normalize_content_block(raw_block)
            if warning is not None:
                warnings.append(warning)
            if block is not None:
                normalized.append(block)
        return normalized, warnings

    def _normalize_content_block(  # noqa: C901
        self,
        raw_block: object,
    ) -> tuple[dict[str, object] | None, str | None]:
        if isinstance(raw_block, dict):
            block_type = raw_block.get("type")
            if block_type == "text":
                text = raw_block.get("text")
                if not isinstance(text, str):
                    return None, "text_block_missing_text"
                return {"type": "text", "text": text}, None
            if block_type == "tool_use":
                name = raw_block.get("name")
                if not isinstance(name, str) or not name.strip():
                    return None, "tool_use_block_missing_name"
                block: dict[str, object] = {
                    "type": "tool_use",
                    "name": name.strip(),
                    "input": raw_block.get("input", {}),
                }
                call_id = raw_block.get("id")
                if isinstance(call_id, str):
                    block["id"] = call_id
                return block, None
            return None, f"unsupported_content_block_type:{block_type}"

        block_type = getattr(raw_block, "type", None)
        if block_type == "text":
            text = getattr(raw_block, "text", None)
            if not isinstance(text, str):
                return None, "text_block_missing_text"
            return {"type": "text", "text": text}, None
        if block_type == "tool_use":
            name = getattr(raw_block, "name", None)
            if not isinstance(name, str) or not name.strip():
                return None, "tool_use_block_missing_name"
            block = {
                "type": "tool_use",
                "name": name.strip(),
                "input": getattr(raw_block, "input", {}),
            }
            call_id = getattr(raw_block, "id", None)
            if isinstance(call_id, str):
                block["id"] = call_id
            return block, None

        return None, f"unsupported_content_block_type:{block_type}"

    @staticmethod
    def _extract_text(content_blocks: list[dict[str, object]]) -> str:
        chunks: list[str] = []
        for block in content_blocks:
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks).strip()

    @staticmethod
    def _to_anthropic_tools(tools: list[dict[str, object]]) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for raw_tool in tools:
            if not isinstance(raw_tool, dict):
                continue
            if raw_tool.get("type") != "function":
                continue
            name = raw_tool.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            input_schema = raw_tool.get("parameters")
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}}
            description = raw_tool.get("description")
            normalized.append(
                {
                    "name": name.strip(),
                    "description": description if isinstance(description, str) else "",
                    "input_schema": input_schema,
                }
            )
        return normalized

    @staticmethod
    def _serialize_tool_output(output: object) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, default=str)

    @staticmethod
    def _serialize_response_for_logging(response: object) -> str:
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

    @staticmethod
    def _serialize_object_for_diagnostics(
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

    def _request_snapshot(self, request: dict[str, object]) -> str:
        return self._serialize_object_for_diagnostics(request)

    async def _resolve_native_skill_refs(
        self,
        bundles: list[ProviderSkillBundle],
    ) -> list[dict[str, object]]:
        refs: list[dict[str, object]] = []
        for bundle in bundles:
            key = bundle.path
            skill_id = self._skill_id_cache.get(key)
            if skill_id is None:
                skill_id = await self._create_native_skill(bundle)
                self._skill_id_cache[key] = skill_id
            refs.append(
                {
                    "type": "custom",
                    "skill_id": skill_id,
                    "version": "latest",
                }
            )
        return refs

    async def _create_native_skill(self, bundle: ProviderSkillBundle) -> str:
        files = self._bundle_files(bundle.path)
        beta = getattr(self._client, "beta", None)
        create = getattr(getattr(beta, "skills", None), "create", None)
        if callable(create):
            try:
                response = await create(
                    display_title=bundle.name,
                    files=files,
                )
                return self._extract_skill_id(response)
            except Exception as err:
                logger.warning(
                    "Anthropic SDK skill upload failed for '%s'; falling back to HTTP: %s",
                    bundle.name,
                    type(err).__name__,
                )
        return await self._create_native_skill_http(bundle=bundle, files=files)

    async def _create_native_skill_http(
        self,
        *,
        bundle: ProviderSkillBundle,
        files: list[tuple[str, Path]],
    ) -> str:
        api_key = getattr(self._client, "api_key", None)
        if not isinstance(api_key, str) or not api_key:
            raise SkillLanguageModelError(
                "Anthropic native skill HTTP fallback requires api_key."
            )
        base_url = str(getattr(self._client, "base_url", "https://api.anthropic.com"))
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "skills-2025-10-02",
        }
        with contextlib.ExitStack() as stack:
            multipart_files: list[tuple[str, tuple[str, object, str]]] = []
            for rel_name, path in files:
                handle = stack.enter_context(path.open("rb"))
                content_type = (
                    "text/markdown"
                    if rel_name.lower().endswith(".md")
                    else "application/octet-stream"
                )
                multipart_files.append(
                    ("files", (rel_name, handle, content_type))
                )

            async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
                response = await client.post(
                    "/v1/skills?beta=true",
                    headers=headers,
                    data={"display_title": bundle.name},
                    files=multipart_files,
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as err:
                    raise SkillLanguageModelError(
                        "Anthropic native skill HTTP upload failed.",
                        diagnostics={
                            "provider": "anthropic",
                            "operation": "skills.create.http_fallback",
                            "error_type": type(err).__name__,
                            "status_code": response.status_code,
                            "response_body": self._serialize_object_for_diagnostics(
                                response.text
                            ),
                            "skill_name": bundle.name,
                            "bundle_path": bundle.path,
                        },
                    ) from err
                payload = response.json()
        return self._extract_skill_id(payload)

    @staticmethod
    def _extract_skill_id(response: object) -> str:
        if isinstance(response, dict):
            raw_id = response.get("id")
        else:
            raw_id = getattr(response, "id", None)
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id
        raise SkillLanguageModelError("Anthropic skill upload returned no skill id.")

    @staticmethod
    def _bundle_files(bundle_path: str) -> list[tuple[str, Path]]:
        root = Path(bundle_path)
        skill_md = root / "SKILL.md"
        if not skill_md.exists():
            raise SkillLanguageModelError(
                f"Skill bundle path must include SKILL.md: {bundle_path}"
            )
        return [
            (path.relative_to(root).as_posix(), path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    @staticmethod
    def _code_execution_tool() -> dict[str, object]:
        return {
            "type": "code_execution_20250825",
            "name": "code_execution",
        }
