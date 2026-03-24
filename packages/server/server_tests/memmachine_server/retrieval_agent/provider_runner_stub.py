from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from memmachine_server.common.language_model.language_model import LanguageModel


@dataclass
class _FakeUploadedFile:
    id: str


class _FakeOpenAIFilesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._counter = 0

    async def create(self, *, file: object, purpose: str) -> _FakeUploadedFile:
        self._counter += 1
        self.calls.append(
            {
                "file_name": getattr(file, "name", ""),
                "purpose": purpose,
            }
        )
        return _FakeUploadedFile(id=f"file-{self._counter}")


class _FakeOpenAIResponsesAPI:
    def __init__(self, scripted_responses: list[object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._scripted_responses = list(scripted_responses or [])

    async def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if not self._scripted_responses:
            return openai_text_response("")
        response = self._scripted_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeOpenAIClient:
    def __init__(self, scripted_responses: list[object] | None = None) -> None:
        self.files = _FakeOpenAIFilesAPI()
        self.responses = _FakeOpenAIResponsesAPI(scripted_responses)


class FakeOpenAIInstalledAgentModel(LanguageModel):
    def __init__(
        self,
        scripted_responses: list[object] | None = None,
        *,
        model_name: str = "gpt-5-mini",
    ) -> None:
        self.client = FakeOpenAIClient(scripted_responses)
        self.model_name = model_name

    async def generate_parsed_response(
        self,
        output_format: type[Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any | None:
        _ = output_format, system_prompt, user_prompt, max_attempts
        return None

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        _ = system_prompt, user_prompt, tools, tool_choice, max_attempts
        return "", []

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        _ = system_prompt, user_prompt, tools, tool_choice, max_attempts
        return "", [], 0, 0


def openai_text_response(
    text: str,
    *,
    response_id: str = "resp-text",
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> dict[str, object]:
    return {
        "id": response_id,
        "output_text": text,
        "output": [],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def openai_function_call(
    *,
    query: str = "",
    tool_name: str = "memmachine_search",
    call_id: str = "call-1",
    arguments: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = arguments if arguments is not None else {"query": query}
    return {
        "type": "function_call",
        "name": tool_name,
        "arguments": json.dumps(payload),
        "call_id": call_id,
    }


def openai_multi_tool_call_response(
    *,
    calls: list[dict[str, object]],
    response_id: str = "resp-tool",
    output_text: str = "",
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> dict[str, object]:
    return {
        "id": response_id,
        "output_text": output_text,
        "output": list(calls),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def openai_tool_call_response(
    *,
    query: str,
    tool_name: str = "memmachine_search",
    call_id: str = "call-1",
    response_id: str = "resp-tool",
    output_text: str = "",
    arguments: dict[str, object] | None = None,
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> dict[str, object]:
    return openai_multi_tool_call_response(
        calls=[
            openai_function_call(
                query=query,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
            )
        ],
        response_id=response_id,
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
