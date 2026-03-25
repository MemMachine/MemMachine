from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from memmachine_common.api import MemoryType

from memmachine_server.common.episode_store import Episode, EpisodeResponse
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.retrieval_agent.common.agent_api import QueryParam
from memmachine_server.semantic_memory.semantic_model import SemanticFeature


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


class FakeRestMemory:
    def __init__(
        self,
        episodic_by_query: dict[str, list[Episode]] | None = None,
        *,
        semantic_by_query: dict[str, list[SemanticFeature]] | None = None,
    ) -> None:
        self._episodic_by_query = episodic_by_query or {}
        self._semantic_by_query = semantic_by_query or {}
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float | None = None,
        agent_mode: bool = False,
    ) -> dict[str, object]:
        _ = expand_context, score_threshold, agent_mode
        self.queries.append(query)
        content: dict[str, object] = {}

        episodic = self._episodic_by_query.get(query, [])
        search_limit = limit if limit is not None else len(episodic)
        content["episodic_memory"] = {
            "long_term_memory": {
                "episodes": [
                    EpisodeResponse(score=1.0, **episode.model_dump()).model_dump(
                        mode="json"
                    )
                    for episode in episodic[:search_limit]
                ]
            },
            "short_term_memory": {
                "episodes": [],
                "episode_summary": [],
            },
        }

        semantic = self._semantic_by_query.get(query)
        if semantic is not None:
            content["semantic_memory"] = [
                feature.model_dump(mode="json") for feature in semantic
            ]

        return {"content": content}


def build_query_param(
    *,
    query: str,
    rest_memory: FakeRestMemory,
    limit: int = 5,
    expand_context: int = 0,
    score_threshold: float = -float("inf"),
    session_key: str = "test-session",
    target_memories: list[MemoryType] | None = None,
) -> QueryParam:
    return QueryParam(
        query=query,
        limit=limit,
        expand_context=expand_context,
        score_threshold=score_threshold,
        session_key=session_key,
        rest_memory=rest_memory,
        target_memories=target_memories or [MemoryType.Episodic],
    )


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
