from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import openai
import pytest

from memmachine_server.common.configuration.retrieval_config import (
    RetrievalAgentConf,
    RetrievalSkillSessionProvider,
)
from memmachine_server.common.language_model import (
    SkillAnthropicSessionLanguageModel,
    SkillLanguageModel,
    create_skill_session_model,
)
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)


class _DummyLanguageModel(LanguageModel):
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
        return "", None

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        _ = system_prompt, user_prompt, tools, tool_choice, max_attempts
        return "", None, 0, 0


@pytest.fixture
def mock_async_openai_client():
    with patch("openai.AsyncOpenAI", spec=openai.AsyncOpenAI):
        client = openai.AsyncOpenAI(api_key="test-key")
        client.responses.create = AsyncMock()
        yield client


def test_factory_builds_openai_runtime(mock_async_openai_client: Any) -> None:
    base = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5-mini",
        )
    )
    conf = RetrievalAgentConf(
        skill_session_provider=RetrievalSkillSessionProvider.OPENAI,
    )

    runtime = create_skill_session_model(model=base, retrieval_conf=conf)
    assert isinstance(runtime, SkillLanguageModel)


def test_factory_rejects_non_openai_model_for_openai_provider() -> None:
    conf = RetrievalAgentConf(
        skill_session_provider=RetrievalSkillSessionProvider.OPENAI,
    )
    with pytest.raises(TypeError, match="requires OpenAIResponsesLanguageModel"):
        _ = create_skill_session_model(
            model=_DummyLanguageModel(),
            retrieval_conf=conf,
        )


def test_factory_requires_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conf = RetrievalAgentConf(
        skill_session_provider=RetrievalSkillSessionProvider.ANTHROPIC,
        anthropic_api_key=None,
    )
    with pytest.raises(ValueError, match="requires anthropic_api_key"):
        _ = create_skill_session_model(
            model=_DummyLanguageModel(),
            retrieval_conf=conf,
        )


def test_factory_builds_anthropic_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.messages = type("_Messages", (), {})()
            self.messages.create = AsyncMock()

    fake_module = ModuleType("anthropic")
    fake_module.AsyncAnthropic = _FakeAsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    conf = RetrievalAgentConf(
        skill_session_provider=RetrievalSkillSessionProvider.ANTHROPIC,
        anthropic_api_key="$ANTHROPIC_API_KEY",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")

    runtime = create_skill_session_model(
        model=_DummyLanguageModel(),
        retrieval_conf=conf,
    )

    assert isinstance(runtime, SkillAnthropicSessionLanguageModel)
