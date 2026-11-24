"""Tests that provider clients are built with request timeouts.

Language-model and reranker calls used to be created with SDK default
timeouts (ten minutes for OpenAI), so one slow or hung provider held
searches and stalled semantic extraction for every project. These tests
pin that the configured timeout reaches the underlying client.
"""

import openai
import pytest
from pydantic import SecretStr

from memmachine_server.common.configuration.language_model_conf import (
    LanguageModelsConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)


def _capture_openai_clients(monkeypatch):
    """Patch AsyncOpenAI to record constructor kwargs without network setup."""
    calls = []

    class FakeAsyncOpenAI(openai.AsyncOpenAI):
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    return calls


@pytest.mark.asyncio
async def test_openai_language_models_pass_configured_timeout(monkeypatch):
    calls = _capture_openai_clients(monkeypatch)
    conf = LanguageModelsConf(
        openai_responses_language_model_confs={
            "responses_model": OpenAIResponsesLanguageModelConf(
                model="gpt-4o-mini",
                api_key=SecretStr("DUMMY_KEY"),
                request_timeout_seconds=45.0,
            ),
        },
        openai_chat_completions_language_model_confs={
            "chat_model": OpenAIChatCompletionsLanguageModelConf(
                model="llama3",
                api_key=SecretStr("DUMMY_KEY"),
                base_url="http://localhost:11434/v1",
                request_timeout_seconds=60.0,
            ),
        },
    )

    from memmachine_server.common.resource_manager.language_model_manager import (
        LanguageModelManager,
    )

    builder = LanguageModelManager(conf)
    await builder.build_all()

    assert "responses_model" in builder._language_models
    assert "chat_model" in builder._language_models
    assert {"api_key": "DUMMY_KEY", "base_url": None, "timeout": 45.0} in calls
    assert {
        "api_key": "DUMMY_KEY",
        "base_url": "http://localhost:11434/v1",
        "timeout": 60.0,
    } in calls


@pytest.mark.asyncio
async def test_openai_language_models_default_timeout(monkeypatch):
    calls = _capture_openai_clients(monkeypatch)
    conf = LanguageModelsConf(
        openai_responses_language_model_confs={
            "responses_model": OpenAIResponsesLanguageModelConf(
                model="gpt-4o-mini",
                api_key=SecretStr("DUMMY_KEY"),
            ),
        },
    )

    from memmachine_server.common.resource_manager.language_model_manager import (
        LanguageModelManager,
    )

    builder = LanguageModelManager(conf)
    await builder.build_all()

    assert "responses_model" in builder._language_models
    assert calls == [{"api_key": "DUMMY_KEY", "base_url": None, "timeout": 120.0}]
