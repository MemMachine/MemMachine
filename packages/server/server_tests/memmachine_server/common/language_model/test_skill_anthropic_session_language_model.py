from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from memmachine_server.common.language_model import (
    ProviderSkillBundle,
    SkillAnthropicSessionLanguageModel,
    SkillAnthropicSessionLanguageModelParams,
    SkillSessionLimitError,
    SkillToolNotFoundError,
)


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = type("_Messages", (), {})()
        self.messages.create = AsyncMock()
        self.beta = type("_Beta", (), {})()
        self.beta.messages = type("_Messages", (), {})()
        self.beta.messages.create = AsyncMock()
        self.beta.skills = type("_Skills", (), {})()
        self.beta.skills.create = AsyncMock()
        self.api_key = "anthropic-test-key"
        self.base_url = "https://api.anthropic.com"


@pytest.mark.asyncio
async def test_anthropic_live_session_chains_tool_results() -> None:
    client = _FakeAnthropicClient()
    client.messages.create.side_effect = [
        {
            "id": "msg_1",
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "lookup",
                    "input": {"query": "alpha"},
                }
            ],
        },
        {
            "id": "msg_2",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "content": [{"type": "text", "text": "final answer"}],
        },
    ]

    model = SkillAnthropicSessionLanguageModel(
        SkillAnthropicSessionLanguageModelParams(
            client=client,
            model="claude-sonnet-4-5",
        )
    )

    async def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"query": arguments["query"], "hits": 2}

    result = await model.run_live_session(
        system_prompt="sys",
        user_prompt="find alpha",
        tools=[
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        ],
        tool_registry={"lookup": lookup},
    )

    assert result.final_response == "final answer"
    assert result.turn_count == 2
    assert result.llm_input_tokens == 5
    assert result.llm_output_tokens == 3
    assert len(result.tool_executions) == 1
    assert result.tool_executions[0].name == "lookup"
    assert result.tool_executions[0].call_id == "tu_1"

    assert client.messages.create.await_count == 2
    second_call = client.messages.create.await_args_list[1]
    second_messages = second_call.kwargs["messages"]
    assert second_messages[0]["role"] == "user"
    assert second_messages[1]["role"] == "assistant"
    assert second_messages[2]["role"] == "user"
    assert second_messages[2]["content"][0]["type"] == "tool_result"
    assert second_messages[2]["content"][0]["tool_use_id"] == "tu_1"


@pytest.mark.asyncio
async def test_anthropic_live_session_raises_for_unknown_tool() -> None:
    client = _FakeAnthropicClient()
    client.messages.create.return_value = {
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "lookup",
                "input": {"query": "alpha"},
            }
        ],
    }
    model = SkillAnthropicSessionLanguageModel(
        SkillAnthropicSessionLanguageModelParams(
            client=client,
            model="claude-sonnet-4-5",
        )
    )

    with pytest.raises(SkillToolNotFoundError):
        await model.run_live_session(
            system_prompt="sys",
            user_prompt="hello",
            tools=[
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
            ],
            tool_registry={},
        )


@pytest.mark.asyncio
async def test_anthropic_live_session_respects_max_turns() -> None:
    client = _FakeAnthropicClient()
    client.messages.create.return_value = {
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "lookup",
                "input": {"query": "alpha"},
            }
        ],
    }

    model = SkillAnthropicSessionLanguageModel(
        SkillAnthropicSessionLanguageModelParams(
            client=client,
            model="claude-sonnet-4-5",
        )
    )

    async def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"query": arguments["query"], "hits": 1}

    with pytest.raises(SkillSessionLimitError):
        await model.run_live_session(
            system_prompt="sys",
            user_prompt="find alpha",
            tools=[
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
            ],
            tool_registry={"lookup": lookup},
            max_turns=1,
        )


@pytest.mark.asyncio
async def test_anthropic_live_session_emits_normalization_warnings() -> None:
    client = _FakeAnthropicClient()
    client.messages.create.return_value = {
        "usage": {},
        "content": [{"type": "text", "text": "ok"}],
    }
    model = SkillAnthropicSessionLanguageModel(
        SkillAnthropicSessionLanguageModelParams(
            client=client,
            model="claude-sonnet-4-5",
        )
    )

    result = await model.run_live_session(
        system_prompt="sys",
        user_prompt="hello",
        tools=[],
        tool_registry={},
    )

    assert result.final_response == "ok"
    assert "usage_input_tokens_missing_or_invalid" in result.normalization_warnings
    assert "usage_output_tokens_missing_or_invalid" in result.normalization_warnings


@pytest.mark.asyncio
async def test_anthropic_live_session_attaches_container_skills_when_enabled(
    tmp_path,
) -> None:
    bundle_dir = tmp_path / "retrieve-skill"
    bundle_dir.mkdir()
    (bundle_dir / "SKILL.md").write_text("# retrieve skill", encoding="utf-8")

    client = _FakeAnthropicClient()
    client.beta.skills.create.return_value = {"id": "skill_123"}
    client.beta.messages.create.return_value = {
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [{"type": "text", "text": "ok"}],
    }

    model = SkillAnthropicSessionLanguageModel(
        SkillAnthropicSessionLanguageModelParams(
            client=client,
            model="claude-sonnet-4-5",
        )
    )

    result = await model.run_live_session(
        system_prompt="sys",
        user_prompt="hello",
        tools=[],
        tool_registry={},
        provider_skill_bundles=[
            ProviderSkillBundle(
                name="retrieve-skill",
                description="Retrieve skill bundle",
                path=str(bundle_dir),
            )
        ],
    )

    assert result.final_response == "ok"
    assert client.beta.skills.create.await_count == 1
    assert client.beta.messages.create.await_count == 1
    request = client.beta.messages.create.await_args_list[0].kwargs
    assert request["container"]["skills"][0]["type"] == "custom"
    assert request["container"]["skills"][0]["skill_id"] == "skill_123"
    assert any(tool.get("type") == "code_execution_20250825" for tool in request["tools"])
