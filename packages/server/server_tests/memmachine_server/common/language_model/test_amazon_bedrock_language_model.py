"""Unit tests for AmazonBedrockLanguageModel."""

import json
import threading
from typing import Any

import boto3
import pytest
from pydantic import BaseModel

from memmachine_server.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockLanguageModel,
    AmazonBedrockLanguageModelParams,
)


class Capital(BaseModel):
    country: str
    capital: str


def make_model_with_stubbed_converse() -> tuple[AmazonBedrockLanguageModel, dict]:
    """Build a model whose bedrock client answers a valid tool use locally."""
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    calls: dict[str, Any] = {"threads": []}
    payload = Capital(country="France", capital="Paris").model_dump()

    def stub_converse(**kwargs: Any) -> dict[str, Any]:
        calls["threads"].append(threading.get_ident())
        tools = (kwargs.get("toolConfig") or {}).get("tools") or []
        if tools:
            name = tools[0]["toolSpec"]["name"]
            content = [
                {"toolUse": {"toolUseId": "call-1", "name": name, "input": payload}}
            ]
            stop_reason = "tool_use"
        else:
            content = [{"text": json.dumps(payload)}]
            stop_reason = "end_turn"
        return {
            "output": {"message": {"role": "ASSISTANT", "content": content}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

    client.converse = stub_converse
    model = AmazonBedrockLanguageModel(
        AmazonBedrockLanguageModelParams(client=client, model_id="stub-model")
    )
    return model, calls


@pytest.mark.asyncio
async def test_generate_parsed_response_parses_the_tool_use():
    model, _ = make_model_with_stubbed_converse()

    response = await model.generate_parsed_response(
        output_format=Capital,
        user_prompt="Capital of France?",
    )

    assert isinstance(response, Capital)
    assert response.country == "France"
    assert response.capital == "Paris"


@pytest.mark.asyncio
async def test_generate_parsed_response_runs_converse_off_the_event_loop():
    model, calls = make_model_with_stubbed_converse()

    await model.generate_parsed_response(
        output_format=Capital,
        user_prompt="Capital of France?",
    )

    assert calls["threads"], "converse was never called"
    assert all(thread != threading.get_ident() for thread in calls["threads"]), (
        "converse ran on the event loop's thread, blocking it for the whole call"
    )
