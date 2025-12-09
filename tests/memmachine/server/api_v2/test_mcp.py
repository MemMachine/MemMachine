import os
import re
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastmcp import Client

from memmachine.common.api.spec import SearchResult
from memmachine.main.memmachine import ALL_MEMORY_TYPES
from memmachine.server.api_v2.mcp import MCP_SUCCESS, Params
from memmachine.server.mcp_stdio import mcp

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def clear_env():
    """Automatically clear MM_USER_ID env var before and after each test."""
    old_env = os.environ.pop("MM_USER_ID", None)
    yield
    if old_env:
        os.environ["MM_USER_ID"] = old_env
    else:
        os.environ.pop("MM_USER_ID", None)


def test_user_id_without_env():
    """Should keep the provided user_id if MM_USER_ID is not set."""
    model = Params(user_id="alice")
    assert model.user_id == "alice"


def test_user_id_with_env_override(monkeypatch):
    """Should override user_id when MM_USER_ID is set in environment."""
    monkeypatch.setenv("MM_USER_ID", "env_user")
    model = Params(user_id="original_user")
    assert model.user_id == "env_user"


def test_user_id_with_empty_env(monkeypatch):
    """Should not override user_id when MM_USER_ID is empty."""
    monkeypatch.setenv("MM_USER_ID", "")
    model = Params(user_id="local_user")
    assert model.user_id == "local_user"


def assert_proj_id(proj_id: str) -> None:
    pattern = r"^mcp-user-0x\w+$"
    if not re.match(pattern, proj_id):
        raise ValueError(f"Invalid proj_id: {proj_id}")


def assert_user_id(user_id: str) -> None:
    pattern = r"^user-0x\w+$"
    if not re.match(pattern, user_id):
        raise ValueError(f"Invalid user_id: {user_id}")


def test_default_param_values():
    params = Params()
    assert params.org_id == "mcp-universal"
    assert_proj_id(params.proj_id)
    assert_user_id(params.user_id)


def test_user_id_field_filled_by_env(monkeypatch):
    """Should accept model creation with missing user_id if env var exists."""
    # Note: This depends on whether you allow missing field — Pydantic will
    # normally require user_id unless you make it Optional[str]
    monkeypatch.setenv("MM_USER_ID", "env_only")
    params = Params()
    assert params.org_id == "mcp-universal"
    assert params.proj_id == "mcp-env_only"
    assert params.user_id == "env_only"


@pytest.fixture
def params():
    return Params(user_id="usr", org_id="org", proj_id="proj")


def test_mcp_response_and_status():
    assert MCP_SUCCESS.status == 200
    assert MCP_SUCCESS.message == "Success"


def test_add_memory_param_get_new_episode(params):
    spec = params.to_add_memories_spec("Hello memory!")
    assert len(spec.messages) == 1
    message = spec.messages[0]
    assert message.timestamp is not None
    assert message.producer == "usr"
    assert message.produced_for == "unknown"
    assert message.content == "Hello memory!"
    assert message.role == "user"
    assert spec.org_id == "org"
    assert spec.project_id == "proj"


def test_search_memory_param_get_search_query(params):
    spec = params.to_search_memories_spec("hello", top_k=7)
    assert spec.org_id == "org"
    assert spec.project_id == "proj"
    assert spec.top_k == 7
    assert spec.query == "hello"
    assert spec.filter == "metadata.user_id='usr'"
    assert spec.types == ALL_MEMORY_TYPES


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as mcp_client:
        yield mcp_client


async def test_list_mcp_tools(mcp_client):
    tools = await mcp_client.list_tools()
    tool_names = [tool.name for tool in tools]
    expected_tools = [
        "add_memory",
        "search_memory",
    ]
    for expected_tool in expected_tools:
        assert expected_tool in tool_names


async def test_mcp_tool_description(mcp_client):
    tools = await mcp_client.list_tools()
    for tool in tools:
        if tool.name == "add_memory":
            assert "into memory" in tool.description
            return
    raise AssertionError


@pytest.fixture(autouse=True)
def patch_memmachine():
    import memmachine.server.api_v2.mcp as mcp_module

    mcp_module.mem_machine = Mock()
    yield
    mcp_module.mem_machine = None  # cleanup


@patch("memmachine.server.api_v2.mcp._add_messages_to", new_callable=AsyncMock)
async def test_add_memory_success(mock_add, params, mcp_client):
    result = await mcp_client.call_tool(
        name="add_memory",
        arguments={
            "content": "hello memory",
            "org_id": params.org_id,
            "proj_id": params.proj_id,
            "user_id": params.user_id,
        },
    )
    mock_add.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root.status == 200
    assert root.message == "Success"


@patch("memmachine.server.api_v2.mcp._add_messages_to", new_callable=AsyncMock)
async def test_add_memory_failure(mock_add, params, mcp_client):
    mock_add.side_effect = HTTPException(status_code=500, detail="DB down")

    result = await mcp_client.call_tool(
        name="add_memory",
        arguments={
            "content": "hello memory",
            "org_id": params.org_id,
            "proj_id": params.proj_id,
            "user_id": params.user_id,
        },
    )
    assert result.data is not None
    assert result.data.status == 422
    assert "DB down" in result.data.message


@patch("memmachine.server.api_v2.mcp._search_target_memories", new_callable=AsyncMock)
async def test_search_memory_failure(mock_search, params, mcp_client):
    mock_search.side_effect = HTTPException(status_code=422, detail="Not found")

    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={
            "org_id": params.org_id,
            "proj_id": params.proj_id,
            "user_id": params.user_id,
            "query": "find me",
        },
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root["status"] == 422
    assert "Not found" in root["message"]


@patch("memmachine.server.api_v2.mcp._search_target_memories", new_callable=AsyncMock)
async def test_search_memory_variants(mock_search, params, mcp_client):
    content = {"ep": "Memory found"}
    mock_search.return_value = SearchResult(status=200, content=content)
    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={
            "org_id": params.org_id,
            "proj_id": params.proj_id,
            "user_id": params.user_id,
            "query": "find me",
        },
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root["status"] == 200
    assert root["content"] == content
