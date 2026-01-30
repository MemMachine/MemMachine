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


@pytest.fixture(autouse=True)
def clear_env():
    """Automatically clear env vars before and after each test."""
    old_org_id = os.getenv("MM_ORG_ID")
    old_proj_id = os.getenv("MM_PROJ_ID")
    old_user_id = os.getenv("MM_USER_ID")
    yield
    if old_user_id:
        os.environ["MM_USER_ID"] = old_user_id
    else:
        os.environ.pop("MM_USER_ID", None)
    if old_org_id:
        os.environ["MM_ORG_ID"] = old_org_id
    else:
        os.environ.pop("MM_ORG_ID", None)
    if old_proj_id:
        os.environ["MM_PROJ_ID"] = old_proj_id
    else:
        os.environ.pop("MM_PROJ_ID", None)


def test_user_id_without_env():
    """Should keep the provided user_id if MM_USER_ID is not set."""
    model = Params(user_id="alice")
    assert model.user_id == "alice"


def test_user_id_with_env_override(monkeypatch):
    """Should override user_id when MM_USER_ID is set in environment."""
    monkeypatch.setenv("MM_USER_ID", "env_user")
    model = Params(user_id="original_user")
    assert model.user_id == "env_user"


def test_org_id_with_env_override(monkeypatch):
    """Should override org_id when MM_ORG_ID is set in environment."""
    monkeypatch.setenv("MM_ORG_ID", "env_org")
    model = Params(org_id="original_org", user_id="user")
    assert model.org_id == "env_org"


def test_proj_id_with_env_override(monkeypatch):
    """Should override proj_id when MM_PROJ_ID is set in environment."""
    monkeypatch.setenv("MM_PROJ_ID", "env_proj")
    model = Params(proj_id="original_proj", user_id="user")
    assert model.proj_id == "env_proj"


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
    assert spec.filter == ""
    assert spec.types == ALL_MEMORY_TYPES


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as mcp_client:
        yield mcp_client


@pytest.mark.asyncio
async def test_list_mcp_tools(mcp_client):
    tools = await mcp_client.list_tools()
    tool_names = [tool.name for tool in tools]
    expected_tools = [
        "add_memory",
        "search_memory",
    ]
    for expected_tool in expected_tools:
        assert expected_tool in tool_names


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
@patch("memmachine.server.api_v2.mcp._search_target_memories", new_callable=AsyncMock)
async def test_search_memory_variants(mock_search, params, mcp_client):
    content = {"semantic_memory": [], "episodic_memory": None}
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


@pytest.mark.asyncio
@patch("memmachine.server.api_v2.mcp._delete_memories", new_callable=AsyncMock)
async def test_delete_memory_success(mock_delete, params, mcp_client):
    result = await mcp_client.call_tool(
        name="delete_memory",
        arguments={
            "org_id": params.org_id,
            "proj_id": params.proj_id,
            "episodic_memory_uids": ["episode1"],
            "semantic_memory_uids": ["semantic1"],
        },
    )
    mock_delete.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root.status == 200
    assert root.message == "Success"


@pytest.mark.asyncio
@patch("memmachine.server.api_v2.mcp._delete_memories", new_callable=AsyncMock)
async def test_delete_memory_failure(mock_delete, params, mcp_client):
    mock_delete.side_effect = HTTPException(status_code=500, detail="Deletion failed")

    result = await mcp_client.call_tool(
        name="delete_memory",
        arguments={
            "org_id": params.org_id,
            "proj_id": params.proj_id,
            "episodic_memory_uids": ["episode1"],
            "semantic_memory_uids": ["semantic1"],
        },
    )
    mock_delete.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root.status == 422
    assert "Deletion failed" in root.message
