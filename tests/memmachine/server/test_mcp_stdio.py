import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastmcp import Client
from pydantic import ValidationError

from memmachine.server.app import (
    AddMemoryParam,
    NewEpisode,
    SearchMemoryParam,
    SearchResult,
    UserIDWithEnv,
    mcpSuccess,
)
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
    model = UserIDWithEnv(user_id="alice")
    assert model.user_id == "alice"


def test_user_id_with_env_override(monkeypatch):
    """Should override user_id when MM_USER_ID is set in environment."""
    monkeypatch.setenv("MM_USER_ID", "env_user")
    model = UserIDWithEnv(user_id="original_user")
    assert model.user_id == "env_user"


def test_user_id_with_empty_env(monkeypatch):
    """Should not override user_id when MM_USER_ID is empty."""
    monkeypatch.setenv("MM_USER_ID", "")
    model = UserIDWithEnv(user_id="local_user")
    assert model.user_id == "local_user"


def test_user_id_field_required():
    """Should raise ValidationError if user_id is missing and no env var set."""
    with pytest.raises(ValidationError):
        UserIDWithEnv()  # type: ignore


def test_user_id_field_filled_by_env(monkeypatch):
    """Should accept model creation with missing user_id if env var exists."""
    # Note: This depends on whether you allow missing field — Pydantic will
    # normally require user_id unless you make it Optional[str]
    monkeypatch.setenv("MM_USER_ID", "env_only")
    # If user_id is required, this will still raise ValidationError
    # but you can change your model to `user_id: Optional[str] = None`
    # to make this test pass
    with pytest.raises(ValidationError):
        UserIDWithEnv()  # type: ignore


@pytest.fixture
def add_param():
    return AddMemoryParam(user_id="u1", content="Hello memory!")


@pytest.fixture
def search_param():
    return SearchMemoryParam(user_id="u1", query="find", limit=5)


def test_mcp_response_and_status():
    assert mcpSuccess.status == 200
    assert mcpSuccess.message == "Success"


def test_add_memory_param_get_new_episode(add_param):
    ep = add_param.get_new_episode()
    assert ep.episode_content == "Hello memory!"
    assert ep.session.user_id == ["u1"]


def test_search_memory_param_get_search_query(search_param):
    q = search_param.get_search_query()
    assert q.query == "find"
    assert q.limit == 5
    assert q.session.user_id == ["u1"]


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
    assert False


@patch("memmachine.server.app._add_memory", new_callable=AsyncMock)
async def test_add_memory_success(mock_add, add_param, mcp_client):
    result = await mcp_client.call_tool(
        name="add_memory", arguments={"param": add_param}
    )
    mock_add.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root.status == 200
    assert root.message == "Success"


@patch("memmachine.server.app._add_memory", new_callable=AsyncMock)
async def test_add_memory_failure(mock_add, add_param, mcp_client):
    mock_add.side_effect = HTTPException(status_code=500, detail="DB down")

    # Patch log_error_with_session on NewEpisode
    with patch.object(NewEpisode, "log_error_with_session") as mock_log:
        result = await mcp_client.call_tool(
            name="add_memory", arguments={"param": add_param}
        )
        mock_log.assert_called_once()
        assert result.data is not None
        assert result.data.status == 500
        assert "DB down" in result.data.message


@patch("memmachine.server.app._search_memory", new_callable=AsyncMock)
async def test_search_memory_failure(mock_search, search_param, mcp_client):
    mock_search.side_effect = HTTPException(status_code=404, detail="Not found")

    result = await mcp_client.call_tool(
        name="search_memory", arguments={"param": search_param}
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root["status"] == 404
    assert "Not found" in root["message"]


@patch("memmachine.server.app._search_memory", new_callable=AsyncMock)
async def test_search_memory_variants(mock_search, search_param, mcp_client):
    content = {"ep": "Memory found"}
    mock_search.return_value = SearchResult(status=200, content=content)
    result = await mcp_client.call_tool(
        name="search_memory", arguments={"param": search_param}
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root["status"] == 200
    assert root["content"] == content


# === Flat Parameter Style Tests ===


@patch("memmachine.server.app._add_memory", new_callable=AsyncMock)
async def test_add_memory_flat_success(mock_add, mcp_client):
    """Test add_memory with flat parameters (user_id, content)."""
    result = await mcp_client.call_tool(
        name="add_memory",
        arguments={"user_id": "u1", "content": "Hello memory!"},
    )
    mock_add.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root.status == 200
    assert root.message == "Success"


@patch("memmachine.server.app._add_memory", new_callable=AsyncMock)
async def test_add_memory_flat_failure(mock_add, mcp_client):
    """Test add_memory with flat parameters when operation fails."""
    mock_add.side_effect = HTTPException(status_code=500, detail="DB down")

    # Patch log_error_with_session on NewEpisode
    with patch.object(NewEpisode, "log_error_with_session") as mock_log:
        result = await mcp_client.call_tool(
            name="add_memory",
            arguments={"user_id": "u1", "content": "Hello memory!"},
        )
        mock_log.assert_called_once()
        assert result.data is not None
        assert result.data.status == 500
        assert "DB down" in result.data.message


async def test_add_memory_flat_missing_params(mcp_client):
    """Test add_memory with flat parameters when required params are missing."""
    result = await mcp_client.call_tool(
        name="add_memory",
        arguments={"user_id": "u1"},  # missing content
    )
    assert result.data is not None
    assert result.data.status == 400
    assert "user_id and content must be provided" in result.data.message


@patch("memmachine.server.app._search_memory", new_callable=AsyncMock)
async def test_search_memory_flat_success(mock_search, mcp_client):
    """Test search_memory with flat parameters (user_id, query, limit)."""
    content = {"ep": "Memory found"}
    mock_search.return_value = SearchResult(status=200, content=content)
    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={"user_id": "u1", "query": "find", "limit": 5},
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root["status"] == 200
    assert root["content"] == content


@patch("memmachine.server.app._search_memory", new_callable=AsyncMock)
async def test_search_memory_flat_failure(mock_search, mcp_client):
    """Test search_memory with flat parameters when operation fails."""
    mock_search.side_effect = HTTPException(status_code=404, detail="Not found")

    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={"user_id": "u1", "query": "find", "limit": 5},
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root["status"] == 404
    assert "Not found" in root["message"]


@patch("memmachine.server.app._search_memory", new_callable=AsyncMock)
async def test_search_memory_flat_default_limit(mock_search, mcp_client):
    """Test search_memory with flat parameters using default limit."""
    content = {"ep": "Memory found"}
    mock_search.return_value = SearchResult(status=200, content=content)
    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={"user_id": "u1", "query": "find"},  # limit defaults to 5
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    root = result.data
    assert root["status"] == 200
    assert root["content"] == content


async def test_search_memory_flat_missing_params(mcp_client):
    """Test search_memory with flat parameters when required params are missing."""
    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={"user_id": "u1"},  # missing query
    )
    assert result.data is not None
    assert result.data["status"] == 400
    assert "user_id and query must be provided" in result.data["message"]
