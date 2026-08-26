import inspect
import os
import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastmcp import Client
from memmachine_common.api.spec import SearchResult

from memmachine_server.main.memmachine import ALL_MEMORY_TYPES
from memmachine_server.server.api_v2.mcp import (
    MCP_SUCCESS,
    Params,
    _format_search_result_for_mcp,
    mcp_add_memory,
    mcp_delete_memory,
    mcp_search_memory,
)
from memmachine_server.server.mcp_stdio import mcp


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
    assert spec.agent_mode is False


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
    import memmachine_server.server.api_v2.mcp as mcp_module

    mcp_module.mem_machine = Mock()
    yield
    mcp_module.mem_machine = None  # cleanup


@pytest.fixture
def host_identity_env(monkeypatch):
    """Simulate an MCP host that bound identity via MM_*_ID env vars."""
    monkeypatch.setenv("MM_ORG_ID", "host_org")
    monkeypatch.setenv("MM_PROJ_ID", "host_proj")
    monkeypatch.setenv("MM_USER_ID", "host_user")


@pytest.mark.asyncio
@patch("memmachine_server.server.api_v2.mcp._add_messages_to", new_callable=AsyncMock)
async def test_add_memory_success(mock_add, host_identity_env, mcp_client):
    result = await mcp_client.call_tool(
        name="add_memory",
        arguments={"content": "hello memory"},
    )
    mock_add.assert_awaited_once()
    # The spec the host built must reflect the host-bound identity, not
    # anything LLM-supplied.
    call_kwargs = mock_add.call_args.kwargs
    spec = call_kwargs["spec"]
    assert spec.org_id == "host_org"
    assert spec.project_id == "host_proj"
    assert spec.messages[0].producer == "host_user"
    assert result.data is not None
    assert result.data.status == 200
    assert result.data.message == "Success"


@pytest.mark.asyncio
@patch("memmachine_server.server.api_v2.mcp._add_messages_to", new_callable=AsyncMock)
async def test_add_memory_failure(mock_add, host_identity_env, mcp_client):
    mock_add.side_effect = HTTPException(status_code=500, detail="DB down")

    result = await mcp_client.call_tool(
        name="add_memory",
        arguments={"content": "hello memory"},
    )
    assert result.data is not None
    assert result.data.status == 422
    assert "DB down" in result.data.message


@pytest.mark.asyncio
@patch(
    "memmachine_server.server.api_v2.mcp._search_target_memories",
    new_callable=AsyncMock,
)
async def test_search_memory_failure(mock_search, host_identity_env, mcp_client):
    mock_search.side_effect = HTTPException(status_code=422, detail="Not found")

    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={"query": "find me"},
    )
    mock_search.assert_awaited_once()
    assert result.data is not None
    assert result.data.status == 422
    assert "Not found" in result.data.message


@pytest.mark.asyncio
@patch(
    "memmachine_server.server.api_v2.mcp._search_target_memories",
    new_callable=AsyncMock,
)
async def test_search_memory_returns_string_for_empty_result(
    mock_search, host_identity_env, mcp_client
):
    """Empty results should return an empty string, never a SearchResult JSON."""
    content = {"semantic_memory": [], "episodic_memory": None}
    mock_search.return_value = SearchResult.model_validate(
        {"status": 200, "content": content}
    )
    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={"query": "find me"},
    )
    mock_search.assert_awaited_once()
    # FastMCP wraps a plain string in a structured result; assert via the
    # raw text content blocks rather than .data (which is None for str
    # returns).
    text_blocks = [block.text for block in result.content if hasattr(block, "text")]
    assert text_blocks == [""]


@pytest.mark.asyncio
@patch(
    "memmachine_server.server.api_v2.mcp._search_target_memories",
    new_callable=AsyncMock,
)
async def test_search_memory_returns_formatted_string(
    mock_search, host_identity_env, mcp_client
):
    """Populated results should render as the LLM-readable format."""
    payload = {
        "status": 200,
        "content": {
            "episodic_memory": {
                "long_term_memory": {
                    "episodes": [
                        {
                            "uid": "ep-1",
                            "content": "I like pizza",
                            "producer_id": "alice",
                            "producer_role": "user",
                            "created_at": datetime(
                                2026, 1, 5, 13, 30, tzinfo=UTC
                            ).isoformat(),
                        },
                    ],
                },
                "short_term_memory": {"episodes": [], "episode_summary": []},
            },
            "semantic_memory": [
                {
                    "category": "food",
                    "tag": "preferences",
                    "feature_name": "favorite_food",
                    "value": "pizza",
                },
            ],
        },
    }
    mock_search.return_value = SearchResult.model_validate(payload)

    result = await mcp_client.call_tool(
        name="search_memory",
        arguments={"query": "what do I like to eat"},
    )

    text_blocks = [block.text for block in result.content if hasattr(block, "text")]
    assert len(text_blocks) == 1
    rendered = text_blocks[0]
    # Episodic section: format mirrors string_from_episode_context.
    assert "[Episodic Memory]" in rendered
    assert "Monday, January 05, 2026" in rendered
    assert "alice:" in rendered
    assert '"I like pizza"' in rendered
    # Semantic section: tag → feature_name → value, no metadata IDs.
    assert "[Semantic Memory]" in rendered
    assert '"preferences"' in rendered
    assert '"favorite_food"' in rendered
    assert '"pizza"' in rendered
    # Critically: no UIDs or scores leak through.
    assert "ep-1" not in rendered
    assert "score" not in rendered


@pytest.mark.asyncio
@patch("memmachine_server.server.api_v2.mcp._delete_memories", new_callable=AsyncMock)
async def test_delete_memory_success(mock_delete, host_identity_env, mcp_client):
    result = await mcp_client.call_tool(
        name="delete_memory",
        arguments={
            "episodic_memory_uids": ["episode1"],
            "semantic_memory_uids": ["semantic1"],
        },
    )
    mock_delete.assert_awaited_once()
    spec = mock_delete.call_args.args[0]
    assert spec.org_id == "host_org"
    assert spec.project_id == "host_proj"
    assert result.data is not None
    assert result.data.status == 200
    assert result.data.message == "Success"


@pytest.mark.asyncio
@patch("memmachine_server.server.api_v2.mcp._delete_memories", new_callable=AsyncMock)
async def test_delete_memory_failure(mock_delete, host_identity_env, mcp_client):
    mock_delete.side_effect = HTTPException(status_code=500, detail="Deletion failed")

    result = await mcp_client.call_tool(
        name="delete_memory",
        arguments={
            "episodic_memory_uids": ["episode1"],
            "semantic_memory_uids": ["semantic1"],
        },
    )
    mock_delete.assert_awaited_once()
    assert result.data is not None
    assert result.data.status == 422
    assert "Deletion failed" in result.data.message


# --- New tests for the #1278 fix -----------------------------------------


def test_search_memory_default_top_k_is_5():
    """Default top_k should be 5, not 20 — addresses the context-creep bug."""
    sig = inspect.signature(mcp_search_memory)
    assert sig.parameters["top_k"].default == 5


@pytest.mark.parametrize(
    ("tool_fn", "forbidden_params"),
    [
        (mcp_add_memory, {"org_id", "proj_id", "user_id"}),
        (mcp_search_memory, {"org_id", "proj_id", "user_id"}),
        (mcp_delete_memory, {"org_id", "proj_id"}),
    ],
)
def test_mcp_tool_signatures_omit_identity_params(tool_fn, forbidden_params):
    """The LLM must not be able to supply identity as tool arguments.

    Identity is bound by the host (env vars in stdio, headers in HTTP);
    leaving these in the tool signature would re-introduce the
    LLM-spoofing surface that issue #1278 reports.
    """
    sig = inspect.signature(tool_fn)
    leaked = forbidden_params & set(sig.parameters)
    assert not leaked, f"{tool_fn.__name__} still exposes {leaked} to the LLM"


@pytest.mark.asyncio
async def test_mcp_tool_schemas_omit_identity_params(mcp_client):
    """Same as above but verified via the wire-level MCP tool schema."""
    tools = await mcp_client.list_tools()
    expected = {
        "add_memory": {"org_id", "proj_id", "user_id"},
        "search_memory": {"org_id", "proj_id", "user_id"},
        "delete_memory": {"org_id", "proj_id"},
    }
    for tool in tools:
        if tool.name not in expected:
            continue
        schema_props = set((tool.inputSchema or {}).get("properties", {}).keys())
        leaked = expected[tool.name] & schema_props
        assert not leaked, (
            f"MCP tool '{tool.name}' advertises identity params: {leaked}"
        )


@pytest.mark.asyncio
@patch("memmachine_server.server.api_v2.mcp._add_messages_to", new_callable=AsyncMock)
async def test_add_memory_rejects_llm_supplied_identity(
    mock_add, host_identity_env, mcp_client
):
    """A malicious LLM trying to pass identity must be rejected outright.

    FastMCP validates each call against the tool's input schema, which
    no longer contains identity fields. Passing them must raise an MCP
    tool-level error and never reach the underlying service.
    """
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as excinfo:
        await mcp_client.call_tool(
            name="add_memory",
            arguments={
                "content": "hello",
                "user_id": "attacker",
                "org_id": "attacker_org",
            },
        )
    # The error must cite the unexpected identity arguments so a host
    # operator can diagnose a misbehaving LLM client.
    msg = str(excinfo.value)
    assert "user_id" in msg
    assert "org_id" in msg
    # The downstream service must never have been called for the spoof
    # attempt.
    mock_add.assert_not_awaited()


# --- Direct tests for the formatter --------------------------------------


def test_format_search_result_empty():
    result = SearchResult.model_validate(
        {"status": 200, "content": {"episodic_memory": None, "semantic_memory": []}}
    )
    assert _format_search_result_for_mcp(result) == ""


def test_format_search_result_semantic_only():
    result = SearchResult.model_validate(
        {
            "status": 200,
            "content": {
                "episodic_memory": None,
                "semantic_memory": [
                    {
                        "category": "food",
                        "tag": "preferences",
                        "feature_name": "favorite_food",
                        "value": "pizza",
                    },
                ],
            },
        }
    )
    rendered = _format_search_result_for_mcp(result)
    assert rendered.startswith("[Semantic Memory]\n")
    assert '{"preferences": {"favorite_food": "pizza"}}' in rendered
    assert "[Episodic Memory]" not in rendered


def test_format_search_result_drops_uids_and_scores():
    """Regression guard for issue #1278: per-entry width must be trimmed."""
    payload = {
        "status": 200,
        "content": {
            "episodic_memory": {
                "long_term_memory": {
                    "episodes": [
                        {
                            "uid": "leak-uid",
                            "content": "secret",
                            "producer_id": "alice",
                            "producer_role": "user",
                            "score": 0.99,
                            "created_at": datetime(
                                2026, 1, 5, 13, 30, tzinfo=UTC
                            ).isoformat(),
                        }
                    ]
                },
                "short_term_memory": {"episodes": [], "episode_summary": []},
            },
            "semantic_memory": [
                {
                    "category": "food",
                    "tag": "preferences",
                    "feature_name": "favorite_food",
                    "value": "pizza",
                    "metadata": {"id": "leak-feature-id"},
                }
            ],
        },
    }
    rendered = _format_search_result_for_mcp(SearchResult.model_validate(payload))
    assert "leak-uid" not in rendered
    assert "leak-feature-id" not in rendered
    assert "0.99" not in rendered
