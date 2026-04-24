"""
Tests for the MemMachine memory tool.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from strands.types.tools import ToolUse

from strands_memmachine import memmachine_memory
from strands_memmachine.tool import MemMachineClient


@pytest.fixture
def mock_tool():
    """Return a mock ToolUse object with default empty input."""
    mock = MagicMock(spec=ToolUse)
    mock.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {}}.get(key, default)
    return mock


@pytest.fixture
def mock_client():
    """Return a mock MemMachineClient instance."""
    return MagicMock(spec=MemMachineClient)


# ---------------------------------------------------------------------------
# Memory Tests
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_store_memory(mock_client_class, mock_tool):
    """Verify that a store action returns success and includes the stored UID."""
    mock_client_class.return_value.store_memory.return_value = {"results": [{"uid": "mem-123"}]}
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "store",
            "content": "User prefers aisle seats on flights",
            "metadata": {"category": "travel"},
        },
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert result_data["results"][0]["uid"] == "mem-123"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_store_memory_with_options(mock_client_class, mock_tool):
    """Verify that store passes producer, produced_for, types, and metadata to the client."""
    mock_client_class.return_value.store_memory.return_value = {"results": [{"uid": "mem-456"}]}
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "store",
            "content": "Meeting at 10 AM tomorrow",
            "producer": "assistant",
            "produced_for": "alice",
            "types": ["episodic"],
            "metadata": {"type": "reminder", "priority": "high"},
        },
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.store_memory.assert_called_once_with(
        content="Meeting at 10 AM tomorrow",
        types=["episodic"],
        producer="assistant",
        produced_for="alice",
        metadata={"type": "reminder", "priority": "high"},
    )


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_search_memories(mock_client_class, mock_tool):
    """Verify that a search action returns success and episodic content."""
    mock_client_class.return_value.search_memories.return_value = {
        "status": 0,
        "content": {
            "episodic_memory": {
                "long_term_memory": {
                    "episodes": [
                        {
                            "content": "User prefers aisle seats on flights",
                            "score": 0.95,
                            "created_at": "2024-03-20T10:00:00Z",
                        }
                    ]
                }
            }
        },
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search", "query": "flight preferences", "top_k": 5},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert "episodic_memory" in result_data["content"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_search_memories_with_filter(mock_client_class, mock_tool):
    """Verify that search passes types and filter_str correctly to the client."""
    mock_client_class.return_value.search_memories.return_value = {
        "status": 0,
        "content": {"semantic_memory": {"memories": [{"content": "Prefers Python", "score": 0.88}]}},
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search",
            "query": "programming preferences",
            "types": ["semantic"],
            "filter": "metadata.user_id=alice",
        },
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.search_memories.assert_called_once_with(
        query="programming preferences",
        top_k=10,
        types=["semantic"],
        filter_str="metadata.user_id=alice",
    )


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_search_memories_empty_results(mock_client_class, mock_tool):
    """Verify that a search returning empty content still reports success."""
    mock_client_class.return_value.search_memories.return_value = {"status": 0, "content": {}}
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search", "query": "nonexistent topic"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_list_memories(mock_client_class, mock_tool):
    """Verify that a list action returns success and the content block."""
    mock_client_class.return_value.list_memories.return_value = {
        "status": 0,
        "content": {
            "episodic_memory": [
                {
                    "uid": "mem-123",
                    "content": "User prefers aisle seats",
                    "created_at": "2024-03-20T10:00:00Z",
                    "metadata": {"category": "travel"},
                }
            ]
        },
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list", "page_size": 50},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert "content" in result_data


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_list_memories_with_filter_and_type(mock_client_class, mock_tool):
    """Verify that list passes memory_type, filter_str, page_size, and page_num to the client."""
    mock_client_class.return_value.list_memories.return_value = {"status": 0, "content": {}}
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "list",
            "memory_type": "episodic",
            "filter": "metadata.user_id=alice",
            "page_size": 25,
            "page_num": 2,
        },
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.list_memories.assert_called_once_with(
        page_size=25,
        page_num=2,
        memory_type="episodic",
        filter_str="metadata.user_id=alice",
    )


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_delete_episodic_memory(mock_client_class, mock_tool):
    """Verify that episodic delete calls the correct client method with the given UID."""
    mock_client_class.return_value.delete_episodic_memory.return_value = None
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_type": "episodic", "memory_id": "mem-123"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    assert "mem-123" in result["content"][0]["text"]
    mock_client_class.return_value.delete_episodic_memory.assert_called_once_with(memory_id="mem-123", memory_ids=None)


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_delete_semantic_memory(mock_client_class, mock_tool):
    """Verify that semantic delete calls the correct client method with the given UID."""
    mock_client_class.return_value.delete_semantic_memory.return_value = None
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_type": "semantic", "memory_id": "sem-456"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    assert "sem-456" in result["content"][0]["text"]
    mock_client_class.return_value.delete_semantic_memory.assert_called_once_with(memory_id="sem-456", memory_ids=None)


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_bulk_delete_episodic(mock_client_class, mock_tool):
    """Verify that bulk episodic delete passes a list of UIDs to the client."""
    mock_client_class.return_value.delete_episodic_memory.return_value = None
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "delete",
            "memory_type": "episodic",
            "memory_ids": ["mem-1", "mem-2", "mem-3"],
        },
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.delete_episodic_memory.assert_called_once_with(
        memory_id=None, memory_ids=["mem-1", "mem-2", "mem-3"]
    )


# ---------------------------------------------------------------------------
# Project Tests
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_create_project(mock_client_class, mock_tool):
    """Verify that create_project returns success with the project record."""
    mock_client_class.return_value.create_project.return_value = {
        "project_id": "my-project",
        "description": "",
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "create_project", "project_id": "my-project"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.create_project.assert_called_once_with(project_id="my-project", description="")


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_create_project_with_description(mock_client_class, mock_tool):
    """Verify that create_project passes description correctly to the client."""
    mock_client_class.return_value.create_project.return_value = {
        "project_id": "my-project",
        "description": "Travel preferences project",
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "create_project",
            "project_id": "my-project",
            "description": "Travel preferences project",
        },
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.create_project.assert_called_once_with(
        project_id="my-project", description="Travel preferences project"
    )


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_project_id_for_create(mock_client_class, mock_tool):
    """Verify that omitting project_id for create_project returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "create_project"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "project_id is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_delete_project(mock_client_class, mock_tool):
    """Verify that delete_project calls the correct client method."""
    mock_client_class.return_value.delete_project.return_value = {}
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete_project", "project_id": "my-project"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.delete_project.assert_called_once_with(project_id="my-project")


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_project_id_for_delete_project(mock_client_class, mock_tool):
    """Verify that omitting project_id for delete_project returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete_project"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "project_id is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_get_project(mock_client_class, mock_tool):
    """Verify that get_project returns the project record."""
    mock_client_class.return_value.get_project.return_value = {
        "project_id": "my-project",
        "description": "Travel preferences project",
        "config": {"embedder": "bge-base-en"},
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get_project", "project_id": "my-project"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.get_project.assert_called_once_with(project_id="my-project")


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_project_id_for_get_project(mock_client_class, mock_tool):
    """Verify that omitting project_id for get_project returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get_project"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "project_id is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_list_projects(mock_client_class, mock_tool):
    """Verify that list_projects returns the list of projects."""
    mock_client_class.return_value.list_projects.return_value = [
        {"project_id": "project-1", "description": "First project"},
        {"project_id": "project-2", "description": "Second project"},
    ]
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list_projects"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.list_projects.assert_called_once()


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_get_episode_count(mock_client_class, mock_tool):
    """Verify that get_episode_count returns the episode count for a project."""
    mock_client_class.return_value.get_episode_count.return_value = {"count": 42}
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get_episode_count", "project_id": "my-project"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client_class.return_value.get_episode_count.assert_called_once_with(project_id="my-project")


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_project_id_for_episode_count(mock_client_class, mock_tool):
    """Verify that omitting project_id for get_episode_count returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get_episode_count"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "project_id is required" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# General / Error Tests
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_invalid_action(mock_client_class, mock_tool):
    """Verify that an unrecognised action returns an error result."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "invalid"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "Invalid action" in result["content"][0]["text"]


def test_missing_api_key(mock_tool, monkeypatch):
    """Verify that a missing API key returns an error referencing MEMMACHINE_API_KEY."""
    monkeypatch.delenv("MEMMACHINE_API_KEY", raising=False)
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "MEMMACHINE_API_KEY" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_content_for_store(mock_client_class, mock_tool):
    """Verify that omitting content for store returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "store"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "content is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_query_for_search(mock_client_class, mock_tool):
    """Verify that omitting query for search returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "query is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_memory_type_for_delete(mock_client_class, mock_tool):
    """Verify that omitting memory_type for delete returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_id": "mem-123"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "memory_type is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_missing_memory_id_for_delete(mock_client_class, mock_tool):
    """Verify that omitting both memory_id and memory_ids for delete returns an error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_type": "episodic"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "memory_id or memory_ids is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_invalid_memory_type_for_delete(mock_client_class, mock_tool):
    """Verify that an invalid memory_type for delete returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_type": "invalid_type", "memory_id": "mem-123"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "Invalid memory_type" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
def test_client_initialization():
    """Verify that the client initialises with default base URL."""
    client = MemMachineClient()
    assert client.api_key == "test-api-key"
    assert client.base_url == "https://api.memmachine.ai"


@patch.dict(
    os.environ,
    {"MEMMACHINE_API_KEY": "test-api-key", "MEMMACHINE_BASE_URL": "http://localhost:8080"},
)
def test_client_custom_base_url():
    """Verify that a custom base URL is accepted."""
    client = MemMachineClient()
    assert client.base_url == "http://localhost:8080"


@patch.dict(
    os.environ,
    {"MEMMACHINE_API_KEY": "test-api-key", "MEMMACHINE_BASE_URL": "http://localhost:8080/"},
)
def test_client_base_url_trailing_slash():
    """Verify that a trailing slash is stripped from the base URL."""
    client = MemMachineClient()
    assert client.base_url == "http://localhost:8080"


def test_client_missing_api_key(monkeypatch):
    """Verify that initialising the client without an API key raises ValueError."""
    monkeypatch.delenv("MEMMACHINE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MEMMACHINE_API_KEY"):
        MemMachineClient()


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_api_error_handling(mock_client_class, mock_tool):
    """Verify that an API exception is caught and returned as an error result."""
    mock_client_class.return_value.search_memories.side_effect = Exception("API connection failed")
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search", "query": "test query"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "API connection failed" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_store_empty_results(mock_client_class, mock_tool):
    """Verify that a store response with an empty results list still returns success."""
    mock_client_class.return_value.store_memory.return_value = {"results": []}
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "store", "content": "Test content"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_search_with_flat_episodic_list(mock_client_class, mock_tool):
    """Verify that search handles a flat episodic memory list."""
    mock_client_class.return_value.search_memories.return_value = {
        "status": 0,
        "content": {
            "episodic_memory": [
                {"content": "Memory 1", "score": 0.9},
                {"content": "Memory 2", "score": 0.7},
            ]
        },
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search", "query": "test"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_memmachine.tool.MemMachineClient")
def test_search_with_semantic_memories(mock_client_class, mock_tool):
    """Verify that search correctly returns semantic memory results."""
    mock_client_class.return_value.search_memories.return_value = {
        "status": 0,
        "content": {"semantic_memory": {"memories": [{"content": "User likes Python", "score": 0.92}]}},
    }
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search", "query": "programming"},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
def test_missing_action(mock_tool):
    """Verify that omitting the action parameter returns a descriptive error."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {},
    }.get(key, default)

    result = memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "action parameter is required" in result["content"][0]["text"]
