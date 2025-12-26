"""Unit tests for Memory class (v2 API)."""

from unittest.mock import Mock

import pytest
import requests

from memmachine.common.api.spec import AddMemoryResult, SearchResult
from memmachine.common.episode_store.episode_model import EpisodeType
from memmachine.rest_client.client import MemMachineClient
from memmachine.rest_client.memory import Memory


class TestMemory:
    """Test cases for Memory class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        client = Mock(spec=MemMachineClient)
        client.base_url = "http://localhost:8080"
        client.timeout = 30
        client.request = Mock()
        return client

    def test_init_with_required_params(self, mock_client):
        """Test Memory initialization with required parameters (org_id and project_id)."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "group_id": "test_group",
                "agent_id": "test_agent",
                "user_id": "test_user",
            },
        )

        assert memory.client == mock_client
        assert memory.org_id == "test_org"
        assert memory.project_id == "test_project"
        assert memory.metadata.get("group_id") == "test_group"
        assert memory.metadata.get("agent_id") == "test_agent"
        assert memory.metadata.get("user_id") == "test_user"
        assert memory.metadata.get("session_id") is None  # Not set

    def test_init_with_only_required_params(self, mock_client):
        """Test Memory initialization with only org_id and project_id."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        assert memory.org_id == "test_org"
        assert memory.project_id == "test_project"
        assert memory.metadata.get("group_id") is None
        assert memory.metadata.get("agent_id") is None
        assert memory.metadata.get("user_id") is None
        assert memory.metadata.get("session_id") is None

    def test_init_missing_org_id_raises_error(self, mock_client):
        """Test that missing org_id raises TypeError."""
        with pytest.raises(TypeError, match=r"missing.*required.*argument.*org_id"):
            Memory(client=mock_client, project_id="test_project")

    def test_init_missing_project_id_raises_error(self, mock_client):
        """Test that missing project_id raises TypeError."""
        with pytest.raises(TypeError, match=r"missing.*required.*argument.*project_id"):
            Memory(client=mock_client, org_id="test_org")

    def test_init_with_string_ids(self, mock_client):
        """Test Memory initialization with string IDs."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        assert memory.metadata.get("agent_id") == "agent1"
        assert memory.metadata.get("user_id") == "user1"

    def test_init_with_custom_session_id(self, mock_client):
        """Test Memory initialization with custom session_id."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "agent_id": "agent1",
                "user_id": "user1",
                "session_id": "custom_session",
            },
        )

        assert memory.metadata.get("session_id") == "custom_session"

    def test_add_success(self, mock_client):
        """Test successful memory addition with v2 API format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "results": [{"uid": "memory_123"}, {"uid": "memory_456"}]
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "group_id": "test_group",
                "agent_id": "test_agent",
                "user_id": "test_user",
            },
        )

        result = memory.add("Test content")

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], AddMemoryResult)
        assert result[0].uid == "memory_123"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert len(json_data["messages"]) == 1
        assert json_data["messages"][0]["content"] == "Test content"
        assert json_data["messages"][0]["role"] == ""
        # producer and produced_for are None if not explicitly provided
        assert "producer" not in json_data["messages"][0], (
            "producer should not be in message when None (server will use default 'user')"
        )
        assert "produced_for" not in json_data["messages"][0], (
            "produced_for should not be in message when None (server will use default '')"
        )
        # timestamp is None if not explicitly provided
        assert "timestamp" not in json_data["messages"][0], (
            "timestamp should not be in message when None (server will use current time)"
        )
        assert "metadata" in json_data["messages"][0]

    def test_add_with_metadata(self, mock_client):
        """Test adding memory with metadata."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"results": [{"uid": "memory_123"}]}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        metadata = {"type": "preference", "category": "food"}
        result = memory.add("I like pizza", metadata=metadata)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], AddMemoryResult)
        assert result[0].uid == "memory_123"

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        message_metadata = json_data["messages"][0]["metadata"]
        assert message_metadata["type"] == "preference"
        assert message_metadata["category"] == "food"
        # Context fields should also be in metadata
        assert message_metadata["user_id"] == "user1"
        assert message_metadata["agent_id"] == "agent1"

    def test_add_with_role(self, mock_client):
        """Test adding memory with different roles."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"results": [{"uid": "memory_123"}]}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"user_id": "user1"},
        )

        # Test user role (default)
        memory.add("User message", role="user")
        call_args = mock_client.request.call_args
        assert call_args[1]["json"]["messages"][0]["role"] == "user"

        # Test assistant role
        memory.add("Assistant message", role="assistant")
        call_args = mock_client.request.call_args
        assert call_args[1]["json"]["messages"][0]["role"] == "assistant"

        # Test system role
        memory.add("System message", role="system")
        call_args = mock_client.request.call_args
        assert call_args[1]["json"]["messages"][0]["role"] == "system"

    def test_add_with_custom_producer(self, mock_client):
        """Test adding memory with custom producer and produced_for."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"results": [{"uid": "memory_123"}]}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        memory.add("Content", producer="user1", produced_for="agent1")

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["messages"][0]["producer"] == "user1"
        # produced_for should be a direct field in the message
        assert json_data["messages"][0]["produced_for"] == "agent1"

    def test_add_with_none_producer_and_produced_for(self, mock_client):
        """Test adding memory with None producer and produced_for."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"results": [{"uid": "memory_123"}]}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.add("Content", producer=None, produced_for=None)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], AddMemoryResult)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        # producer and produced_for should NOT be in the message when None
        assert "producer" not in json_data["messages"][0], (
            "producer should not be in message when None (server will use default 'user')"
        )
        assert "produced_for" not in json_data["messages"][0], (
            "produced_for should not be in message when None (server will use default '')"
        )

    def test_add_with_episode_type(self, mock_client):
        """Test adding memory with episode_type."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"results": [{"uid": "memory_123"}]}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"user_id": "user1"},
        )

        memory.add("Content", episode_type=EpisodeType.MESSAGE)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert (
            json_data["messages"][0]["metadata"]["episode_type"]
            == EpisodeType.MESSAGE.value
        )

    def test_add_request_exception(self, mock_client):
        """Test add raises exception on request failure."""
        mock_client.request.side_effect = requests.RequestException("Network error")

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        with pytest.raises(requests.RequestException):
            memory.add("Content")

    def test_add_http_error(self, mock_client):
        """Test add handles HTTP errors correctly."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        with pytest.raises(requests.RequestException):
            memory.add("Content")

    def test_add_client_closed(self, mock_client):
        """Test add raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(RuntimeError, match="client has been closed"):
            memory.add("Content")

    def test_search_success(self, mock_client):
        """Test successful memory search with v2 API format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": 0,
            "content": {
                "episodic_memory": [["result1", "result2"]],
                "semantic_memory": [],
            },
        }
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        results = memory.search("test query")

        assert isinstance(results, SearchResult)
        assert results.status == 0
        assert "episodic_memory" in results.content
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/search" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert json_data["query"] == "test query"
        assert json_data["top_k"] == 10
        assert "episodic" in json_data["types"]
        assert "semantic" in json_data["types"]

    def test_search_with_limit(self, mock_client):
        """Test search with limit parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 0, "content": {}}
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        memory.search("query", limit=20)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["top_k"] == 20

    def test_search_with_filters(self, mock_client):
        """Test search with filter dictionary."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 0, "content": {}}
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        filters = {"category": "work", "type": "preference"}
        memory.search("query", filter_dict=filters)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        # Filter should be SQL-like string: key='value' AND key='value'
        filter_str = json_data["filter"]
        assert "category='work'" in filter_str
        assert "type='preference'" in filter_str
        assert " AND " in filter_str

    def test_dict_to_filter_string_single_string(self, mock_client):
        """Test _dict_to_filter_string with single string value."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        filter_dict = {"category": "work"}
        filter_str = memory._dict_to_filter_string(filter_dict)
        assert filter_str == "category='work'"

    def test_dict_to_filter_string_multiple_conditions(self, mock_client):
        """Test _dict_to_filter_string with multiple conditions."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        filter_dict = {"category": "work", "type": "preference"}
        filter_str = memory._dict_to_filter_string(filter_dict)
        assert "category='work'" in filter_str
        assert "type='preference'" in filter_str
        assert " AND " in filter_str

    def test_dict_to_filter_string_with_escaped_quotes(self, mock_client):
        """Test _dict_to_filter_string with string containing single quotes."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        filter_dict = {"name": "O'Brien"}
        filter_str = memory._dict_to_filter_string(filter_dict)
        assert filter_str == "name='O''Brien'"  # SQL escape: ' -> ''

    def test_dict_to_filter_string_with_non_string_value(self, mock_client):
        """Test _dict_to_filter_string raises TypeError for non-string values."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        # Test with integer value
        with pytest.raises(TypeError, match="All filter_dict values must be strings"):
            memory._dict_to_filter_string({"rating": 5})

        # Test with None value
        with pytest.raises(TypeError, match="All filter_dict values must be strings"):
            memory._dict_to_filter_string({"deleted_at": None})

        # Test with list value
        with pytest.raises(TypeError, match="All filter_dict values must be strings"):
            memory._dict_to_filter_string({"tags": ["tag1", "tag2"]})

        # Test with boolean value
        with pytest.raises(TypeError, match="All filter_dict values must be strings"):
            memory._dict_to_filter_string({"active": True})

    def test_dict_to_filter_string_with_non_string_key(self, mock_client):
        """Test _dict_to_filter_string raises TypeError for non-string keys."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        # Test with integer key
        with pytest.raises(TypeError, match="All filter_dict keys must be strings"):
            memory._dict_to_filter_string({123: "value"})

    def test_get_default_filter_dict_with_all_fields(self, mock_client):
        """Test get_default_filter_dict with all context fields set."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "user_id": "user1",
                "agent_id": "agent1",
                "session_id": "session1",
            },
        )

        default_filters = memory.get_default_filter_dict()
        assert default_filters == {
            "metadata.user_id": "user1",
            "metadata.agent_id": "agent1",
            "metadata.session_id": "session1",
        }

    def test_get_default_filter_dict_with_partial_fields(self, mock_client):
        """Test get_default_filter_dict with only some fields set."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"user_id": "user1", "agent_id": None, "session_id": "session1"},
        )

        default_filters = memory.get_default_filter_dict()
        assert default_filters == {
            "metadata.user_id": "user1",
            "metadata.session_id": "session1",
        }
        assert "metadata.agent_id" not in default_filters

    def test_get_default_filter_dict_with_no_fields(self, mock_client):
        """Test get_default_filter_dict with no context fields set."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        default_filters = memory.get_default_filter_dict()
        assert default_filters == {}

    def test_search_with_default_filter_dict(self, mock_client):
        """Test search automatically applies built-in filters and merges with user filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 0, "content": {}}
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"user_id": "user1", "agent_id": "agent1"},
        )

        # Search with user filters only - built-in filters should be automatically merged
        user_filters = {"category": "work"}
        memory.search("query", filter_dict=user_filters)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        filter_str = json_data["filter"]

        # Should contain both built-in filters (automatically applied) and user filters
        assert "metadata.user_id='user1'" in filter_str
        assert "metadata.agent_id='agent1'" in filter_str
        assert "category='work'" in filter_str

    def test_search_automatically_applies_built_in_filters(self, mock_client):
        """Test that search automatically applies built-in filters even without user filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 0, "content": {}}
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "user_id": "user1",
                "agent_id": "agent1",
                "session_id": "session1",
            },
        )

        # Search without user filters - built-in filters should still be applied
        memory.search("query")

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        filter_str = json_data["filter"]

        # Should contain built-in filters automatically
        assert "metadata.user_id='user1'" in filter_str
        assert "metadata.agent_id='agent1'" in filter_str
        assert "metadata.session_id='session1'" in filter_str

    def test_search_user_filters_override_built_in_filters(self, mock_client):
        """Test that user-provided filters override built-in filters for the same key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 0, "content": {}}
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"user_id": "user1", "agent_id": "agent1"},
        )

        # User provides a filter that conflicts with built-in filter
        user_filters = {"metadata.user_id": "user2"}
        memory.search("query", filter_dict=user_filters)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        filter_str = json_data["filter"]

        # User-provided filter should override built-in filter
        assert "metadata.user_id='user2'" in filter_str
        assert "metadata.user_id='user1'" not in filter_str
        # But other built-in filters should still be present
        assert "metadata.agent_id='agent1'" in filter_str

    def test_get_current_metadata(self, mock_client):
        """Test get_current_metadata method returns context, filters, and filter string."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "user_id": "user1",
                "agent_id": "agent1",
                "session_id": "session1",
                "group_id": "group1",
            },
        )

        metadata = memory.get_current_metadata()

        # Check structure
        assert "context" in metadata
        assert "built_in_filters" in metadata
        assert "built_in_filter_string" in metadata

        # Check context
        context = metadata["context"]
        assert context["org_id"] == "test_org"
        assert context["project_id"] == "test_project"
        assert context["metadata"]["user_id"] == "user1"
        assert context["metadata"]["agent_id"] == "agent1"
        assert context["metadata"]["session_id"] == "session1"
        assert context["metadata"]["group_id"] == "group1"

        # Check built-in filters
        filters = metadata["built_in_filters"]
        assert filters["metadata.user_id"] == "user1"
        assert filters["metadata.agent_id"] == "agent1"
        assert filters["metadata.session_id"] == "session1"

        # Check filter string
        filter_str = metadata["built_in_filter_string"]
        assert "metadata.user_id='user1'" in filter_str
        assert "metadata.agent_id='agent1'" in filter_str
        assert "metadata.session_id='session1'" in filter_str

    def test_get_current_metadata_with_partial_context(self, mock_client):
        """Test get_current_metadata with only some context fields set."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"user_id": "user1"},
            # agent_id and session_id are None in metadata
        )

        metadata = memory.get_current_metadata()

        # Only user_id should be in built-in filters
        filters = metadata["built_in_filters"]
        assert "metadata.user_id" in filters
        assert filters["metadata.user_id"] == "user1"
        assert "metadata.agent_id" not in filters
        assert "metadata.session_id" not in filters

        # Filter string should only contain user_id
        filter_str = metadata["built_in_filter_string"]
        assert "metadata.user_id='user1'" in filter_str
        assert "metadata.agent_id" not in filter_str
        assert "metadata.session_id" not in filter_str
        assert "category='work'" not in filter_str

    def test_search_client_closed(self, mock_client):
        """Test search raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(RuntimeError, match="client has been closed"):
            memory.search("query")

    def test_get_context(self, mock_client):
        """Test getting memory context."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "group_id": "test_group",
                "agent_id": "agent1",
                "user_id": "user1",
                "session_id": "test_session",
            },
        )

        context = memory.get_context()

        assert context["org_id"] == "test_org"
        assert context["project_id"] == "test_project"
        assert context["metadata"]["group_id"] == "test_group"
        assert context["metadata"]["agent_id"] == "agent1"
        assert context["metadata"]["user_id"] == "user1"
        assert context["metadata"]["session_id"] == "test_session"

    def test_repr(self, mock_client):
        """Test string representation."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "group_id": "test_group",
                "agent_id": "agent1",
                "user_id": "user1",
                "session_id": "test_session",
            },
        )

        repr_str = repr(memory)
        assert "test_org" in repr_str
        assert "test_project" in repr_str
        assert "test_group" in repr_str
        assert "test_session" in repr_str

    def test_mark_client_closed(self, mock_client):
        """Test marking memory as closed by client."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        assert memory._client_closed is False
        memory.mark_client_closed()
        assert memory._client_closed is True

    def test_build_metadata(self, mock_client):
        """Test that _build_metadata includes context fields."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={
                "group_id": "test_group",
                "agent_id": "agent1",
                "user_id": "user1",
                "session_id": "test_session",
            },
        )

        metadata = memory._build_metadata({"custom": "value"})

        assert metadata["custom"] == "value"
        assert metadata["group_id"] == "test_group"
        assert metadata["user_id"] == "user1"
        assert metadata["agent_id"] == "agent1"
        assert metadata["session_id"] == "test_session"

    def test_build_metadata_with_string_ids(self, mock_client):
        """Test that _build_metadata handles string IDs correctly."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            metadata={"agent_id": "agent1", "user_id": "user1"},
        )

        metadata = memory._build_metadata({})

        # Should store as strings
        assert metadata["agent_id"] == "agent1"
        assert metadata["user_id"] == "user1"

    # Delete episodic method tests
    def test_delete_episodic_success(self, mock_client):
        """Test successful deletion of episodic memory."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        episodic_id = "episode_123"
        result = memory.delete_episodic(episodic_id=episodic_id)

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/episodic/delete" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert json_data["episodic_id"] == episodic_id

    def test_delete_episodic_with_timeout(self, mock_client):
        """Test delete_episodic with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.delete_episodic(episodic_id="episode_123", timeout=60)

        assert result is True
        call_args = mock_client.request.call_args
        assert call_args[1]["timeout"] == 60

    def test_delete_episodic_http_error(self, mock_client):
        """Test delete_episodic handles HTTP errors correctly."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.delete_episodic(episodic_id="episode_123")

    def test_delete_episodic_client_closed(self, mock_client):
        """Test delete_episodic raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot delete episodic memory: client has been closed"
        ):
            memory.delete_episodic(episodic_id="episode_123")

    def test_delete_semantic_success(self, mock_client):
        """Test successful deletion of semantic memory."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        semantic_id = "feature_456"
        result = memory.delete_semantic(semantic_id=semantic_id)

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/delete" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert json_data["semantic_id"] == semantic_id

    def test_delete_semantic_with_timeout(self, mock_client):
        """Test delete_semantic with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.delete_semantic(semantic_id="feature_456", timeout=60)

        assert result is True
        call_args = mock_client.request.call_args
        assert call_args[1]["timeout"] == 60

    def test_delete_semantic_http_error(self, mock_client):
        """Test delete_semantic handles HTTP errors correctly."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.delete_semantic(semantic_id="feature_456")

    def test_delete_semantic_client_closed(self, mock_client):
        """Test delete_semantic raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot delete semantic memory: client has been closed"
        ):
            memory.delete_semantic(semantic_id="feature_456")
