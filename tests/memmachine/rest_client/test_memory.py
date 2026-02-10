"""Unit tests for Memory class (v2 API)."""

from typing import cast
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
            Memory(client=mock_client, project_id="test_project")  # type: ignore[call-arg]

    def test_init_missing_project_id_raises_error(self, mock_client):
        """Test that missing project_id raises TypeError."""
        with pytest.raises(TypeError, match=r"missing.*required.*argument.*project_id"):
            Memory(client=mock_client, org_id="test_org")  # type: ignore[call-arg]

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
                "episodic_memory": {
                    "long_term_memory": {"episodes": []},
                    "short_term_memory": {
                        "episodes": [],
                        "episode_summary": [],
                    },
                },
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
        assert results.content.episodic_memory is not None
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/search" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert json_data["query"] == "test query"
        assert json_data["top_k"] == 10
        assert json_data["agent_mode"] is False
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

    def test_search_with_agent_mode(self, mock_client):
        """Test search with agent_mode enabled."""
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

        memory.search("query", agent_mode=True)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["agent_mode"] is True

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
            memory._dict_to_filter_string(cast(dict[str, str], {"rating": 5}))

        # Test with None value
        with pytest.raises(TypeError, match="All filter_dict values must be strings"):
            memory._dict_to_filter_string(cast(dict[str, str], {"deleted_at": None}))

        # Test with list value
        with pytest.raises(TypeError, match="All filter_dict values must be strings"):
            memory._dict_to_filter_string(
                cast(dict[str, str], {"tags": ["tag1", "tag2"]})
            )

        # Test with boolean value
        with pytest.raises(TypeError, match="All filter_dict values must be strings"):
            memory._dict_to_filter_string(cast(dict[str, str], {"active": True}))

    def test_dict_to_filter_string_with_non_string_key(self, mock_client):
        """Test _dict_to_filter_string raises TypeError for non-string keys."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        # Test with integer key
        with pytest.raises(TypeError, match="All filter_dict keys must be strings"):
            memory._dict_to_filter_string(cast(dict[str, str], {123: "value"}))

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
            metadata=cast(
                dict[str, str],
                {"user_id": "user1", "agent_id": None, "session_id": "session1"},
            ),
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

    def test_search_with_set_metadata(self, mock_client):
        """set_metadata is included in the JSON payload sent to /memories/search."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 0, "content": {}}
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        memory.search("query", set_metadata={"user_id": "user123", "tenant": "acme"})

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["set_metadata"] == {"user_id": "user123", "tenant": "acme"}

    def test_search_without_set_metadata_excludes_key(self, mock_client):
        """When set_metadata is None, it is excluded from the request payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 0, "content": {}}
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        memory.search("query")

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert "set_metadata" not in json_data

    def test_list_with_set_metadata(self, mock_client):
        """set_metadata is included in the JSON payload sent to /memories/list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": 0,
            "content": {"episodic_memory": [], "semantic_memory": []},
        }
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        memory.list(set_metadata={"user_id": "user456", "region": "us-east"})

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["set_metadata"] == {"user_id": "user456", "region": "us-east"}

    def test_list_without_set_metadata_excludes_key(self, mock_client):
        """When set_metadata is None, it is excluded from the list request payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": 0,
            "content": {"episodic_memory": [], "semantic_memory": []},
        }
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        memory.list()

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert "set_metadata" not in json_data

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

    # Feature management tests
    def test_add_feature_success(self, mock_client):
        """Test successful feature addition."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"feature_id": "feature_123"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.add_feature(
            set_id="test_set",
            category_name="preferences",
            tag="food",
            feature="favorite_food",
            value="pizza",
        )

        assert result == "feature_123"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/feature" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert json_data["set_id"] == "test_set"
        assert json_data["category_name"] == "preferences"
        assert json_data["tag"] == "food"
        assert json_data["feature"] == "favorite_food"
        assert json_data["value"] == "pizza"

    def test_add_feature_with_metadata_and_citations(self, mock_client):
        """Test feature addition with metadata and citations."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"feature_id": "feature_456"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.add_feature(
            set_id="test_set",
            category_name="preferences",
            tag="food",
            feature="favorite_food",
            value="pizza",
            feature_metadata={"source": "conversation"},
            citations=["ep1", "ep2"],
        )

        assert result == "feature_456"
        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["feature_metadata"] == {"source": "conversation"}
        assert json_data["citations"] == ["ep1", "ep2"]

    def test_add_feature_http_error(self, mock_client):
        """Test add_feature handles HTTP errors correctly."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.add_feature(
                set_id="test_set",
                category_name="preferences",
                tag="food",
                feature="favorite_food",
                value="pizza",
            )

    def test_add_feature_client_closed(self, mock_client):
        """Test add_feature raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot add feature: client has been closed"
        ):
            memory.add_feature(
                set_id="test_set",
                category_name="preferences",
                tag="food",
                feature="favorite_food",
                value="pizza",
            )

    def test_get_feature_success(self, mock_client):
        """Test successful feature retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "set_id": "test_set",
            "category": "preferences",
            "tag": "food",
            "feature_name": "favorite_food",
            "value": "pizza",
            "metadata": {"id": "feature_123", "citations": None, "other": None},
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_feature(feature_id="feature_123")

        assert result is not None
        assert result.set_id == "test_set"
        assert result.category == "preferences"
        assert result.tag == "food"
        assert result.feature_name == "favorite_food"
        assert result.value == "pizza"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/feature/get" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["feature_id"] == "feature_123"
        assert json_data["load_citations"] is False

    def test_get_feature_with_citations(self, mock_client):
        """Test feature retrieval with citations."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "set_id": "test_set",
            "category": "preferences",
            "tag": "food",
            "feature_name": "favorite_food",
            "value": "pizza",
            "metadata": {
                "id": "feature_123",
                "citations": ["ep1", "ep2"],
                "other": None,
            },
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_feature(feature_id="feature_123", load_citations=True)

        assert result is not None
        assert result.metadata.citations == ["ep1", "ep2"]
        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["load_citations"] is True

    def test_get_feature_not_found(self, mock_client):
        """Test get_feature returns None when feature not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_feature(feature_id="nonexistent")
        assert result is None

    def test_get_feature_http_error_not_found(self, mock_client):
        """Test get_feature returns None when HTTPError with 404."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_feature(feature_id="nonexistent")
        assert result is None

    def test_get_feature_client_closed(self, mock_client):
        """Test get_feature raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot get feature: client has been closed"
        ):
            memory.get_feature(feature_id="feature_123")

    def test_update_feature_success(self, mock_client):
        """Test successful feature update."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.update_feature(feature_id="feature_123", value="sushi")

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/feature/update" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["feature_id"] == "feature_123"
        assert json_data["value"] == "sushi"
        # None values should be excluded
        assert "category_name" not in json_data
        assert "feature" not in json_data
        assert "tag" not in json_data
        assert "metadata" not in json_data

    def test_update_feature_all_fields(self, mock_client):
        """Test feature update with all fields."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.update_feature(
            feature_id="feature_123",
            category_name="new_preferences",
            tag="cuisine",
            feature="top_food",
            value="sushi",
            metadata={"updated": "true"},
        )

        assert result is True
        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["category_name"] == "new_preferences"
        assert json_data["tag"] == "cuisine"
        assert json_data["feature"] == "top_food"
        assert json_data["value"] == "sushi"
        assert json_data["metadata"] == {"updated": "true"}

    def test_update_feature_http_error(self, mock_client):
        """Test update_feature handles HTTP errors correctly."""
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
            memory.update_feature(feature_id="feature_123", value="sushi")

    def test_update_feature_client_closed(self, mock_client):
        """Test update_feature raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot update feature: client has been closed"
        ):
            memory.update_feature(feature_id="feature_123", value="sushi")

    # --- Semantic Set Type SDK Tests ---

    def test_create_semantic_set_type_success(self, mock_client):
        """Test successful semantic set type creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"set_type_id": "set_type_123"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.create_semantic_set_type(
            metadata_tags=["user_id", "session_id"],
            is_org_level=False,
            name="User Sessions",
            description="Set type for user sessions",
        )

        assert result == "set_type_123"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/set_type" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert json_data["metadata_tags"] == ["user_id", "session_id"]
        assert json_data["is_org_level"] is False
        assert json_data["name"] == "User Sessions"
        assert json_data["description"] == "Set type for user sessions"

    def test_create_semantic_set_type_minimal(self, mock_client):
        """Test set type creation with minimal parameters."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"set_type_id": "set_type_456"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.create_semantic_set_type(metadata_tags=["user_id"])

        assert result == "set_type_456"
        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["metadata_tags"] == ["user_id"]
        assert "name" not in json_data or json_data.get("name") is None
        assert "description" not in json_data or json_data.get("description") is None

    def test_create_semantic_set_type_http_error(self, mock_client):
        """Test set type creation handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.create_semantic_set_type(metadata_tags=["user_id"])

    def test_create_semantic_set_type_client_closed(self, mock_client):
        """Test set type creation raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot create set type: client has been closed"
        ):
            memory.create_semantic_set_type(metadata_tags=["user_id"])

    def test_list_semantic_set_types_success(self, mock_client):
        """Test successful listing of semantic set types."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "set_types": [
                {
                    "id": "st_1",
                    "is_org_level": False,
                    "tags": ["user_id"],
                    "name": "User Set",
                    "description": "User-scoped sets",
                },
                {
                    "id": "st_2",
                    "is_org_level": True,
                    "tags": [],
                    "name": None,
                    "description": None,
                },
            ]
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.list_semantic_set_types()

        assert len(result) == 2
        assert result[0].id == "st_1"
        assert result[0].is_org_level is False
        assert result[0].tags == ["user_id"]
        assert result[0].name == "User Set"
        assert result[1].id == "st_2"
        assert result[1].is_org_level is True

    def test_list_semantic_set_types_http_error(self, mock_client):
        """Test listing set types handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.list_semantic_set_types()

    def test_list_semantic_set_types_client_closed(self, mock_client):
        """Test listing set types raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot list set types: client has been closed"
        ):
            memory.list_semantic_set_types()

    def test_delete_semantic_set_type_success(self, mock_client):
        """Test successful semantic set type deletion."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.delete_semantic_set_type(set_type_id="st_123")

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/set_type/delete" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["set_type_id"] == "st_123"

    def test_delete_semantic_set_type_http_error(self, mock_client):
        """Test set type deletion handles HTTP errors."""
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
            memory.delete_semantic_set_type(set_type_id="nonexistent")

    def test_delete_semantic_set_type_client_closed(self, mock_client):
        """Test set type deletion raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot delete set type: client has been closed"
        ):
            memory.delete_semantic_set_type(set_type_id="st_123")

    def test_get_semantic_set_id_success(self, mock_client):
        """Test successful semantic set ID retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"set_id": "mem_user_set_abc123"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_semantic_set_id(
            metadata_tags=["user_id"],
            is_org_level=False,
            set_metadata={"user_id": "user123"},
        )

        assert result == "mem_user_set_abc123"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/set_id/get" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["metadata_tags"] == ["user_id"]
        assert json_data["is_org_level"] is False
        assert json_data["set_metadata"] == {"user_id": "user123"}

    def test_get_semantic_set_id_minimal(self, mock_client):
        """Test set ID retrieval with minimal parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"set_id": "mem_project_set_def456"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_semantic_set_id(metadata_tags=[])

        assert result == "mem_project_set_def456"

    def test_get_semantic_set_id_http_error(self, mock_client):
        """Test set ID retrieval handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.get_semantic_set_id(metadata_tags=["user_id"])

    def test_get_semantic_set_id_client_closed(self, mock_client):
        """Test set ID retrieval raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot get set ID: client has been closed"
        ):
            memory.get_semantic_set_id(metadata_tags=["user_id"])

    def test_list_semantic_set_ids_success(self, mock_client):
        """Test successful listing of semantic sets."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "sets": [
                {
                    "id": "mem_user_set_abc",
                    "is_org_level": False,
                    "tags": ["user_id"],
                    "name": "User Set",
                    "description": None,
                },
                {
                    "id": "mem_project_set_def",
                    "is_org_level": True,
                    "tags": [],
                    "name": None,
                    "description": None,
                },
            ]
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.list_semantic_set_ids()

        assert len(result) == 2
        assert result[0].id == "mem_user_set_abc"
        assert result[0].is_org_level is False
        assert result[0].tags == ["user_id"]
        assert result[1].id == "mem_project_set_def"

    def test_list_semantic_set_ids_with_metadata_filter(self, mock_client):
        """Test listing sets with metadata filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"sets": []}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        memory.list_semantic_set_ids(set_metadata={"user_id": "user123"})

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["set_metadata"] == {"user_id": "user123"}

    def test_list_semantic_set_ids_http_error(self, mock_client):
        """Test listing sets handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.list_semantic_set_ids()

    def test_list_semantic_set_ids_client_closed(self, mock_client):
        """Test listing sets raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot list sets: client has been closed"
        ):
            memory.list_semantic_set_ids()

    def test_configure_semantic_set_success(self, mock_client):
        """Test successful semantic set configuration."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.configure_semantic_set(
            set_id="mem_user_set_abc",
            embedder_name="openai-embed",
            llm_name="gpt-4",
        )

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/set/configure" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["set_id"] == "mem_user_set_abc"
        assert json_data["embedder_name"] == "openai-embed"
        assert json_data["llm_name"] == "gpt-4"

    def test_configure_semantic_set_partial(self, mock_client):
        """Test set configuration with only embedder."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.configure_semantic_set(
            set_id="mem_user_set_abc",
            embedder_name="openai-embed",
        )

        assert result is True
        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["embedder_name"] == "openai-embed"
        assert "llm_name" not in json_data or json_data.get("llm_name") is None

    def test_configure_semantic_set_http_error(self, mock_client):
        """Test set configuration handles HTTP errors."""
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
            memory.configure_semantic_set(
                set_id="nonexistent",
                embedder_name="openai-embed",
            )

    def test_configure_semantic_set_client_closed(self, mock_client):
        """Test set configuration raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot configure set: client has been closed"
        ):
            memory.configure_semantic_set(
                set_id="mem_user_set_abc",
                embedder_name="openai-embed",
            )

    # --- Semantic Category Tests ---

    def test_get_semantic_category_success(self, mock_client):
        """Test successful category retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "id": "cat_123",
            "name": "preferences",
            "prompt": "Extract user preferences",
            "description": "Category for user preferences",
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_semantic_category("cat_123")

        assert result is not None
        assert result.id == "cat_123"
        assert result.name == "preferences"
        assert result.prompt == "Extract user preferences"
        assert result.description == "Category for user preferences"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category/get" in call_args[0][1]

    def test_get_semantic_category_not_found(self, mock_client):
        """Test category retrieval returns None for 404."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = None
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_semantic_category("nonexistent")
        assert result is None

    def test_get_semantic_category_client_closed(self, mock_client):
        """Test category retrieval raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot get category: client has been closed"
        ):
            memory.get_semantic_category("cat_123")

    def test_add_semantic_category_success(self, mock_client):
        """Test successful category creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"category_id": "cat_456"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.add_semantic_category(
            set_id="set_123",
            category_name="preferences",
            prompt="Extract user preferences",
            description="Category for user preferences",
        )

        assert result == "cat_456"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["set_id"] == "set_123"
        assert json_data["category_name"] == "preferences"
        assert json_data["prompt"] == "Extract user preferences"
        assert json_data["description"] == "Category for user preferences"

    def test_add_semantic_category_minimal(self, mock_client):
        """Test category creation with minimal parameters."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"category_id": "cat_789"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.add_semantic_category(
            set_id="set_123",
            category_name="preferences",
            prompt="Extract preferences",
        )

        assert result == "cat_789"
        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert "description" not in json_data or json_data.get("description") is None

    def test_add_semantic_category_http_error(self, mock_client):
        """Test category creation handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.add_semantic_category(
                set_id="set_123",
                category_name="preferences",
                prompt="Extract preferences",
            )

    def test_add_semantic_category_client_closed(self, mock_client):
        """Test category creation raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot add category: client has been closed"
        ):
            memory.add_semantic_category(
                set_id="set_123",
                category_name="preferences",
                prompt="Extract preferences",
            )

    def test_add_semantic_category_template_success(self, mock_client):
        """Test successful category template creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"category_id": "cat_template_456"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.add_semantic_category_template(
            set_type_id="st_123",
            category_name="preferences",
            prompt="Extract user preferences",
            description="Template for user preferences",
        )

        assert result == "cat_template_456"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category/template" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["set_type_id"] == "st_123"
        assert json_data["category_name"] == "preferences"

    def test_add_semantic_category_template_http_error(self, mock_client):
        """Test category template creation handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.add_semantic_category_template(
                set_type_id="st_123",
                category_name="preferences",
                prompt="Extract preferences",
            )

    def test_add_semantic_category_template_client_closed(self, mock_client):
        """Test category template creation raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot add category template: client has been closed"
        ):
            memory.add_semantic_category_template(
                set_type_id="st_123",
                category_name="preferences",
                prompt="Extract preferences",
            )

    def test_list_semantic_category_templates_success(self, mock_client):
        """Test successful listing of category templates."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "categories": [
                {
                    "id": "cat_1",
                    "name": "preferences",
                    "origin_type": "set_type",
                    "origin_id": "st_123",
                    "inherited": False,
                },
                {
                    "id": "cat_2",
                    "name": "facts",
                    "origin_type": "set",
                    "origin_id": "set_456",
                    "inherited": True,
                },
            ]
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.list_semantic_category_templates("st_123")

        assert len(result) == 2
        assert result[0].id == "cat_1"
        assert result[0].name == "preferences"
        assert result[0].origin_type == "set_type"
        assert result[0].inherited is False
        assert result[1].id == "cat_2"
        assert result[1].inherited is True

    def test_list_semantic_category_templates_http_error(self, mock_client):
        """Test listing category templates handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.list_semantic_category_templates("st_123")

    def test_list_semantic_category_templates_client_closed(self, mock_client):
        """Test listing category templates raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError,
            match="Cannot list category templates: client has been closed",
        ):
            memory.list_semantic_category_templates("st_123")

    def test_disable_semantic_category_success(self, mock_client):
        """Test successful category disabling."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.disable_semantic_category(
            set_id="set_123",
            category_name="preferences",
        )

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category/disable" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["set_id"] == "set_123"
        assert json_data["category_name"] == "preferences"

    def test_disable_semantic_category_http_error(self, mock_client):
        """Test category disabling handles HTTP errors."""
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
            memory.disable_semantic_category(
                set_id="set_123",
                category_name="nonexistent",
            )

    def test_disable_semantic_category_client_closed(self, mock_client):
        """Test category disabling raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot disable category: client has been closed"
        ):
            memory.disable_semantic_category(
                set_id="set_123",
                category_name="preferences",
            )

    def test_get_semantic_category_set_ids_success(self, mock_client):
        """Test successful retrieval of category set IDs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"set_ids": ["set_1", "set_2", "set_3"]}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_semantic_category_set_ids("cat_123")

        assert result == ["set_1", "set_2", "set_3"]
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category/set_ids/get" in call_args[0][1]

    def test_get_semantic_category_set_ids_empty(self, mock_client):
        """Test category set IDs retrieval with empty result."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"set_ids": []}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_semantic_category_set_ids("cat_123")
        assert result == []

    def test_get_semantic_category_set_ids_http_error(self, mock_client):
        """Test category set IDs retrieval handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        with pytest.raises(requests.RequestException):
            memory.get_semantic_category_set_ids("cat_123")

    def test_get_semantic_category_set_ids_client_closed(self, mock_client):
        """Test category set IDs retrieval raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot get category set IDs: client has been closed"
        ):
            memory.get_semantic_category_set_ids("cat_123")

    def test_delete_semantic_category_success(self, mock_client):
        """Test successful category deletion."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.delete_semantic_category("cat_123")

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category/delete" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["category_id"] == "cat_123"

    def test_delete_semantic_category_http_error(self, mock_client):
        """Test category deletion handles HTTP errors."""
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
            memory.delete_semantic_category("nonexistent")

    def test_delete_semantic_category_client_closed(self, mock_client):
        """Test category deletion raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot delete category: client has been closed"
        ):
            memory.delete_semantic_category("cat_123")

    # --- Semantic Tag Tests ---

    def test_add_semantic_tag_success(self, mock_client):
        """Test successful tag creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"tag_id": "tag_456"}
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.add_semantic_tag(
            category_id="cat_123",
            tag_name="food_preference",
            tag_description="User food preferences",
        )

        assert result == "tag_456"
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category/tag" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["category_id"] == "cat_123"
        assert json_data["tag_name"] == "food_preference"
        assert json_data["tag_description"] == "User food preferences"

    def test_add_semantic_tag_http_error(self, mock_client):
        """Test tag creation handles HTTP errors."""
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
            memory.add_semantic_tag(
                category_id="nonexistent",
                tag_name="food_preference",
                tag_description="User food preferences",
            )

    def test_add_semantic_tag_client_closed(self, mock_client):
        """Test tag creation raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot add tag: client has been closed"
        ):
            memory.add_semantic_tag(
                category_id="cat_123",
                tag_name="food_preference",
                tag_description="User food preferences",
            )

    def test_delete_semantic_tag_success(self, mock_client):
        """Test successful tag deletion."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.delete_semantic_tag("tag_123")

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories/semantic/category/tag/delete" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["tag_id"] == "tag_123"

    def test_delete_semantic_tag_http_error(self, mock_client):
        """Test tag deletion handles HTTP errors."""
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
            memory.delete_semantic_tag("nonexistent")

    def test_delete_semantic_tag_client_closed(self, mock_client):
        """Test tag deletion raises RuntimeError when client is closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError, match="Cannot delete tag: client has been closed"
        ):
            memory.delete_semantic_tag("tag_123")

    # --- Episodic Memory Configuration Tests ---

    def test_get_episodic_memory_config_success(self, mock_client):
        """Test successful episodic memory config retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "enabled": True,
            "long_term_memory_enabled": True,
            "short_term_memory_enabled": False,
        }
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.get_episodic_memory_config()

        assert result.enabled is True
        assert result.long_term_memory_enabled is True
        assert result.short_term_memory_enabled is False
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memory/episodic/config/get" in call_args[0][1]

    def test_get_episodic_memory_config_http_error(self, mock_client):
        """Test episodic memory config retrieval handles HTTP errors."""
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
            memory.get_episodic_memory_config()

    def test_get_episodic_memory_config_client_closed(self, mock_client):
        """Test episodic memory config retrieval raises RuntimeError when client closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError,
            match="Cannot get episodic memory config: client has been closed",
        ):
            memory.get_episodic_memory_config()

    def test_configure_episodic_memory_success(self, mock_client):
        """Test successful episodic memory configuration."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.configure_episodic_memory(
            enabled=True,
            long_term_memory_enabled=False,
            short_term_memory_enabled=True,
        )

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memory/episodic/config" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["enabled"] is True
        assert json_data["long_term_memory_enabled"] is False
        assert json_data["short_term_memory_enabled"] is True

    def test_configure_episodic_memory_partial(self, mock_client):
        """Test episodic memory configuration with partial parameters."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        result = memory.configure_episodic_memory(enabled=False)

        assert result is True
        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["enabled"] is False
        assert "long_term_memory_enabled" not in json_data
        assert "short_term_memory_enabled" not in json_data

    def test_configure_episodic_memory_http_error(self, mock_client):
        """Test episodic memory configuration handles HTTP errors."""
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
            memory.configure_episodic_memory(enabled=False)

    def test_configure_episodic_memory_client_closed(self, mock_client):
        """Test episodic memory configuration raises RuntimeError when client closed."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )
        memory._client_closed = True

        with pytest.raises(
            RuntimeError,
            match="Cannot configure episodic memory: client has been closed",
        ):
            memory.configure_episodic_memory(enabled=False)
