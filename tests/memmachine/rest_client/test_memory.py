"""Unit tests for Memory class (v2 API)."""

from unittest.mock import Mock

import pytest
import requests

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
            group_id="test_group",
            agent_id="test_agent",
            user_id="test_user",
        )

        assert memory.client == mock_client
        assert memory.org_id == "test_org"
        assert memory.project_id == "test_project"
        assert memory.group_id == "test_group"
        assert memory.agent_id == ["test_agent"]
        assert memory.user_id == ["test_user"]
        assert memory.session_id is None  # Not auto-generated in v2

    def test_init_with_only_required_params(self, mock_client):
        """Test Memory initialization with only org_id and project_id."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
        )

        assert memory.org_id == "test_org"
        assert memory.project_id == "test_project"
        assert memory.group_id is None
        assert memory.agent_id is None
        assert memory.user_id is None
        assert memory.session_id is None

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
            agent_id="agent1",
            user_id="user1",
        )

        assert memory.agent_id == ["agent1"]
        assert memory.user_id == ["user1"]

    def test_init_with_list_ids(self, mock_client):
        """Test Memory initialization with list IDs."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            agent_id=["agent1", "agent2"],
            user_id=["user1", "user2"],
        )

        assert memory.agent_id == ["agent1", "agent2"]
        assert memory.user_id == ["user1", "user2"]

    def test_init_with_custom_session_id(self, mock_client):
        """Test Memory initialization with custom session_id."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            agent_id="agent1",
            user_id="user1",
            session_id="custom_session",
        )

        assert memory.session_id == "custom_session"

    def test_add_success(self, mock_client):
        """Test successful memory addition with v2 API format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            group_id="test_group",
            agent_id="test_agent",
            user_id="test_user",
        )

        result = memory.add("Test content")

        assert result is True
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v2/memories" in call_args[0][1]
        json_data = call_args[1]["json"]
        assert json_data["org_id"] == "test_org"
        assert json_data["project_id"] == "test_project"
        assert len(json_data["messages"]) == 1
        assert json_data["messages"][0]["content"] == "Test content"
        assert json_data["messages"][0]["role"] == "user"
        assert json_data["messages"][0]["producer"] == "test_user"
        assert "timestamp" in json_data["messages"][0]
        assert "metadata" in json_data["messages"][0]

    def test_add_with_metadata(self, mock_client):
        """Test adding memory with metadata."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            agent_id="agent1",
            user_id="user1",
        )

        metadata = {"type": "preference", "category": "food"}
        memory.add("I like pizza", metadata=metadata)

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
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            user_id="user1",
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
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            agent_id=["agent1", "agent2"],
            user_id=["user1", "user2"],
        )

        memory.add("Content", producer="user2", produced_for="agent2")

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["messages"][0]["producer"] == "user2"
        # produced_for should be a direct field in the message
        assert json_data["messages"][0]["produced_for"] == "agent2"

    def test_add_with_episode_type(self, mock_client):
        """Test adding memory with episode_type."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.request.return_value = mock_response

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            user_id="user1",
        )

        memory.add("Content", episode_type="text")

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        assert json_data["messages"][0]["metadata"]["episode_type"] == "text"

    def test_add_request_exception(self, mock_client):
        """Test add raises exception on request failure."""
        mock_client.request.side_effect = requests.RequestException("Network error")

        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            agent_id="agent1",
            user_id="user1",
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
            agent_id="agent1",
            user_id="user1",
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
            agent_id="agent1",
            user_id="user1",
        )

        results = memory.search("test query")

        assert "episodic_memory" in results
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
            agent_id="agent1",
            user_id="user1",
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
            agent_id="agent1",
            user_id="user1",
        )

        filters = {"category": "work", "type": "preference"}
        memory.search("query", filter_dict=filters)

        call_args = mock_client.request.call_args
        json_data = call_args[1]["json"]
        # Filter should be JSON string in v2 API
        import json

        filter_dict = json.loads(json_data["filter"])
        assert filter_dict["category"] == "work"
        assert filter_dict["type"] == "preference"

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
            group_id="test_group",
            agent_id="agent1",
            user_id="user1",
            session_id="test_session",
        )

        context = memory.get_context()

        assert context["org_id"] == "test_org"
        assert context["project_id"] == "test_project"
        assert context["group_id"] == "test_group"
        assert context["agent_id"] == ["agent1"]
        assert context["user_id"] == ["user1"]
        assert context["session_id"] == "test_session"

    def test_repr(self, mock_client):
        """Test string representation."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            group_id="test_group",
            agent_id="agent1",
            user_id="user1",
            session_id="test_session",
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
            group_id="test_group",
            agent_id="agent1",
            user_id="user1",
            session_id="test_session",
        )

        metadata = memory._build_metadata({"custom": "value"})

        assert metadata["custom"] == "value"
        assert metadata["group_id"] == "test_group"
        assert metadata["user_id"] == "user1"
        assert metadata["agent_id"] == "agent1"
        assert metadata["session_id"] == "test_session"

    def test_build_metadata_with_list_ids(self, mock_client):
        """Test that _build_metadata handles list IDs correctly."""
        memory = Memory(
            client=mock_client,
            org_id="test_org",
            project_id="test_project",
            agent_id=["agent1", "agent2"],
            user_id=["user1", "user2"],
        )

        metadata = memory._build_metadata({})

        # When multiple IDs, should keep as list
        assert metadata["agent_id"] == ["agent1", "agent2"]
        assert metadata["user_id"] == ["user1", "user2"]

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
