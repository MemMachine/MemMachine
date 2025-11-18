"""Unit tests for Memory class."""

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
        client._session = Mock()
        return client

    def test_init_with_required_params(self, mock_client):
        """Test Memory initialization with required parameters."""
        memory = Memory(
            client=mock_client,
            group_id="test_group",
            agent_id="test_agent",
            user_id="test_user",
        )

        assert memory.client == mock_client
        assert memory.group_id == "test_group"
        assert memory.agent_id == ["test_agent"]
        assert memory.user_id == ["test_user"]
        assert memory.session_id is not None  # Auto-generated

    def test_init_with_string_ids(self, mock_client):
        """Test Memory initialization with string IDs."""
        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        assert memory.agent_id == ["agent1"]
        assert memory.user_id == ["user1"]

    def test_init_with_list_ids(self, mock_client):
        """Test Memory initialization with list IDs."""
        memory = Memory(
            client=mock_client,
            agent_id=["agent1", "agent2"],
            user_id=["user1", "user2"],
        )

        assert memory.agent_id == ["agent1", "agent2"]
        assert memory.user_id == ["user1", "user2"]

    def test_init_with_custom_session_id(self, mock_client):
        """Test Memory initialization with custom session_id."""
        memory = Memory(
            client=mock_client,
            agent_id="agent1",
            user_id="user1",
            session_id="custom_session",
        )

        assert memory.session_id == "custom_session"

    def test_init_missing_user_id_raises_error(self, mock_client):
        """Test that missing user_id raises ValueError."""
        with pytest.raises(ValueError, match="Both user_id and agent_id are required"):
            Memory(client=mock_client, agent_id="agent1")

    def test_init_missing_agent_id_raises_error(self, mock_client):
        """Test that missing agent_id raises ValueError."""
        with pytest.raises(ValueError, match="Both user_id and agent_id are required"):
            Memory(client=mock_client, user_id="user1")

    def test_init_empty_user_id_raises_error(self, mock_client):
        """Test that empty user_id raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Memory(client=mock_client, agent_id="agent1", user_id=[])

    def test_init_defaults_group_id(self, mock_client):
        """Test that group_id defaults to first user_id if not provided."""
        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        assert memory.group_id == "user1"

    def test_add_success(self, mock_client):
        """Test successful memory addition."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(
            client=mock_client,
            group_id="test_group",
            agent_id="test_agent",
            user_id="test_user",
        )

        result = memory.add("Test content")

        assert result is True
        mock_client._session.post.assert_called_once()
        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["episode_content"] == "Test content"
        assert call_kwargs["json"]["producer"] == "test_user"
        assert call_kwargs["json"]["produced_for"] == "test_agent"

    def test_add_with_metadata(self, mock_client):
        """Test adding memory with metadata."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        metadata = {"type": "preference", "category": "food"}
        memory.add("I like pizza", metadata=metadata)

        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["metadata"] == metadata

    def test_add_with_custom_producer(self, mock_client):
        """Test adding memory with custom producer and produced_for that are in the context."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(
            client=mock_client,
            agent_id=["agent1", "agent2"],
            user_id=["user1", "user2"],
        )

        # Use valid producer and produced_for from the context
        memory.add("Content", producer="user2", produced_for="agent2")

        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["producer"] == "user2"
        assert call_kwargs["json"]["produced_for"] == "agent2"

    def test_add_with_invalid_producer_raises_error(self, mock_client):
        """Test that invalid producer raises ValueError."""
        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        with pytest.raises(ValueError, match="producer.*must be in"):
            memory.add("Content", producer="invalid_user")

    def test_add_with_invalid_produced_for_raises_error(self, mock_client):
        """Test that invalid produced_for raises ValueError."""
        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        with pytest.raises(ValueError, match="produced_for.*must be in"):
            memory.add("Content", produced_for="invalid_agent")

    def test_add_with_valid_producer_from_agent_list(self, mock_client):
        """Test that producer can be from agent_id list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(
            client=mock_client, agent_id=["agent1", "agent2"], user_id="user1"
        )

        # Producer can be an agent
        memory.add("Content", producer="agent1")

        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["producer"] == "agent1"

    def test_add_with_valid_produced_for_from_user_list(self, mock_client):
        """Test that produced_for can be from user_id list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(
            client=mock_client, agent_id="agent1", user_id=["user1", "user2"]
        )

        # produced_for can be a user
        memory.add("Content", produced_for="user2")

        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["produced_for"] == "user2"

    def test_add_sets_headers(self, mock_client):
        """Test that add sets correct headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(
            client=mock_client,
            group_id="test_group",
            agent_id="agent1",
            user_id="user1",
            session_id="test_session",
        )

        memory.add("Content")

        call_kwargs = mock_client._session.post.call_args[1]
        headers = call_kwargs["headers"]
        assert headers["group-id"] == "test_group"
        assert headers["session-id"] == "test_session"
        assert headers["agent-id"] == "agent1"
        assert headers["user-id"] == "user1"

    def test_add_with_multiple_users_agents(self, mock_client):
        """Test add with multiple users and agents."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(
            client=mock_client,
            agent_id=["agent1", "agent2"],
            user_id=["user1", "user2"],
        )

        memory.add("Content")

        call_kwargs = mock_client._session.post.call_args[1]
        headers = call_kwargs["headers"]
        assert headers["agent-id"] == "agent1,agent2"
        assert headers["user-id"] == "user1,user2"

    def test_add_request_exception(self, mock_client):
        """Test add raises exception on request failure."""
        mock_client._session.post.side_effect = requests.RequestException(
            "Network error"
        )

        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        with pytest.raises(requests.RequestException):
            memory.add("Content")

    def test_add_http_error(self, mock_client):
        """Test add handles HTTP errors correctly."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_client._session.post.return_value = mock_response

        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        with pytest.raises(requests.RequestException):
            memory.add("Content")

    def test_search_success(self, mock_client):
        """Test successful memory search."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": {
                "episodic_memory": [["result1", "result2"]],
                "profile_memory": [],
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        results = memory.search("test query")

        assert "episodic_memory" in results
        mock_client._session.post.assert_called_once()
        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["query"] == "test query"

    def test_search_with_limit(self, mock_client):
        """Test search with limit parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": {}}
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        memory.search("query", limit=10)

        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["limit"] == 10

    def test_search_with_filters(self, mock_client):
        """Test search with filter dictionary."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": {}}
        mock_response.raise_for_status = Mock()
        mock_client._session.post.return_value = mock_response

        memory = Memory(client=mock_client, agent_id="agent1", user_id="user1")

        filters = {"category": "work", "type": "preference"}
        memory.search("query", filter_dict=filters)

        call_kwargs = mock_client._session.post.call_args[1]
        assert call_kwargs["json"]["filter"] == filters

    def test_get_context(self, mock_client):
        """Test getting memory context."""
        memory = Memory(
            client=mock_client,
            group_id="test_group",
            agent_id="agent1",
            user_id="user1",
            session_id="test_session",
        )

        context = memory.get_context()

        assert context["group_id"] == "test_group"
        assert context["agent_id"] == ["agent1"]
        assert context["user_id"] == ["user1"]
        assert context["session_id"] == "test_session"

    def test_repr(self, mock_client):
        """Test string representation."""
        memory = Memory(
            client=mock_client,
            group_id="test_group",
            agent_id="agent1",
            user_id="user1",
            session_id="test_session",
        )

        repr_str = repr(memory)
        assert "test_group" in repr_str
        assert "test_session" in repr_str
