"""Unit tests for MemMachineClient class."""

from unittest.mock import Mock, patch

import pytest
import requests

from memmachine.rest_client.client import MemMachineClient
from memmachine.rest_client.memory import Memory


class TestMemMachineClient:
    """Test cases for MemMachineClient."""

    def test_init_default_values(self):
        """Test client initialization with default values."""
        client = MemMachineClient(base_url="http://localhost:8080")

        assert client.api_key is None
        assert client.base_url == "http://localhost:8080"
        assert client.timeout == 30
        assert client.max_retries == 3
        assert client._session is not None

    def test_init_custom_values(self):
        """Test client initialization with custom values."""
        client = MemMachineClient(
            api_key="test_key", base_url="http://test:9000", timeout=60, max_retries=5
        )

        assert client.api_key == "test_key"
        assert client.base_url == "http://test:9000"
        assert client.timeout == 60
        assert client.max_retries == 5
        assert "Authorization" in client._session.headers
        assert client._session.headers["Authorization"] == "Bearer test_key"

    def test_init_base_url_trailing_slash(self):
        """Test that trailing slashes are removed from base_url."""
        client = MemMachineClient(base_url="http://localhost:8080/")
        assert client.base_url == "http://localhost:8080"

    def test_init_missing_base_url_raises_error(self):
        """Test that missing base_url raises ValueError."""
        with pytest.raises(ValueError, match="base_url is required"):
            MemMachineClient()

    def test_memory_creation(self):
        """Test creating a Memory instance."""
        client = MemMachineClient(base_url="http://localhost:8080")
        memory = client.memory(
            group_id="test_group",
            agent_id="test_agent",
            user_id="test_user",
            session_id="test_session",
        )

        assert isinstance(memory, Memory)
        assert memory.client == client
        assert memory.group_id == "test_group"
        assert memory.agent_id == ["test_agent"]
        assert memory.user_id == ["test_user"]
        assert memory.session_id == "test_session"

    def test_memory_creation_with_lists(self):
        """Test creating Memory with list IDs."""
        client = MemMachineClient(base_url="http://localhost:8080")
        memory = client.memory(
            group_id="test_group",
            agent_id=["agent1", "agent2"],
            user_id=["user1", "user2"],
        )

        assert memory.agent_id == ["agent1", "agent2"]
        assert memory.user_id == ["user1", "user2"]

    @patch("requests.Session.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = MemMachineClient(base_url="http://localhost:8080")
        result = client.health_check()

        assert result == {"status": "healthy"}
        mock_get.assert_called_once()
        assert "/health" in mock_get.call_args[0][0]

    @patch("requests.Session.get")
    def test_health_check_failure(self, mock_get):
        """Test health check failure."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        client = MemMachineClient(base_url="http://localhost:8080")

        with pytest.raises(requests.RequestException):
            client.health_check()

    def test_close(self):
        """Test closing the client."""
        client = MemMachineClient(base_url="http://localhost:8080")
        mock_session = Mock()
        client._session = mock_session

        client.close()

        mock_session.close.assert_called_once()

    def test_context_manager(self):
        """Test using client as context manager."""
        with MemMachineClient(base_url="http://localhost:8080") as client:
            assert isinstance(client, MemMachineClient)

        # After context exit, session should be closed
        # Note: We can't easily verify this without more complex mocking

    def test_repr(self):
        """Test string representation."""
        client = MemMachineClient(base_url="http://test:8080")
        assert repr(client) == "MemMachineClient(base_url='http://test:8080')"

    def test_default_headers(self):
        """Test that default headers are set correctly."""
        client = MemMachineClient(base_url="http://localhost:8080")

        assert client._session.headers["Content-Type"] == "application/json"
        assert "User-Agent" in client._session.headers
        assert "MemMachineClient" in client._session.headers["User-Agent"]

    def test_authorization_header_with_api_key(self):
        """Test authorization header is set when API key is provided."""
        client = MemMachineClient(api_key="test_key", base_url="http://localhost:8080")
        assert client._session.headers["Authorization"] == "Bearer test_key"

    def test_authorization_header_without_api_key(self):
        """Test authorization header is not set when API key is not provided."""
        client = MemMachineClient(base_url="http://localhost:8080")
        assert "Authorization" not in client._session.headers
