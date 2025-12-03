"""Unit tests for MemMachineClient class (v2 API)."""

from unittest.mock import Mock, patch

import pytest
import requests

from memmachine.rest_client.client import MemMachineClient
from memmachine.rest_client.memory import Memory
from memmachine.rest_client.project import Project


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
            api_key="test_key",
            base_url="http://test:9000",
            timeout=60,
            max_retries=5,
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

    def test_project_memory_creation(self):
        """Test creating a Memory instance from a Project."""
        client = MemMachineClient(base_url="http://localhost:8080")
        # Mock project creation
        with patch.object(client, "_session") as mock_session:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_session.post.return_value = mock_response

            project = client.create_project(
                org_id="test_org",
                project_id="test_project",
            )

            memory = project.memory(
                group_id="test_group",
                agent_id="test_agent",
                user_id="test_user",
                session_id="test_session",
            )

            assert isinstance(memory, Memory)
            assert memory.client == client
            assert memory.org_id == "test_org"
            assert memory.project_id == "test_project"
            assert memory.group_id == "test_group"
            assert memory.agent_id == "test_agent"
            assert memory.user_id == "test_user"
            assert memory.session_id == "test_session"

    def test_project_memory_creation_with_lists(self):
        """Test creating Memory with list IDs from Project."""
        client = MemMachineClient(base_url="http://localhost:8080")
        # Mock project creation
        with patch.object(client, "_session") as mock_session:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_session.post.return_value = mock_response

            project = client.create_project(
                org_id="test_org",
                project_id="test_project",
            )

            memory = project.memory(
                group_id="test_group",
                agent_id="agent1",
                user_id="user1",
            )

            assert memory.org_id == "test_org"
            assert memory.project_id == "test_project"
            assert memory.agent_id == "agent1"
            assert memory.user_id == "user1"

    def test_project_memory_creation_without_optional_params(self):
        """Test creating Memory with only required params from Project."""
        client = MemMachineClient(base_url="http://localhost:8080")
        # Mock project creation
        with patch.object(client, "_session") as mock_session:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_session.post.return_value = mock_response

            project = client.create_project(
                org_id="test_org",
                project_id="test_project",
            )

            memory = project.memory()

            assert memory.org_id == "test_org"
            assert memory.project_id == "test_project"
            assert memory.group_id is None
            assert memory.agent_id is None
            assert memory.user_id is None

    def test_create_project_success(self):
        """Test successful project creation."""
        client = MemMachineClient(base_url="http://localhost:8080")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        # Mock the JSON response to return a dictionary with the expected fields
        mock_response.json.return_value = {
            "org_id": "test_org",
            "project_id": "test_project",
            "description": "Test project",
            "config": {
                "embedder": "default",
                "reranker": "default",
            },
        }
        client._session.post = Mock(return_value=mock_response)

        result = client.create_project(
            org_id="test_org",
            project_id="test_project",
            description="Test project",
            embedder="default",
            reranker="default",
        )

        assert isinstance(result, Project)
        assert result.org_id == "test_org"
        assert result.project_id == "test_project"
        assert result.description == "Test project"
        client._session.post.assert_called_once()
        call_args = client._session.post.call_args
        assert "/api/v2/projects" in call_args[0][0]
        assert call_args[1]["json"]["org_id"] == "test_org"
        assert call_args[1]["json"]["project_id"] == "test_project"
        assert call_args[1]["json"]["description"] == "Test project"
        assert call_args[1]["json"]["config"]["embedder"] == "default"
        assert call_args[1]["json"]["config"]["reranker"] == "default"

    def test_create_project_failure(self):
        """Test project creation failure."""
        client = MemMachineClient(base_url="http://localhost:8080")
        client._session.post = Mock(
            side_effect=requests.RequestException("Connection failed")
        )

        with pytest.raises(requests.RequestException):
            client.create_project(
                org_id="test_org",
                project_id="test_project",
            )

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
        assert "/api/v2/health" in mock_get.call_args[0][0]

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
        assert client._closed is True

    def test_context_manager(self):
        """Test using client as context manager."""
        with MemMachineClient(base_url="http://localhost:8080") as client:
            assert isinstance(client, MemMachineClient)

        # After context exit, client should be closed
        assert client._closed is True

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

    def test_request_method(self):
        """Test the request method delegates to session.request."""
        client = MemMachineClient(base_url="http://localhost:8080")
        mock_response = Mock()
        client._session.request = Mock(return_value=mock_response)

        result = client.request("GET", "http://example.com", json={"key": "value"})

        assert result == mock_response
        client._session.request.assert_called_once_with(
            "GET", "http://example.com", timeout=30, json={"key": "value"}
        )
