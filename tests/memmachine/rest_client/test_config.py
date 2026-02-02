"""Unit tests for Config class."""

from unittest.mock import Mock

import pytest
import requests

from memmachine.common.api.config_spec import (
    DeleteResourceResponse,
    GetConfigResponse,
    ResourcesStatus,
    UpdateEpisodicMemorySpec,
    UpdateMemoryConfigResponse,
    UpdateResourceResponse,
    UpdateSemanticMemorySpec,
)
from memmachine.rest_client.config import Config


@pytest.fixture
def mock_client():
    """Create a mock MemMachineClient."""
    client = Mock()
    client.closed = False
    client.base_url = "http://localhost:8080"
    return client


@pytest.fixture
def config(mock_client):
    """Create a Config instance with a mock client."""
    return Config(client=mock_client)


def _mock_response(json_data: dict) -> Mock:
    """Create a mock response with the given JSON data."""
    resp = Mock(spec=requests.Response)
    resp.json.return_value = json_data
    resp.raise_for_status = Mock()
    return resp


class TestConfig:
    """Test cases for the Config class."""

    def test_get_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {
                "resources": {
                    "embedders": [],
                    "language_models": [],
                    "rerankers": [],
                    "databases": [],
                }
            }
        )
        result = config.get_config()
        assert isinstance(result, GetConfigResponse)
        mock_client.request.assert_called_once_with(
            "GET",
            "http://localhost:8080/api/v2/config",
            timeout=None,
        )

    def test_get_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.get_config()

    def test_get_resources(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {
                "embedders": [
                    {"name": "default", "provider": "openai", "status": "ready"}
                ],
                "language_models": [],
                "rerankers": [],
                "databases": [],
            }
        )
        result = config.get_resources()
        assert isinstance(result, ResourcesStatus)
        assert len(result.embedders) == 1
        assert result.embedders[0].name == "default"
        mock_client.request.assert_called_once_with(
            "GET",
            "http://localhost:8080/api/v2/config/resources",
            timeout=None,
        )

    def test_get_resources_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.get_resources()

    def test_update_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Memory configuration updated"}
        )
        result = config.update_memory_config(
            episodic_memory=UpdateEpisodicMemorySpec.model_validate({"enabled": False}),
            semantic_memory=UpdateSemanticMemorySpec.model_validate({"enabled": True}),
        )
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        assert call_args[0] == ("PUT", "http://localhost:8080/api/v2/config/memory")
        assert "episodic_memory" in call_args[1]["json"]

    def test_update_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.update_memory_config()

    def test_add_embedder(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "status": "ready", "error": None}
        )
        result = config.add_embedder(
            name="my-embedder",
            provider="openai",
            config={"api_key": "sk-test", "model": "text-embedding-3-small"},
        )
        assert isinstance(result, UpdateResourceResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        assert call_args[0] == (
            "POST",
            "http://localhost:8080/api/v2/config/resources/embedders",
        )
        body = call_args[1]["json"]
        assert body["name"] == "my-embedder"
        assert body["provider"] == "openai"

    def test_add_embedder_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.add_embedder(name="x", provider="openai", config={})

    def test_add_language_model(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "status": "ready", "error": None}
        )
        result = config.add_language_model(
            name="my-llm",
            provider="openai-responses",
            config={"api_key": "sk-test"},
        )
        assert isinstance(result, UpdateResourceResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        assert call_args[0] == (
            "POST",
            "http://localhost:8080/api/v2/config/resources/language_models",
        )

    def test_add_language_model_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.add_language_model(name="x", provider="openai-responses", config={})

    def test_delete_embedder(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Embedder deleted"}
        )
        result = config.delete_embedder(name="my-embedder")
        assert isinstance(result, DeleteResourceResponse)
        assert result.success is True
        mock_client.request.assert_called_once_with(
            "DELETE",
            "http://localhost:8080/api/v2/config/resources/embedders/my-embedder",
            timeout=None,
        )

    def test_delete_embedder_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.delete_embedder(name="x")

    def test_delete_language_model(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Language model deleted"}
        )
        result = config.delete_language_model(name="my-llm")
        assert isinstance(result, DeleteResourceResponse)
        assert result.success is True
        mock_client.request.assert_called_once_with(
            "DELETE",
            "http://localhost:8080/api/v2/config/resources/language_models/my-llm",
            timeout=None,
        )

    def test_delete_language_model_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.delete_language_model(name="x")

    def test_retry_embedder(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "status": "ready", "error": None}
        )
        result = config.retry_embedder(name="my-embedder")
        assert isinstance(result, UpdateResourceResponse)
        mock_client.request.assert_called_once_with(
            "POST",
            "http://localhost:8080/api/v2/config/resources/embedders/my-embedder/retry",
            timeout=None,
        )

    def test_retry_embedder_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.retry_embedder(name="x")

    def test_retry_language_model(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "status": "ready", "error": None}
        )
        result = config.retry_language_model(name="my-llm")
        assert isinstance(result, UpdateResourceResponse)
        mock_client.request.assert_called_once_with(
            "POST",
            "http://localhost:8080/api/v2/config/resources/language_models/my-llm/retry",
            timeout=None,
        )

    def test_retry_language_model_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.retry_language_model(name="x")

    def test_retry_reranker(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "status": "ready", "error": None}
        )
        result = config.retry_reranker(name="my-reranker")
        assert isinstance(result, UpdateResourceResponse)
        mock_client.request.assert_called_once_with(
            "POST",
            "http://localhost:8080/api/v2/config/resources/rerankers/my-reranker/retry",
            timeout=None,
        )

    def test_retry_reranker_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.retry_reranker(name="x")

    def test_get_config_with_timeout(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {
                "resources": {
                    "embedders": [],
                    "language_models": [],
                    "rerankers": [],
                    "databases": [],
                }
            }
        )
        config.get_config(timeout=60)
        mock_client.request.assert_called_once_with(
            "GET",
            "http://localhost:8080/api/v2/config",
            timeout=60,
        )

    def test_request_exception_propagated(self, config, mock_client):
        mock_client.request.side_effect = requests.ConnectionError("Connection refused")
        with pytest.raises(requests.ConnectionError):
            config.get_config()

    def test_repr(self, config):
        result = repr(config)
        assert "Config" in result
