"""Unit tests for Config class."""

from unittest.mock import Mock

import pytest
import requests
from memmachine_common.api.config_spec import (
    DeleteResourceResponse,
    EpisodicMemoryConfigResponse,
    GetConfigResponse,
    LongTermMemoryConfigResponse,
    ResourcesStatus,
    SemanticMemoryConfigResponse,
    ShortTermMemoryConfigResponse,
    UpdateEpisodicMemorySpec,
    UpdateLongTermMemorySpec,
    UpdateMemoryConfigResponse,
    UpdateResourceResponse,
    UpdateSemanticMemorySpec,
    UpdateShortTermMemorySpec,
)

from memmachine_client import Config


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

    def test_get_long_term_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {
                "embedder": "default-embedder",
                "reranker": "default-reranker",
                "vector_graph_store": "neo4j-store",
                "enabled": True,
            }
        )
        result = config.get_long_term_memory_config()
        assert isinstance(result, LongTermMemoryConfigResponse)
        assert result.embedder == "default-embedder"
        assert result.reranker == "default-reranker"
        assert result.vector_graph_store == "neo4j-store"
        assert result.enabled is True
        mock_client.request.assert_called_once_with(
            "GET",
            "http://localhost:8080/api/v2/config/memory/episodic/long_term",
            timeout=None,
        )

    def test_get_long_term_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.get_long_term_memory_config()

    def test_get_short_term_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {
                "llm_model": "default-llm",
                "message_capacity": 100,
                "enabled": True,
            }
        )
        result = config.get_short_term_memory_config()
        assert isinstance(result, ShortTermMemoryConfigResponse)
        assert result.llm_model == "default-llm"
        assert result.message_capacity == 100
        assert result.enabled is True
        mock_client.request.assert_called_once_with(
            "GET",
            "http://localhost:8080/api/v2/config/memory/episodic/short_term",
            timeout=None,
        )

    def test_get_short_term_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.get_short_term_memory_config()

    def test_update_long_term_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Long-term memory configuration updated"}
        )
        result = config.update_long_term_memory_config(
            embedder="new-embedder",
            reranker="new-reranker",
            vector_graph_store="new-store",
            enabled=False,
        )
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        assert call_args[0] == (
            "PUT",
            "http://localhost:8080/api/v2/config/memory/episodic/long_term",
        )
        body = call_args[1]["json"]
        assert body["embedder"] == "new-embedder"
        assert body["reranker"] == "new-reranker"
        assert body["vector_graph_store"] == "new-store"
        assert body["enabled"] is False

    def test_update_long_term_memory_config_partial(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Long-term memory configuration updated"}
        )
        result = config.update_long_term_memory_config(embedder="new-embedder")
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        body = call_args[1]["json"]
        assert body == {"embedder": "new-embedder"}

    def test_update_long_term_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.update_long_term_memory_config(embedder="x")

    def test_update_short_term_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Short-term memory configuration updated"}
        )
        result = config.update_short_term_memory_config(
            llm_model="new-llm",
            message_capacity=200,
            enabled=False,
        )
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        assert call_args[0] == (
            "PUT",
            "http://localhost:8080/api/v2/config/memory/episodic/short_term",
        )
        body = call_args[1]["json"]
        assert body["llm_model"] == "new-llm"
        assert body["message_capacity"] == 200
        assert body["enabled"] is False

    def test_update_short_term_memory_config_partial(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Short-term memory configuration updated"}
        )
        result = config.update_short_term_memory_config(message_capacity=50)
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        body = call_args[1]["json"]
        assert body == {"message_capacity": 50}

    def test_update_short_term_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.update_short_term_memory_config(llm_model="x")

    def test_get_semantic_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {
                "enabled": True,
                "database": "postgres-db",
                "llm_model": "gpt-4",
                "embedding_model": "openai-embedder",
            }
        )
        result = config.get_semantic_memory_config()
        assert isinstance(result, SemanticMemoryConfigResponse)
        assert result.enabled is True
        assert result.database == "postgres-db"
        assert result.llm_model == "gpt-4"
        assert result.embedding_model == "openai-embedder"
        mock_client.request.assert_called_once_with(
            "GET",
            "http://localhost:8080/api/v2/config/memory/semantic",
            timeout=None,
        )

    def test_get_semantic_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.get_semantic_memory_config()

    def test_update_semantic_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Semantic memory configuration updated"}
        )
        result = config.update_semantic_memory_config(
            enabled=True,
            database="postgres-db",
            llm_model="gpt-4",
            embedding_model="openai-embedder",
            ingestion_trigger_messages=10,
            ingestion_trigger_age_seconds=3600,
        )
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        assert call_args[0] == (
            "PUT",
            "http://localhost:8080/api/v2/config/memory/semantic",
        )
        body = call_args[1]["json"]
        assert body["enabled"] is True
        assert body["database"] == "postgres-db"
        assert body["llm_model"] == "gpt-4"
        assert body["embedding_model"] == "openai-embedder"
        assert body["ingestion_trigger_messages"] == 10
        assert body["ingestion_trigger_age_seconds"] == 3600

    def test_update_semantic_memory_config_partial(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Semantic memory configuration updated"}
        )
        result = config.update_semantic_memory_config(enabled=False)
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        body = call_args[1]["json"]
        assert body == {"enabled": False}

    def test_update_semantic_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.update_semantic_memory_config(enabled=True)

    def test_get_episodic_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {
                "long_term_memory": {
                    "embedder": "default-embedder",
                    "reranker": "default-reranker",
                    "vector_graph_store": "neo4j-store",
                    "enabled": True,
                },
                "short_term_memory": {
                    "llm_model": "default-llm",
                    "message_capacity": 100,
                    "enabled": True,
                },
                "enabled": True,
            }
        )
        result = config.get_episodic_memory_config()
        assert isinstance(result, EpisodicMemoryConfigResponse)
        assert result.enabled is True
        assert result.long_term_memory.embedder == "default-embedder"
        assert result.short_term_memory.llm_model == "default-llm"
        mock_client.request.assert_called_once_with(
            "GET",
            "http://localhost:8080/api/v2/config/memory/episodic",
            timeout=None,
        )

    def test_get_episodic_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.get_episodic_memory_config()

    def test_update_episodic_memory_config(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Episodic memory configuration updated"}
        )
        result = config.update_episodic_memory_config(
            enabled=False,
            long_term_memory_enabled=True,
            short_term_memory_enabled=True,
        )
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        assert call_args[0] == (
            "PUT",
            "http://localhost:8080/api/v2/config/memory/episodic",
        )
        body = call_args[1]["json"]
        assert body["enabled"] is False
        assert body["long_term_memory_enabled"] is True
        assert body["short_term_memory_enabled"] is True

    def test_update_episodic_memory_config_with_nested_specs(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Episodic memory configuration updated"}
        )
        ltm_spec = UpdateLongTermMemorySpec(
            embedder="new-embedder",
            reranker="new-reranker",
            vector_graph_store="new-store",
        )
        stm_spec = UpdateShortTermMemorySpec(
            llm_model="new-llm",
            message_capacity=200,
        )
        result = config.update_episodic_memory_config(
            long_term_memory=ltm_spec,
            short_term_memory=stm_spec,
        )
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        body = call_args[1]["json"]
        assert body["long_term_memory"]["embedder"] == "new-embedder"
        assert body["long_term_memory"]["reranker"] == "new-reranker"
        assert body["long_term_memory"]["vector_graph_store"] == "new-store"
        assert body["short_term_memory"]["llm_model"] == "new-llm"
        assert body["short_term_memory"]["message_capacity"] == 200

    def test_update_episodic_memory_config_partial(self, config, mock_client):
        mock_client.request.return_value = _mock_response(
            {"success": True, "message": "Episodic memory configuration updated"}
        )
        result = config.update_episodic_memory_config(enabled=False)
        assert isinstance(result, UpdateMemoryConfigResponse)
        assert result.success is True
        call_args = mock_client.request.call_args
        body = call_args[1]["json"]
        assert body == {"enabled": False}

    def test_update_episodic_memory_config_closed_client(self, config, mock_client):
        mock_client.closed = True
        with pytest.raises(RuntimeError, match="client has been closed"):
            config.update_episodic_memory_config(enabled=True)
