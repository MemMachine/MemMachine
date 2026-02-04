"""Tests for the configuration API router."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from memmachine.common.api.config_spec import ResourceStatus
from memmachine.common.configuration import SemanticMemoryConf
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.errors import (
    InvalidEmbedderError,
    InvalidLanguageModelError,
    InvalidRerankerError,
)
from memmachine.server.api_v2.config_router import get_config_service
from memmachine.server.api_v2.config_service import ConfigService
from memmachine.server.app import MemMachineAPI


@pytest.fixture
def mock_resource_manager():
    """Create a mock resource manager with mock managers."""
    resource_manager = MagicMock()

    # Mock embedder manager
    embedder_manager = MagicMock()
    embedder_manager.get_all_names.return_value = {"openai-embedder"}
    embedder_manager.get_resource_status.return_value = ResourceStatus.READY
    embedder_manager.get_resource_error.return_value = None
    embedder_manager.conf.openai = {"openai-embedder": MagicMock()}
    embedder_manager.conf.amazon_bedrock = {}
    embedder_manager.conf.sentence_transformer = {}
    resource_manager.embedder_manager = embedder_manager

    # Mock language model manager
    lm_manager = MagicMock()
    lm_manager.get_all_names.return_value = {"gpt-4"}
    lm_manager.get_resource_status.return_value = ResourceStatus.READY
    lm_manager.get_resource_error.return_value = None
    lm_manager.conf.openai_responses_language_model_confs = {"gpt-4": MagicMock()}
    lm_manager.conf.openai_chat_completions_language_model_confs = {}
    lm_manager.conf.amazon_bedrock_language_model_confs = {}
    resource_manager.language_model_manager = lm_manager

    # Mock reranker manager
    reranker_manager = MagicMock()
    reranker_manager.get_all_names.return_value = {"bm25-reranker"}
    reranker_manager.get_resource_status.return_value = ResourceStatus.READY
    reranker_manager.get_resource_error.return_value = None
    reranker_manager.conf.bm25 = {"bm25-reranker": MagicMock()}
    reranker_manager.conf.cohere = {}
    reranker_manager.conf.cross_encoder = {}
    reranker_manager.conf.amazon_bedrock = {}
    reranker_manager.conf.embedder = {}
    reranker_manager.conf.identity = {}
    reranker_manager.conf.rrf_hybrid = {}
    resource_manager.reranker_manager = reranker_manager

    # Mock database manager
    db_manager = MagicMock()
    db_manager.conf.neo4j_confs = {}
    db_manager.conf.relational_db_confs = {"sqlite-db": MagicMock(dialect="sqlite")}
    resource_manager.database_manager = db_manager

    # Mock config for persistence
    mock_config = MagicMock()
    mock_config.config_file_path = "/tmp/test_config.yml"

    # Use real memory config objects for update_memory_config tests
    mock_config.episodic_memory = EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(
            embedder="old-embedder",
            reranker="old-reranker",
            vector_graph_store="old-store",
        ),
        short_term_memory=ShortTermMemoryConfPartial(
            llm_model="old-model",
            message_capacity=500,
        ),
        enabled=True,
    )
    mock_config.semantic_memory = SemanticMemoryConf(
        database="old-db",
        config_database="old-db",
        llm_model="old-llm",
        embedding_model="old-embedder",
    )

    resource_manager.config = mock_config
    resource_manager.save_config = MagicMock()

    return resource_manager


@pytest.fixture
def mock_memmachine(mock_resource_manager):
    """Create a mock memmachine with a resource manager."""
    memmachine = MagicMock()
    memmachine.resource_manager = mock_resource_manager
    return memmachine


@pytest.fixture
def client(mock_memmachine, mock_resource_manager):
    """Create a test client with mocked dependencies."""
    app = MemMachineAPI(with_config_api=True)
    app.dependency_overrides[get_config_service] = lambda: ConfigService(
        mock_resource_manager
    )

    with TestClient(app) as c:
        yield c

    app.dependency_overrides = {}


def test_get_config(client, mock_resource_manager):
    """Test GET /api/v2/config returns resource status."""
    response = client.get("/api/v2/config")
    assert response.status_code == 200

    data = response.json()
    assert "resources" in data

    resources = data["resources"]
    assert len(resources["embedders"]) == 1
    assert resources["embedders"][0]["name"] == "openai-embedder"
    assert resources["embedders"][0]["status"] == "ready"

    assert len(resources["language_models"]) == 1
    assert resources["language_models"][0]["name"] == "gpt-4"
    assert resources["language_models"][0]["status"] == "ready"

    assert len(resources["rerankers"]) == 1
    assert resources["rerankers"][0]["name"] == "bm25-reranker"
    assert resources["rerankers"][0]["status"] == "ready"

    assert len(resources["databases"]) == 1
    assert resources["databases"][0]["name"] == "sqlite-db"
    assert resources["databases"][0]["provider"] == "sqlite"


def test_get_resources(client, mock_resource_manager):
    """Test GET /api/v2/config/resources returns resource status."""
    response = client.get("/api/v2/config/resources")
    assert response.status_code == 200

    data = response.json()
    assert "embedders" in data
    assert "language_models" in data
    assert "rerankers" in data
    assert "databases" in data


def test_get_config_with_failed_resource(client, mock_resource_manager):
    """Test that failed resources show error messages."""
    mock_resource_manager.embedder_manager.get_resource_status.return_value = (
        ResourceStatus.FAILED
    )
    mock_resource_manager.embedder_manager.get_resource_error.return_value = Exception(
        "API key invalid"
    )

    response = client.get("/api/v2/config")
    assert response.status_code == 200

    data = response.json()
    embedder = data["resources"]["embedders"][0]
    assert embedder["status"] == "failed"
    assert "API key invalid" in embedder["error"]


def test_add_embedder_success(client, mock_resource_manager):
    """Test POST /api/v2/config/resources/embedders creates a new embedder."""
    mock_resource_manager.embedder_manager.add_embedder_config = MagicMock()
    mock_resource_manager.embedder_manager.get_embedder = AsyncMock()

    payload = {
        "name": "new-embedder",
        "provider": "openai",
        "config": {
            "api_key": "test-key",
            "model": "text-embedding-3-small",
        },
    }

    response = client.post("/api/v2/config/resources/embedders", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["success"] is True
    assert data["status"] == "ready"


def test_add_embedder_failure(client, mock_resource_manager):
    """Test that embedder addition returns failure status when build fails."""
    mock_resource_manager.embedder_manager.add_embedder_config = MagicMock()
    mock_resource_manager.embedder_manager.get_embedder = AsyncMock(
        side_effect=Exception("Connection refused")
    )
    mock_resource_manager.embedder_manager.get_resource_error.return_value = Exception(
        "Connection refused"
    )

    payload = {
        "name": "new-embedder",
        "provider": "openai",
        "config": {
            "api_key": "test-key",
        },
    }

    response = client.post("/api/v2/config/resources/embedders", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["success"] is False
    assert data["status"] == "failed"
    assert "Connection refused" in data["error"]


def test_add_language_model_success(client, mock_resource_manager):
    """Test POST /api/v2/config/resources/language_models creates a new model."""
    mock_resource_manager.language_model_manager.add_language_model_config = MagicMock()
    mock_resource_manager.language_model_manager.get_language_model = AsyncMock()

    payload = {
        "name": "new-model",
        "provider": "openai-responses",
        "config": {
            "api_key": "test-key",
            "model": "gpt-4",
        },
    }

    response = client.post("/api/v2/config/resources/language_models", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["success"] is True
    assert data["status"] == "ready"


def test_delete_embedder_success(client, mock_resource_manager):
    """Test DELETE /api/v2/config/resources/embedders/{name} removes an embedder."""
    mock_resource_manager.embedder_manager.remove_embedder = MagicMock(
        return_value=True
    )

    response = client.delete("/api/v2/config/resources/embedders/openai-embedder")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "removed successfully" in data["message"]


def test_delete_embedder_not_found(client, mock_resource_manager):
    """Test DELETE returns 404 when embedder doesn't exist."""
    mock_resource_manager.embedder_manager.remove_embedder = MagicMock(
        return_value=False
    )

    response = client.delete("/api/v2/config/resources/embedders/nonexistent")
    assert response.status_code == 404


def test_delete_language_model_success(client, mock_resource_manager):
    """Test DELETE /api/v2/config/resources/language_models/{name} removes a model."""
    mock_resource_manager.language_model_manager.remove_language_model = MagicMock(
        return_value=True
    )

    response = client.delete("/api/v2/config/resources/language_models/gpt-4")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True


def test_retry_embedder_success(client, mock_resource_manager):
    """Test POST /api/v2/config/resources/embedders/{name}/retry retries build."""
    mock_resource_manager.embedder_manager.retry_build = AsyncMock()

    response = client.post("/api/v2/config/resources/embedders/openai-embedder/retry")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["status"] == "ready"


def test_retry_embedder_not_found(client, mock_resource_manager):
    """Test retry returns 404 when embedder doesn't exist."""
    mock_resource_manager.embedder_manager.retry_build = AsyncMock(
        side_effect=InvalidEmbedderError("nonexistent")
    )

    response = client.post("/api/v2/config/resources/embedders/nonexistent/retry")
    assert response.status_code == 404


def test_retry_language_model_success(client, mock_resource_manager):
    """Test POST /api/v2/config/resources/language_models/{name}/retry retries build."""
    mock_resource_manager.language_model_manager.retry_build = AsyncMock()

    response = client.post("/api/v2/config/resources/language_models/gpt-4/retry")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True


def test_retry_language_model_not_found(client, mock_resource_manager):
    """Test retry returns 404 when model doesn't exist."""
    mock_resource_manager.language_model_manager.retry_build = AsyncMock(
        side_effect=InvalidLanguageModelError(
            "Language model with name nonexistent not found."
        )
    )

    response = client.post("/api/v2/config/resources/language_models/nonexistent/retry")
    assert response.status_code == 404


def test_retry_reranker_success(client, mock_resource_manager):
    """Test POST /api/v2/config/resources/rerankers/{name}/retry retries build."""
    mock_resource_manager.reranker_manager.retry_build = AsyncMock()

    response = client.post("/api/v2/config/resources/rerankers/bm25-reranker/retry")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True


def test_retry_reranker_not_found(client, mock_resource_manager):
    """Test retry returns 404 when reranker doesn't exist."""
    mock_resource_manager.reranker_manager.retry_build = AsyncMock(
        side_effect=InvalidRerankerError("nonexistent")
    )

    response = client.post("/api/v2/config/resources/rerankers/nonexistent/retry")
    assert response.status_code == 404


def test_retry_reranker_failure(client, mock_resource_manager):
    """Test retry returns failure status when build fails again."""
    mock_resource_manager.reranker_manager.retry_build = AsyncMock(
        side_effect=Exception("Still failing")
    )
    mock_resource_manager.reranker_manager.get_resource_error.return_value = Exception(
        "Still failing"
    )

    response = client.post("/api/v2/config/resources/rerankers/bm25-reranker/retry")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is False
    assert data["status"] == "failed"


def test_add_embedder_persists_config(client, mock_resource_manager):
    """Test that adding an embedder persists the configuration to file."""
    mock_resource_manager.embedder_manager.add_embedder_config = MagicMock()
    mock_resource_manager.embedder_manager.get_embedder = AsyncMock()

    payload = {
        "name": "new-embedder",
        "provider": "openai",
        "config": {
            "api_key": "test-key",
            "model": "text-embedding-3-small",
        },
    }

    response = client.post("/api/v2/config/resources/embedders", json=payload)
    assert response.status_code == 201

    # Verify save_config was called
    mock_resource_manager.save_config.assert_called_once()


def test_add_language_model_persists_config(client, mock_resource_manager):
    """Test that adding a language model persists the configuration to file."""
    mock_resource_manager.language_model_manager.add_language_model_config = MagicMock()
    mock_resource_manager.language_model_manager.get_language_model = AsyncMock()

    payload = {
        "name": "new-model",
        "provider": "openai-responses",
        "config": {
            "api_key": "test-key",
            "model": "gpt-4",
        },
    }

    response = client.post("/api/v2/config/resources/language_models", json=payload)
    assert response.status_code == 201

    # Verify save_config was called
    mock_resource_manager.save_config.assert_called_once()


def test_delete_embedder_persists_config(client, mock_resource_manager):
    """Test that deleting an embedder persists the configuration to file."""
    mock_resource_manager.embedder_manager.remove_embedder = MagicMock(
        return_value=True
    )

    response = client.delete("/api/v2/config/resources/embedders/openai-embedder")
    assert response.status_code == 200

    # Verify save_config was called
    mock_resource_manager.save_config.assert_called_once()


def test_delete_language_model_persists_config(client, mock_resource_manager):
    """Test that deleting a language model persists the configuration to file."""
    mock_resource_manager.language_model_manager.remove_language_model = MagicMock(
        return_value=True
    )

    response = client.delete("/api/v2/config/resources/language_models/gpt-4")
    assert response.status_code == 200

    # Verify save_config was called
    mock_resource_manager.save_config.assert_called_once()


# --- Memory Configuration Update Tests ---


def test_update_memory_episodic_ltm(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory updates episodic LTM config."""
    payload = {
        "episodic_memory": {
            "long_term_memory": {
                "embedder": "new-embedder",
                "reranker": "new-reranker",
            }
        }
    }

    response = client.put("/api/v2/config/memory", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "embedder=new-embedder" in data["message"]
    assert "reranker=new-reranker" in data["message"]

    # Verify config was persisted
    mock_resource_manager.save_config.assert_called_once()


def test_update_memory_episodic_stm(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory updates episodic STM config."""
    payload = {
        "episodic_memory": {
            "short_term_memory": {
                "llm_model": "new-stm-model",
                "message_capacity": 1000,
            }
        }
    }

    response = client.put("/api/v2/config/memory", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "llm_model=new-stm-model" in data["message"]
    assert "message_capacity=1000" in data["message"]


def test_update_memory_semantic(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory updates semantic memory config."""
    payload = {
        "semantic_memory": {
            "database": "new-db",
            "llm_model": "new-llm",
            "embedding_model": "new-embedder",
        }
    }

    response = client.put("/api/v2/config/memory", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "database=new-db" in data["message"]
    assert "llm_model=new-llm" in data["message"]
    assert "embedding_model=new-embedder" in data["message"]

    mock_resource_manager.save_config.assert_called_once()


def test_update_memory_both_sections(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory updates both episodic and semantic."""
    payload = {
        "episodic_memory": {
            "long_term_memory": {"embedder": "new-embedder"},
        },
        "semantic_memory": {
            "llm_model": "new-llm",
        },
    }

    response = client.put("/api/v2/config/memory", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "embedder=new-embedder" in data["message"]
    assert "llm_model=new-llm" in data["message"]


def test_update_memory_empty_body_returns_400(client):
    """Test PUT /api/v2/config/memory returns 400 when both sections are null."""
    payload = {}

    response = client.put("/api/v2/config/memory", json=payload)
    assert response.status_code == 400


def test_update_memory_episodic_enabled_flags(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory can toggle enabled flags."""
    payload = {
        "episodic_memory": {
            "long_term_memory_enabled": False,
            "enabled": False,
        }
    }

    response = client.put("/api/v2/config/memory", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "long_term_memory_enabled=False" in data["message"]
    assert "enabled=False" in data["message"]


def test_update_memory_semantic_ingestion(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory can update ingestion settings."""
    payload = {
        "semantic_memory": {
            "ingestion_trigger_messages": 10,
            "ingestion_trigger_age_seconds": 600,
        }
    }

    response = client.put("/api/v2/config/memory", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "ingestion_trigger_messages=10" in data["message"]
    assert "ingestion_trigger_age=600s" in data["message"]


# --- Get Config with Memory Configuration Tests ---


def test_get_config_includes_memory_config(client, mock_resource_manager):
    """Test GET /api/v2/config returns memory configuration."""
    response = client.get("/api/v2/config")
    assert response.status_code == 200

    data = response.json()

    # Check resources are present
    assert "resources" in data

    # Check episodic_memory is present
    assert "episodic_memory" in data
    em = data["episodic_memory"]
    assert "long_term_memory" in em
    assert "short_term_memory" in em
    assert "enabled" in em

    # Check long_term_memory config
    ltm = em["long_term_memory"]
    assert ltm["embedder"] == "old-embedder"
    assert ltm["reranker"] == "old-reranker"
    assert ltm["vector_graph_store"] == "old-store"
    assert ltm["enabled"] is True

    # Check short_term_memory config
    stm = em["short_term_memory"]
    assert stm["llm_model"] == "old-model"
    assert stm["message_capacity"] == 500
    assert stm["enabled"] is True

    # Check semantic_memory is present
    assert "semantic_memory" in data
    sm = data["semantic_memory"]
    assert sm["database"] == "old-db"
    assert sm["llm_model"] == "old-llm"
    assert sm["embedding_model"] == "old-embedder"


# --- Long-Term Memory Configuration Tests ---


def test_get_long_term_memory_config(client, mock_resource_manager):
    """Test GET /api/v2/config/memory/episodic/long_term returns LTM config."""
    response = client.get("/api/v2/config/memory/episodic/long_term")
    assert response.status_code == 200

    data = response.json()
    assert data["embedder"] == "old-embedder"
    assert data["reranker"] == "old-reranker"
    assert data["vector_graph_store"] == "old-store"
    assert data["enabled"] is True


def test_update_long_term_memory_config(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic/long_term updates LTM config."""
    payload = {
        "embedder": "new-embedder",
        "reranker": "new-reranker",
    }

    response = client.put("/api/v2/config/memory/episodic/long_term", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "embedder=new-embedder" in data["message"]
    assert "reranker=new-reranker" in data["message"]

    # Verify config was persisted
    mock_resource_manager.save_config.assert_called()


def test_update_long_term_memory_config_with_enabled(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic/long_term can toggle enabled flag."""
    payload = {
        "enabled": False,
    }

    response = client.put("/api/v2/config/memory/episodic/long_term", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "long_term_memory_enabled=False" in data["message"]


def test_update_long_term_memory_config_all_fields(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic/long_term can update all fields."""
    payload = {
        "embedder": "new-embedder",
        "reranker": "new-reranker",
        "vector_graph_store": "new-store",
        "enabled": False,
    }

    response = client.put("/api/v2/config/memory/episodic/long_term", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "embedder=new-embedder" in data["message"]
    assert "reranker=new-reranker" in data["message"]
    assert "vector_graph_store=new-store" in data["message"]
    assert "long_term_memory_enabled=False" in data["message"]


# --- Short-Term Memory Configuration Tests ---


def test_get_short_term_memory_config(client, mock_resource_manager):
    """Test GET /api/v2/config/memory/episodic/short_term returns STM config."""
    response = client.get("/api/v2/config/memory/episodic/short_term")
    assert response.status_code == 200

    data = response.json()
    assert data["llm_model"] == "old-model"
    assert data["message_capacity"] == 500
    assert data["enabled"] is True


def test_update_short_term_memory_config(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic/short_term updates STM config."""
    payload = {
        "llm_model": "new-model",
        "message_capacity": 1000,
    }

    response = client.put("/api/v2/config/memory/episodic/short_term", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "llm_model=new-model" in data["message"]
    assert "message_capacity=1000" in data["message"]

    # Verify config was persisted
    mock_resource_manager.save_config.assert_called()


def test_update_short_term_memory_config_with_enabled(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic/short_term can toggle enabled flag."""
    payload = {
        "enabled": False,
    }

    response = client.put("/api/v2/config/memory/episodic/short_term", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "short_term_memory_enabled=False" in data["message"]


def test_update_short_term_memory_config_all_fields(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic/short_term can update all fields."""
    payload = {
        "llm_model": "new-model",
        "message_capacity": 2000,
        "enabled": False,
    }

    response = client.put("/api/v2/config/memory/episodic/short_term", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "llm_model=new-model" in data["message"]
    assert "message_capacity=2000" in data["message"]
    assert "short_term_memory_enabled=False" in data["message"]


def test_get_semantic_memory_config(client, mock_resource_manager):
    """Test GET /api/v2/config/memory/semantic returns semantic memory config."""
    response = client.get("/api/v2/config/memory/semantic")
    assert response.status_code == 200

    data = response.json()
    assert "enabled" in data
    assert "database" in data
    assert "llm_model" in data
    assert "embedding_model" in data


def test_update_semantic_memory_config(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/semantic updates semantic config."""
    payload = {
        "enabled": True,
        "database": "postgres-db",
    }

    response = client.put("/api/v2/config/memory/semantic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "enabled=True" in data["message"]
    assert "database=postgres-db" in data["message"]

    # Verify config was persisted
    mock_resource_manager.save_config.assert_called()


def test_update_semantic_memory_config_all_fields(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/semantic can update all fields."""
    payload = {
        "enabled": True,
        "database": "postgres-db",
        "llm_model": "gpt-4",
        "embedding_model": "openai-embedder",
        "ingestion_trigger_messages": 10,
        "ingestion_trigger_age_seconds": 3600,
    }

    response = client.put("/api/v2/config/memory/semantic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "enabled=True" in data["message"]
    assert "database=postgres-db" in data["message"]
    assert "llm_model=gpt-4" in data["message"]
    assert "embedding_model=openai-embedder" in data["message"]
    assert "ingestion_trigger_messages=10" in data["message"]
    assert "ingestion_trigger_age=3600s" in data["message"]


def test_update_semantic_memory_config_partial(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/semantic with partial fields."""
    payload = {
        "llm_model": "gpt-4o",
    }

    response = client.put("/api/v2/config/memory/semantic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "llm_model=gpt-4o" in data["message"]


def test_get_episodic_memory_config(client, mock_resource_manager):
    """Test GET /api/v2/config/memory/episodic returns episodic memory config."""
    response = client.get("/api/v2/config/memory/episodic")
    assert response.status_code == 200

    data = response.json()
    assert "long_term_memory" in data
    assert "short_term_memory" in data
    assert "enabled" in data
    # Check nested structures
    assert "embedder" in data["long_term_memory"]
    assert "reranker" in data["long_term_memory"]
    assert "llm_model" in data["short_term_memory"]
    assert "message_capacity" in data["short_term_memory"]


def test_update_episodic_memory_config(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic updates episodic config."""
    payload = {
        "enabled": False,
    }

    response = client.put("/api/v2/config/memory/episodic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "enabled=False" in data["message"]

    # Verify config was persisted
    mock_resource_manager.save_config.assert_called()


def test_update_episodic_memory_config_ltm_enabled(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic can toggle LTM enabled flag."""
    payload = {
        "long_term_memory_enabled": False,
    }

    response = client.put("/api/v2/config/memory/episodic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "long_term_memory_enabled=False" in data["message"]


def test_update_episodic_memory_config_stm_enabled(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic can toggle STM enabled flag."""
    payload = {
        "short_term_memory_enabled": False,
    }

    response = client.put("/api/v2/config/memory/episodic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "short_term_memory_enabled=False" in data["message"]


def test_update_episodic_memory_config_with_ltm_settings(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic can update LTM settings."""
    payload = {
        "long_term_memory": {
            "embedder": "new-embedder",
            "reranker": "new-reranker",
        },
    }

    response = client.put("/api/v2/config/memory/episodic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "embedder=new-embedder" in data["message"]
    assert "reranker=new-reranker" in data["message"]


def test_update_episodic_memory_config_with_stm_settings(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic can update STM settings."""
    payload = {
        "short_term_memory": {
            "llm_model": "new-llm",
            "message_capacity": 500,
        },
    }

    response = client.put("/api/v2/config/memory/episodic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "llm_model=new-llm" in data["message"]
    assert "message_capacity=500" in data["message"]


def test_update_episodic_memory_config_all_fields(client, mock_resource_manager):
    """Test PUT /api/v2/config/memory/episodic can update all fields."""
    payload = {
        "long_term_memory": {
            "embedder": "new-embedder",
            "reranker": "new-reranker",
            "vector_graph_store": "new-store",
        },
        "short_term_memory": {
            "llm_model": "new-llm",
            "message_capacity": 500,
        },
        "long_term_memory_enabled": True,
        "short_term_memory_enabled": True,
        "enabled": True,
    }

    response = client.put("/api/v2/config/memory/episodic", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
