"""
Test profile memory initialization
"""

from unittest.mock import MagicMock

import pytest
import yaml

from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
)
from memmachine.profile_memory.profile_memory import ProfileMemory
from memmachine.server.app import initialize_resource


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mocks all external dependencies for initialize_resource."""
    mock_llm_builder = MagicMock()
    mock_embedder_builder = MagicMock()
    mock_metrics_builder = MagicMock()
    mock_profile_memory = MagicMock(spec=ProfileMemory)
    mock_episodic_manager = MagicMock(spec=EpisodicMemoryManager)
    mock_import_module = MagicMock()

    monkeypatch.setattr("memmachine.server.app.LanguageModelBuilder", mock_llm_builder)
    monkeypatch.setattr("memmachine.server.app.EmbedderBuilder", mock_embedder_builder)
    monkeypatch.setattr(
        "memmachine.server.app.MetricsFactoryBuilder", mock_metrics_builder
    )
    monkeypatch.setattr("memmachine.server.app.ProfileMemory", mock_profile_memory)
    monkeypatch.setattr(
        "memmachine.server.app.EpisodicMemoryManager", mock_episodic_manager
    )
    monkeypatch.setattr("memmachine.server.app.import_module", mock_import_module)

    # Mock the create_episodic_memory_manager class method
    mock_episodic_manager.create_episodic_memory_manager.return_value = (
        mock_episodic_manager
    )

    return {
        "llm_builder": mock_llm_builder,
        "embedder_builder": mock_embedder_builder,
        "metrics_builder": mock_metrics_builder,
        "profile_memory": mock_profile_memory,
        "episodic_manager": mock_episodic_manager,
        "import_module": mock_import_module,
    }


@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a temporary YAML config file for testing."""
    config_content = {
        "profile_memory": {
            "llm_model": "test_llm",
            "embedding_model": "test_embedder",
            "database": "test_db",
            "prompt": "test_prompt",
        },
        "model": {
            "test_llm": {
                "model_vendor": "openai",
                "model_name": "gpt-3",
                "api_key": "TEST_API_KEY_VAR",
            }
        },
        "embedder": {
            "test_embedder": {
                "name": "openai",
                "config": {
                    "model_name": "text-embedding-ada-002",
                    "api_key": "TEST_EMBED_KEY_VAR",
                },
            }
        },
        "storage": {
            "test_db": {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "database": "test_db",
                "password": "TEST_DB_PASS_VAR",
            }
        },
    }

    config_file = tmp_path / "test_config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)
    return str(config_file)


@pytest.mark.asyncio
async def test_initialize_resource_success(
    mock_dependencies, mock_config_file, monkeypatch
):
    """
    Tests that initialize_resource successfully creates and returns
    EpisodicMemoryManager and ProfileMemory instances with correct configurations.
    """

    # Call the function under test
    episodic_mgr, profile_mem = await initialize_resource(mock_config_file)

    # Assert that the correct instances were returned
    assert episodic_mgr == mock_dependencies["episodic_manager"]
    assert profile_mem == mock_dependencies["profile_memory"].return_value

    # Verify that dependencies were called correctly
    mock_dependencies[
        "episodic_manager"
    ].create_episodic_memory_manager.assert_called_once_with(mock_config_file)
    mock_dependencies["profile_memory"].assert_called_once()

    # Verify prompt module was imported
    mock_dependencies["import_module"].assert_called_with(
        ".prompt.test_prompt", "memmachine.server"
    )

    llm_builder_args = mock_dependencies["llm_builder"].build.call_args[0]
    assert llm_builder_args[0] == "openai"
    llm_builder_args = llm_builder_args[1]
    assert llm_builder_args["api_key"] == "TEST_API_KEY_VAR"
    assert llm_builder_args["model_name"] == "gpt-3"
    assert llm_builder_args["metrics_factory_id"] == "prometheus"

    embedder_builder_args = mock_dependencies["embedder_builder"].build.call_args[0]
    assert embedder_builder_args[0] == "openai"
    embedder_builder_args = embedder_builder_args[1]
    assert embedder_builder_args["api_key"] == "TEST_EMBED_KEY_VAR"
    assert embedder_builder_args["metrics_factory_id"] == "prometheus"
    assert embedder_builder_args["model_name"] == "text-embedding-ada-002"

    _, profile_kwargs = mock_dependencies["profile_memory"].call_args
    db_config = profile_kwargs["profile_storage"]._config
    assert db_config["host"] == "localhost"
    assert db_config["port"] == 5432
    assert db_config["user"] == "postgres"
    assert db_config["password"] == "TEST_DB_PASS_VAR"
    assert db_config["database"] == "test_db"

    # You could add more specific assertions here to check the arguments
    # passed to the builders and ProfileMemory constructor if needed.
