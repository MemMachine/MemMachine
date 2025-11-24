"""Test profile memory initialization"""

from unittest.mock import MagicMock

import pytest
import yaml
from memmachine.episodic_memory_manager import EpisodicMemoryManager
from memmachine.semantic_memory.semantic_session_resource import SessionIdManager

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    SemanticCategory,
)
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager
from memmachine.server.app import initialize_resource


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mocks all external dependencies for initialize_resource."""
    mock_llm = MagicMock(spec=LanguageModel)
    mock_embedder = MagicMock(spec=Embedder)

    mock_metrics_builder = MagicMock()
    mock_semantic_service = MagicMock(spec=SemanticService)
    mock_semantic_session_manager = MagicMock(spec=SemanticSessionManager)
    mock_episodic_manager = MagicMock(spec=EpisodicMemoryManager)
    mock_import_module = MagicMock()
    mock_sqlalchemy_engine = MagicMock()
    mock_semantic_storage = MagicMock()

    mock_embedder_builder = MagicMock()
    mock_embedder_builder.build.return_value = mock_embedder

    monkeypatch.setattr("memmachine.server.app.EmbedderBuilder", mock_embedder_builder)
    monkeypatch.setattr(
        "memmachine.server.app.MetricsFactoryBuilder",
        mock_metrics_builder,
    )
    monkeypatch.setattr("memmachine.server.app.SemanticService", mock_semantic_service)
    monkeypatch.setattr(
        "memmachine.server.app.SemanticSessionManager",
        mock_semantic_session_manager,
    )
    monkeypatch.setattr(
        "memmachine.server.app.EpisodicMemoryManager",
        mock_episodic_manager,
    )
    monkeypatch.setattr("memmachine.server.app.import_module", mock_import_module)

    # Mock SQLAlchemy and storage
    mock_sqlalchemy_create_engine = MagicMock(return_value=mock_sqlalchemy_engine)
    monkeypatch.setattr(
        "memmachine.server.app.create_async_engine",
        mock_sqlalchemy_create_engine,
    )

    mock_storage_class = MagicMock(return_value=mock_semantic_storage)
    monkeypatch.setattr(
        "memmachine.server.app.SqlAlchemyPgVectorSemanticStorage",
        mock_storage_class,
    )

    # Mock the create_episodic_memory_manager class method
    mock_episodic_manager.create_episodic_memory_manager.return_value = (
        mock_episodic_manager
    )

    # Mock the prompt module with SEMANTIC_TYPE
    mock_prompt_module = MagicMock()
    mock_prompt_module.UPDATE_PROMPT = "test update prompt"
    mock_prompt_module.CONSOLIDATION_PROMPT = "test consolidation prompt"
    mock_prompt_module.SEMANTIC_TYPE = SemanticCategory(
        name="profile",
        tags=set(),
        prompt=RawSemanticPrompt(
            update_prompt="test update prompt",
            consolidation_prompt="test consolidation prompt",
        ),
    )
    mock_import_module.return_value = mock_prompt_module

    return {
        "llm_model": mock_llm,
        "embedder_builder": mock_embedder_builder,
        "metrics_builder": mock_metrics_builder,
        "semantic_service": mock_semantic_service,
        "semantic_session_manager": mock_semantic_session_manager,
        "episodic_manager": mock_episodic_manager,
        "import_module": mock_import_module,
        "prompt_module": mock_prompt_module,
        "sqlalchemy_engine": mock_sqlalchemy_engine,
        "semantic_storage": mock_semantic_storage,
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
                "provider": "openai",
                "model_name": "gpt-3",
                "api_key": "TEST_API_KEY_VAR",
            },
        },
        "embedder": {
            "test_embedder": {
                "provider": "openai",
                "config": {
                    "model_name": "text-embedding-ada-002",
                    "api_key": "TEST_EMBED_KEY_VAR",
                },
            },
        },
        "storage": {
            "test_db": {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "database": "test_db",
                "password": "TEST_DB_PASS_VAR",
            },
        },
    }

    config_file = tmp_path / "test_config.yml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.dump(config_content, f)
    return str(config_file)


@pytest.mark.asyncio
async def test_initialize_resource_success(
    mock_dependencies,
    mock_config_file,
    monkeypatch,
):
    """Tests that initialize_resource successfully creates and returns
    EpisodicMemoryManager, SemanticSessionManager, and SessionIdManager instances
    with correct configurations.
    """
    # Call the function under test
    (
        episodic_manager,
        semantic_session_manager,
        session_id_manager,
    ) = await initialize_resource(mock_config_file)

    # Assert that the correct instances were returned
    assert episodic_manager == mock_dependencies["episodic_manager"]
    assert (
        semantic_session_manager
        == mock_dependencies["semantic_session_manager"].return_value
    )
    assert isinstance(session_id_manager, SessionIdManager)

    # Verify that dependencies were called correctly
    mock_dependencies[
        "episodic_manager"
    ].create_episodic_memory_manager.assert_called_once_with(mock_config_file)
    mock_dependencies["semantic_service"].assert_called_once()
    mock_dependencies["semantic_session_manager"].assert_called_once()

    # Verify prompt module was imported
    mock_dependencies["import_module"].assert_called_with(
        ".prompt.test_prompt",
        "memmachine.server",
    )

    # Verify LLM builder was called correctly
    llm_builder_args = mock_dependencies["llm_builder"].build.call_args[0]
    assert llm_builder_args[0] == "openai"
    llm_builder_args = llm_builder_args[1]
    assert llm_builder_args["api_key"] == "TEST_API_KEY_VAR"
    assert llm_builder_args["model_name"] == "gpt-3"
    assert llm_builder_args["metrics_factory_id"] == "prometheus"

    # Verify embedder builder was called correctly
    embedder_builder_args = mock_dependencies["embedder_builder"].build.call_args[0]
    assert embedder_builder_args[0] == "openai"
    embedder_builder_args = embedder_builder_args[1]
    assert embedder_builder_args["api_key"] == "TEST_EMBED_KEY_VAR"
    assert embedder_builder_args["metrics_factory_id"] == "prometheus"
    assert embedder_builder_args["model_name"] == "text-embedding-ada-002"

    # Verify semantic service was created (we can't easily inspect Pydantic BaseModel params)
    assert mock_dependencies["semantic_service"].called

    # Verify semantic session manager was created with correct parameters
    session_manager_call = mock_dependencies["semantic_session_manager"].call_args
    assert session_manager_call is not None
    assert "semantic_service" in session_manager_call.kwargs
    assert "history_storage" in session_manager_call.kwargs
    assert (
        session_manager_call.kwargs["semantic_service"]
        == mock_dependencies["semantic_service"].return_value
    )
    assert (
        session_manager_call.kwargs["history_storage"]
        == mock_dependencies["semantic_storage"]
    )
