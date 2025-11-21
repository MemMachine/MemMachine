from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memmachine.server.api_v2.router import (
    get_episodic_memory,
    get_global_config,
    get_semantic_memory_manager,
    get_session_manager,
    load_v2_api_router,
)

app = FastAPI()
load_v2_api_router(app)


@pytest.fixture
def mock_session_manager():
    manager = AsyncMock()
    return manager


@pytest.fixture
def mock_episodic_memory():
    memory = AsyncMock()
    return memory


@pytest.fixture
def mock_semantic_manager():
    manager = AsyncMock()
    semantic_session_mgr = AsyncMock()
    manager.get_semantic_session_manager.return_value = semantic_session_mgr
    return manager


@pytest.fixture
def client(mock_session_manager, mock_episodic_memory, mock_semantic_manager):
    app.dependency_overrides[get_session_manager] = lambda: mock_session_manager
    app.dependency_overrides[get_episodic_memory] = lambda: mock_episodic_memory
    app.dependency_overrides[get_semantic_memory_manager] = (
        lambda: mock_semantic_manager
    )

    mock_conf = MagicMock()
    mock_conf.episodic_memory.long_term_memory.vector_graph_store = "mock_vector_store"
    mock_conf.episodic_memory.short_term_memory.llm_model = "mock_llm_model"
    mock_conf.episodic_memory.short_term_memory.summary_prompt_system = (
        "You are a helpful assistant."
    )
    mock_conf.episodic_memory.short_term_memory.summary_prompt_user = "Based on the following episodes: {episodes}, and the previous summary: {summary}, please update the summary. Keep it under {max_length} characters."

    app.dependency_overrides[get_global_config] = lambda: mock_conf

    with TestClient(app) as c:
        yield c

    app.dependency_overrides = {}


def test_create_project(client, mock_session_manager):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "description": "A test project",
        "config": {"embedder": "openai", "reranker": "cohere"},
    }

    response = client.post("/api/v2/projects", json=payload)

    assert response.status_code == 200

    mock_session_manager.create_new_session.assert_awaited_once()
    call_args = mock_session_manager.create_new_session.call_args[1]
    assert call_args["session_key"] == "test_org/test_proj"
    assert call_args["description"] == "A test project"
