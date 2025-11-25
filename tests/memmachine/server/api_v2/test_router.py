from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memmachine.server.api_v2.router import get_memmachine, load_v2_api_router

app = FastAPI()
load_v2_api_router(app)


@pytest.fixture
def mock_memmachine():
    memmachine = AsyncMock()
    return memmachine


@pytest.fixture
def client(mock_memmachine):
    app.dependency_overrides[get_memmachine] = lambda: mock_memmachine

    with TestClient(app) as c:
        yield c

    app.dependency_overrides = {}


def test_create_project(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "description": "A test project",
        "config": {"embedder": "openai", "reranker": "cohere"},
    }

    mock_session = MagicMock()
    mock_session.episode_memory_conf.long_term_memory.embedder = "openai"
    mock_session.episode_memory_conf.long_term_memory.reranker = "cohere"

    mock_memmachine.create_session.return_value = mock_session

    response = client.post("/api/v2/projects", json=payload)

    assert response.status_code == 201

    mock_memmachine.create_session.assert_awaited_once()
    call_args = mock_memmachine.create_session.call_args[1]
    assert call_args["session_key"] == "test_org/test_proj"
    assert call_args["description"] == "A test project"
    assert call_args["embedder_name"] == "openai"
    assert call_args["reranker_name"] == "cohere"
