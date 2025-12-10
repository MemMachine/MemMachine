from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memmachine.common.errors import (
    ConfigurationError,
    InvalidArgumentError,
    ResourceNotFoundError,
)
from memmachine.main.memmachine import ALL_MEMORY_TYPES, MemoryType
from memmachine.server.api_v2.router import get_memmachine, load_v2_api_router
from memmachine.server.api_v2.service import _SessionData


@pytest.fixture
def mock_memmachine():
    memmachine = AsyncMock()
    return memmachine


@pytest.fixture
def client(mock_memmachine):
    app = FastAPI()
    load_v2_api_router(app)

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
    assert response.json()["org_id"] == "test_org"
    assert response.json()["project_id"] == "test_proj"
    assert response.json()["description"] == "A test project"
    assert response.json()["config"]["embedder"] == "openai"
    assert response.json()["config"]["reranker"] == "cohere"

    mock_memmachine.create_session.assert_awaited_once()
    call_args = mock_memmachine.create_session.call_args[1]
    assert call_args["session_key"] == "test_org/test_proj"
    assert call_args["description"] == "A test project"
    assert call_args["embedder_name"] == "openai"
    assert call_args["reranker_name"] == "cohere"

    mock_memmachine.create_session.reset_mock()
    mock_memmachine.create_session.side_effect = InvalidArgumentError(
        "mock invalid argument"
    )
    response = client.post("/api/v2/projects", json=payload)
    assert response.status_code == 422
    assert "mock invalid argument" in response.json()["detail"]

    mock_memmachine.create_session.reset_mock()
    mock_memmachine.create_session.side_effect = ConfigurationError("mock config error")
    response = client.post("/api/v2/projects", json=payload)
    assert response.status_code == 500
    assert "mock config error" in response.json()["detail"]["internal_error"]

    mock_memmachine.create_session.reset_mock()
    mock_memmachine.create_session.side_effect = ValueError(
        "Session test_org/test_proj already exists"
    )
    response = client.post("/api/v2/projects", json=payload)
    assert response.status_code == 409
    assert response.json()["detail"]["message"] == "Project already exists"


def test_get_project(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_session_info = MagicMock()
    mock_session_info.description = "A test project"
    mock_session_info.episode_memory_conf.long_term_memory.embedder = "openai"
    mock_session_info.episode_memory_conf.long_term_memory.reranker = "cohere"
    mock_memmachine.get_session.return_value = mock_session_info

    response = client.post("/api/v2/projects/get", json=payload)
    assert response.status_code == 200
    assert response.json()["org_id"] == "test_org"
    assert response.json()["project_id"] == "test_proj"
    assert response.json()["description"] == "A test project"
    assert response.json()["config"]["embedder"] == "openai"
    assert response.json()["config"]["reranker"] == "cohere"

    mock_memmachine.get_session.assert_awaited_once()
    call_args = mock_memmachine.get_session.call_args[1]
    assert call_args["session_key"] == "test_org/test_proj"

    mock_memmachine.get_session.reset_mock()
    mock_memmachine.get_session.side_effect = Exception("Some other error")
    response = client.post("/api/v2/projects/get", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "Internal server error"

    mock_memmachine.get_session.reset_mock()
    mock_memmachine.get_session.side_effect = None
    mock_memmachine.get_session.return_value = None
    response = client.post("/api/v2/projects/get", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Project does not exist"


def test_get_episode_count(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_memmachine.episodes_count.return_value = 42

    response = client.post("/api/v2/projects/episode_count/get", json=payload)

    assert response.status_code == 200
    assert response.json()["count"] == 42

    mock_memmachine.episodes_count.assert_awaited_once()
    call_args = mock_memmachine.episodes_count.call_args[1]
    assert call_args["session_data"].session_key == "test_org/test_proj"

    mock_memmachine.episodes_count.reset_mock()
    mock_memmachine.episodes_count.side_effect = Exception("Some error")
    response = client.post("/api/v2/projects/episode_count/get", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "Internal server error"


def test_list_projects(client, mock_memmachine):
    mock_memmachine.search_sessions.return_value = [
        "org1/proj1",
        "org2/proj2",
        "not-project-session",
    ]

    response = client.post("/api/v2/projects/list")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0] == {"org_id": "org1", "project_id": "proj1"}
    assert data[1] == {"org_id": "org2", "project_id": "proj2"}

    mock_memmachine.search_sessions.assert_awaited_once()


def test_delete_project(client, mock_memmachine):
    payload = {"org_id": "test_org", "project_id": "test_proj"}

    # Success
    response = client.post("/api/v2/projects/delete", json=payload)
    assert response.status_code == 204
    mock_memmachine.delete_session.assert_awaited_once()

    # Not found
    mock_memmachine.delete_session.reset_mock()
    mock_memmachine.delete_session.side_effect = ValueError(
        "Session test_org/test_proj does not exist"
    )
    response = client.post("/api/v2/projects/delete", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"]["message"] == "Project does not exist"

    # Error
    mock_memmachine.delete_session.reset_mock()
    mock_memmachine.delete_session.side_effect = Exception("Delete error")
    response = client.post("/api/v2/projects/delete", json=payload)
    assert response.status_code == 500
    assert "Unable to delete project" in response.json()["detail"]["message"]


def test_add_memories(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("memmachine.server.api_v2.router._add_messages_to") as mock_add_messages:
        mock_add_messages.return_value = [{"status": "ok", "uid": "123"}]

        # Generic add
        response = client.post("/api/v2/memories", json=payload)
        assert response.status_code == 200
        assert response.json() == {"results": [{"uid": "123"}]}
        mock_add_messages.assert_awaited_once()
        call_args = mock_add_messages.call_args[1]
        assert call_args["target_memories"] == ALL_MEMORY_TYPES

        # Episodic add
        mock_add_messages.reset_mock()
        mock_add_messages.return_value = [{"status": "ok", "uid": "123"}]
        payload["types"] = [MemoryType.Episodic.value]
        response = client.post("/api/v2/memories", json=payload)
        assert response.status_code == 200
        assert response.json() == {"results": [{"uid": "123"}]}
        call_args = mock_add_messages.call_args[1]
        assert call_args["target_memories"] == [MemoryType.Episodic]

        # Semantic add
        mock_add_messages.reset_mock()
        mock_add_messages.return_value = [{"status": "ok", "uid": "123"}]
        payload["types"] = [MemoryType.Semantic.value]
        response = client.post("/api/v2/memories", json=payload)
        assert response.status_code == 200
        assert response.json() == {"results": [{"uid": "123"}]}
        call_args = mock_add_messages.call_args[1]
        assert call_args["target_memories"] == [MemoryType.Semantic]


def test_search_memories(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "query": "hello",
    }

    with patch(
        "memmachine.server.api_v2.router._search_target_memories"
    ) as mock_search:
        mock_search.return_value = {
            "status": 0,
            "content": {"episodic_memory": [], "semantic_memory": []},
        }

        # Success
        response = client.post("/api/v2/memories/search", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "status": 0,
            "content": {"episodic_memory": [], "semantic_memory": []},
        }
        mock_search.assert_awaited_once()

        # Invalid argument
        mock_search.reset_mock()
        mock_search.side_effect = ValueError("Invalid arg")
        response = client.post("/api/v2/memories/search", json=payload)
        assert response.status_code == 422
        assert "invalid argument" in response.json()["detail"]["message"]

        # Not found
        mock_search.reset_mock()
        mock_search.side_effect = RuntimeError("No session info found for session")
        response = client.post("/api/v2/memories/search", json=payload)
        assert response.status_code == 404
        assert response.json()["detail"]["message"] == "Project does not exist"


def test_list_memories(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "type": "episodic",
        "page_size": 10,
        "page_num": 1,
    }

    mock_results = MagicMock()
    mock_results.episodic_memory = [{"id": "1", "content": "mem1"}]
    mock_results.semantic_memory = None
    mock_memmachine.list_search.return_value = mock_results

    response = client.post("/api/v2/memories/list", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["content"]["episodic_memory"] == [{"id": "1", "content": "mem1"}]
    assert data["content"]["semantic_memory"] == []

    mock_memmachine.list_search.assert_awaited_once()


def test_delete_episodic_memory(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "episodic_id": "ep1",
    }

    # Success
    response = client.post("/api/v2/memories/episodic/delete", json=payload)
    assert response.status_code == 204
    mock_memmachine.delete_episodes.assert_awaited_once()

    # Invalid arg
    mock_memmachine.delete_episodes.reset_mock()
    mock_memmachine.delete_episodes.side_effect = ValueError("Invalid")
    response = client.post("/api/v2/memories/episodic/delete", json=payload)
    assert response.status_code == 422
    assert "invalid argument" in response.json()["detail"]["message"]

    # Not found
    mock_memmachine.delete_episodes.reset_mock()
    mock_memmachine.delete_episodes.side_effect = ResourceNotFoundError("Not found")
    response = client.post("/api/v2/memories/episodic/delete", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"]["message"] == "Not found"

    # Error
    mock_memmachine.delete_episodes.reset_mock()
    mock_memmachine.delete_episodes.side_effect = Exception("Error")
    response = client.post("/api/v2/memories/episodic/delete", json=payload)
    assert response.status_code == 500
    assert "Unable to delete episodic memory" in response.json()["detail"]["message"]


def test_delete_episodic_memories(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "episodic_id": "ep1",
        "episodic_ids": ["ep3", "ep1"],
    }

    # Success
    response = client.post("/api/v2/memories/episodic/delete", json=payload)
    assert response.status_code == 204
    mock_memmachine.delete_episodes.assert_awaited_once_with(
        session_data=_SessionData(
            org_id="test_org",
            project_id="test_proj",
        ),
        episode_ids=["ep1", "ep3"],
    )


def test_delete_episodic_memories_empty(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }
    response = client.post("/api/v2/memories/episodic/delete", json=payload)
    assert response.status_code == 422
    assert "At least one episodic ID" in response.json()["detail"][0]["msg"]


def test_delete_semantic_memory(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "semantic_id": "sem1",
    }

    # Success
    response = client.post("/api/v2/memories/semantic/delete", json=payload)
    assert response.status_code == 204
    mock_memmachine.delete_features.assert_awaited_once_with(feature_ids=["sem1"])

    # Invalid arg
    mock_memmachine.delete_features.reset_mock()
    mock_memmachine.delete_features.side_effect = ValueError("Invalid")
    response = client.post("/api/v2/memories/semantic/delete", json=payload)
    assert response.status_code == 422
    assert "invalid argument" in response.json()["detail"]["message"]

    # Not found
    mock_memmachine.delete_features.reset_mock()
    mock_memmachine.delete_features.side_effect = ResourceNotFoundError("Not found")
    response = client.post("/api/v2/memories/semantic/delete", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"]["message"] == "Not found"

    # Error
    mock_memmachine.delete_features.reset_mock()
    mock_memmachine.delete_features.side_effect = Exception("Error")
    response = client.post("/api/v2/memories/semantic/delete", json=payload)
    assert response.status_code == 500
    assert "Unable to delete semantic memory" in response.json()["detail"]["message"]


def test_delete_semantic_memories(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "semantic_id": "sem1",
        "semantic_ids": ["sem3", "sem1"],
    }

    # Success
    response = client.post("/api/v2/memories/semantic/delete", json=payload)
    assert response.status_code == 204
    mock_memmachine.delete_features.assert_awaited_once_with(
        feature_ids=["sem1", "sem3"]
    )


def test_delete_semantic_memories_empty(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }
    response = client.post("/api/v2/memories/semantic/delete", json=payload)
    assert response.status_code == 422
    assert "At least one semantic ID" in response.json()["detail"][0]["msg"]


def test_metrics(client):
    response = client.get("/api/v2/metrics")
    assert response.status_code == 200


def test_health_check(client):
    response = client.get("/api/v2/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "memmachine"}
