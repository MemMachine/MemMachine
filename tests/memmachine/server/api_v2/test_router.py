from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from memmachine.common.api.spec import ContentType, Episode, SearchResult
from memmachine.common.episode_store.episode_model import EpisodeType
from memmachine.common.errors import (
    ConfigurationError,
    InvalidArgumentError,
    ResourceNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from memmachine.main.memmachine import ALL_MEMORY_TYPES, MemoryType
from memmachine.server.api_v2.router import RestError, get_memmachine
from memmachine.server.api_v2.service import _SessionData
from memmachine.server.app import MemMachineAPI


@pytest.fixture
def mock_memmachine():
    memmachine = AsyncMock()
    return memmachine


@pytest.fixture
def client(mock_memmachine):
    app = MemMachineAPI()
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
    assert call_args["user_conf"].long_term_memory.embedder == "openai"
    assert call_args["user_conf"].long_term_memory.reranker == "cohere"

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
    response_detail = response.json()["detail"]
    assert "mock config error" in response_detail["internal_error"]
    assert "Traceback (most recent call last)" in response_detail["trace"]

    mock_memmachine.create_session.reset_mock()
    mock_memmachine.create_session.side_effect = SessionAlreadyExistsError(
        "test_org/test_proj"
    )
    response = client.post("/api/v2/projects", json=payload)
    assert response.status_code == 409
    response_detail = response.json()["detail"]
    assert "already exists" in response_detail["message"]
    assert response_detail["trace"] == ""


def test_create_project_with_invalid_name(client):
    response = client.post(
        "/api/v2/projects",
        json={
            "org_id": "test_org/abc",
            "project_id": "test_proj",
        },
    )
    assert response.status_code == 422
    response_detail = response.json()["detail"]
    assert response_detail["trace"] == ""
    error_message = response_detail["message"]
    assert "Invalid request payload: org_id" in error_message
    assert "found: 'test_org/abc'" in error_message


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
    mock_memmachine.delete_session.side_effect = SessionNotFoundError(
        "test_org/test_proj"
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


def test_add_memories_episode_type_forwarded(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "messages": [
            {"role": "user", "content": "hello", "episode_type": "message"},
            {"role": "user", "content": "world"},
        ],
    }

    mock_memmachine.add_episodes.return_value = ["ep-1", "ep-2"]

    response = client.post("/api/v2/memories", json=payload)
    assert response.status_code == 200
    assert response.json() == {"results": [{"uid": "ep-1"}, {"uid": "ep-2"}]}

    mock_memmachine.add_episodes.assert_awaited_once()
    call_kwargs = mock_memmachine.add_episodes.call_args[1]
    assert call_kwargs["target_memories"] == ALL_MEMORY_TYPES

    episode_entries = call_kwargs["episode_entries"]
    assert len(episode_entries) == 2
    assert episode_entries[0].episode_type == EpisodeType.MESSAGE
    assert episode_entries[1].episode_type is None


def test_search_memories(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "query": "hello",
        "agent_mode": True,
    }

    with patch(
        "memmachine.server.api_v2.router._search_target_memories"
    ) as mock_search:
        mock_search.return_value = SearchResult.model_validate(
            {
                "status": 0,
                "content": {
                    "episodic_memory": {
                        "long_term_memory": {"episodes": []},
                        "short_term_memory": {
                            "episodes": [],
                            "episode_summary": [],
                        },
                    },
                    "semantic_memory": [],
                },
            }
        )

        # Success
        response = client.post("/api/v2/memories/search", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "status": 0,
            "content": {
                "episodic_memory": {
                    "long_term_memory": {"episodes": []},
                    "short_term_memory": {
                        "episodes": [],
                        "episode_summary": [],
                    },
                },
                "semantic_memory": [],
            },
        }
        mock_search.assert_awaited_once()
        await_args = mock_search.await_args
        assert await_args is not None
        search_call = cast(dict[str, Any], await_args.kwargs)
        assert search_call["spec"].agent_mode is True

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


def test_search_memories_with_set_metadata(client, mock_memmachine):
    """set_metadata in the request body is forwarded to _search_target_memories."""
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "query": "hello",
        "set_metadata": {"user_id": "user123", "tenant": "acme"},
    }

    with patch(
        "memmachine.server.api_v2.router._search_target_memories"
    ) as mock_search:
        mock_search.return_value = SearchResult.model_validate(
            {
                "status": 0,
                "content": {
                    "episodic_memory": None,
                    "semantic_memory": [],
                },
            }
        )

        response = client.post("/api/v2/memories/search", json=payload)
        assert response.status_code == 200

        mock_search.assert_awaited_once()
        # The spec passed to _search_target_memories must carry set_metadata
        _, call_kwargs = mock_search.call_args
        assert call_kwargs["spec"].set_metadata == {
            "user_id": "user123",
            "tenant": "acme",
        }


def test_list_memories(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "type": "episodic",
        "page_size": 10,
        "page_num": 1,
    }

    mock_results = MagicMock()
    mock_results.episodic_memory = [
        Episode(
            uid="1",
            content="mem1",
            session_key="test_org/test_proj",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            producer_id="user",
            producer_role="user",
            produced_for_id=None,
            sequence_num=0,
            episode_type=EpisodeType.MESSAGE,
            content_type=ContentType.STRING,
            filterable_metadata=None,
            metadata=None,
        )
    ]
    mock_results.semantic_memory = None
    mock_memmachine.list_search.return_value = mock_results

    response = client.post("/api/v2/memories/list", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["content"]["episodic_memory"][0]["uid"] == "1"
    assert data["content"]["episodic_memory"][0]["content"] == "mem1"
    assert "semantic_memory" not in data["content"]

    mock_memmachine.list_search.assert_awaited_once()


def test_list_memories_with_set_metadata(client, mock_memmachine):
    """set_metadata in the request body is forwarded to list_search via _list_target_memories."""
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_metadata": {"user_id": "user456", "region": "us-east"},
    }

    mock_results = MagicMock()
    mock_results.episodic_memory = None
    mock_results.semantic_memory = None
    mock_memmachine.list_search.return_value = mock_results

    response = client.post("/api/v2/memories/list", json=payload)
    assert response.status_code == 200

    mock_memmachine.list_search.assert_awaited_once()
    call_kwargs = mock_memmachine.list_search.call_args[1]
    assert call_kwargs["set_metadata"] == {"user_id": "user456", "region": "us-east"}


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

    # Session does not exist
    mock_memmachine.delete_episodes.reset_mock()
    mock_memmachine.delete_episodes.side_effect = SessionNotFoundError(
        "test_org/test_proj"
    )
    response = client.post("/api/v2/memories/episodic/delete", json=payload)
    assert response.status_code == 404
    assert (
        response.json()["detail"]["message"]
        == "Session 'test_org/test_proj' does not exist."
    )

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
    response_detail = response.json()["detail"]
    assert "At least one episodic ID" in response_detail["message"]


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
    response_detail = response.json()["detail"]
    assert "At least one semantic ID" in response_detail["message"]


def test_metrics(client):
    response = client.get("/api/v2/metrics")
    assert response.status_code == 200


def test_health_check(client):
    response = client.get("/api/v2/health")
    assert response.status_code == 200
    resp_json = response.json()
    assert resp_json["status"] == "healthy"
    assert resp_json["service"] == "memmachine"
    assert len(resp_json["version"]) > 0


def test_rest_error():
    err = RestError(422, "sample", RuntimeError("for test"))
    assert err.status_code == 422
    assert isinstance(err.detail, dict)
    detail = cast(dict[str, object], err.detail)
    assert detail["message"] == "sample"
    assert detail["code"] == 422
    assert err.payload is not None
    assert err.payload.exception == "RuntimeError"
    assert err.payload.internal_error == "for test"
    assert err.payload.trace == "RuntimeError: for test"


def test_rest_error_without_exception():
    err = RestError(404, "resource not found")
    assert err.status_code == 404
    assert err.detail == "resource not found"
    assert err.payload is None


def test_add_feature(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "test_set",
        "category_name": "preferences",
        "tag": "food",
        "feature": "favorite_food",
        "value": "pizza",
    }

    mock_memmachine.add_feature.return_value = "feature_123"

    response = client.post("/api/v2/memories/semantic/feature", json=payload)
    assert response.status_code == 201
    assert response.json()["feature_id"] == "feature_123"

    mock_memmachine.add_feature.assert_awaited_once()
    call_args = mock_memmachine.add_feature.call_args[1]
    assert call_args["set_id"] == "test_set"
    assert call_args["category_name"] == "preferences"
    assert call_args["tag"] == "food"
    assert call_args["feature"] == "favorite_food"
    assert call_args["value"] == "pizza"


def test_add_feature_with_metadata_and_citations(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "test_set",
        "category_name": "preferences",
        "tag": "food",
        "feature": "favorite_food",
        "value": "pizza",
        "feature_metadata": {"source": "conversation"},
        "citations": ["ep1", "ep2"],
    }

    mock_memmachine.add_feature.return_value = "feature_456"

    response = client.post("/api/v2/memories/semantic/feature", json=payload)
    assert response.status_code == 201
    assert response.json()["feature_id"] == "feature_456"

    call_args = mock_memmachine.add_feature.call_args[1]
    assert call_args["feature_metadata"] == {"source": "conversation"}
    assert call_args["citations"] == ["ep1", "ep2"]


def test_add_feature_invalid_arg(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "test_set",
        "category_name": "preferences",
        "tag": "food",
        "feature": "favorite_food",
        "value": "pizza",
    }

    mock_memmachine.add_feature.side_effect = ValueError("Invalid set_id")
    response = client.post("/api/v2/memories/semantic/feature", json=payload)
    assert response.status_code == 422
    assert "invalid argument" in response.json()["detail"]["message"]


def test_add_feature_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "test_set",
        "category_name": "preferences",
        "tag": "food",
        "feature": "favorite_food",
        "value": "pizza",
    }

    mock_memmachine.add_feature.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/feature", json=payload)
    assert response.status_code == 500
    assert "Unable to add feature" in response.json()["detail"]["message"]


def test_get_feature(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "feature_123",
    }

    mock_feature = MagicMock()
    mock_feature.set_id = "test_set"
    mock_feature.category = "preferences"
    mock_feature.tag = "food"
    mock_feature.feature_name = "favorite_food"
    mock_feature.value = "pizza"
    mock_feature.metadata = MagicMock()
    mock_feature.metadata.id = "feature_123"
    mock_feature.metadata.citations = None
    mock_feature.metadata.other = None

    mock_memmachine.get_feature.return_value = mock_feature

    response = client.post("/api/v2/memories/semantic/feature/get", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["set_id"] == "test_set"
    assert data["category"] == "preferences"
    assert data["tag"] == "food"
    assert data["feature_name"] == "favorite_food"
    assert data["value"] == "pizza"

    mock_memmachine.get_feature.assert_awaited_once()
    call_args = mock_memmachine.get_feature.call_args[1]
    assert call_args["feature_id"] == "feature_123"
    assert call_args["load_citations"] is False


def test_get_feature_with_citations(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "feature_123",
        "load_citations": True,
    }

    mock_feature = MagicMock()
    mock_feature.set_id = "test_set"
    mock_feature.category = "preferences"
    mock_feature.tag = "food"
    mock_feature.feature_name = "favorite_food"
    mock_feature.value = "pizza"
    mock_feature.metadata = MagicMock()
    mock_feature.metadata.id = "feature_123"
    mock_feature.metadata.citations = ["ep1", "ep2"]
    mock_feature.metadata.other = None

    mock_memmachine.get_feature.return_value = mock_feature

    response = client.post("/api/v2/memories/semantic/feature/get", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["citations"] == ["ep1", "ep2"]

    call_args = mock_memmachine.get_feature.call_args[1]
    assert call_args["load_citations"] is True


def test_get_feature_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "nonexistent",
    }

    mock_memmachine.get_feature.return_value = None

    response = client.post("/api/v2/memories/semantic/feature/get", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Feature not found"


def test_get_feature_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "feature_123",
    }

    mock_memmachine.get_feature.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/feature/get", json=payload)
    assert response.status_code == 500
    assert "Unable to get feature" in response.json()["detail"]["message"]


def test_update_feature(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "feature_123",
        "value": "sushi",
    }

    response = client.post("/api/v2/memories/semantic/feature/update", json=payload)
    assert response.status_code == 204

    mock_memmachine.update_feature.assert_awaited_once()
    call_args = mock_memmachine.update_feature.call_args[1]
    assert call_args["feature_id"] == "feature_123"
    assert call_args["value"] == "sushi"
    assert call_args["category_name"] is None
    assert call_args["feature"] is None
    assert call_args["tag"] is None
    assert call_args["metadata"] is None


def test_update_feature_all_fields(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "feature_123",
        "category_name": "new_preferences",
        "tag": "cuisine",
        "feature": "top_food",
        "value": "sushi",
        "metadata": {"updated": "true"},
    }

    response = client.post("/api/v2/memories/semantic/feature/update", json=payload)
    assert response.status_code == 204

    call_args = mock_memmachine.update_feature.call_args[1]
    assert call_args["category_name"] == "new_preferences"
    assert call_args["tag"] == "cuisine"
    assert call_args["feature"] == "top_food"
    assert call_args["value"] == "sushi"
    assert call_args["metadata"] == {"updated": "true"}


def test_update_feature_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "nonexistent",
        "value": "sushi",
    }

    mock_memmachine.update_feature.side_effect = ResourceNotFoundError(
        "Feature not found"
    )
    response = client.post("/api/v2/memories/semantic/feature/update", json=payload)
    assert response.status_code == 404
    assert "Feature not found" in response.json()["detail"]["message"]


def test_update_feature_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "feature_id": "feature_123",
        "value": "sushi",
    }

    mock_memmachine.update_feature.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/feature/update", json=payload)
    assert response.status_code == 500
    assert "Unable to update feature" in response.json()["detail"]["message"]


# --- Semantic Set Type API Tests ---


def test_create_semantic_set_type(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id", "session_id"],
        "is_org_level": False,
        "name": "User Sessions",
        "description": "Set type for user sessions",
    }

    mock_memmachine.create_semantic_set_type.return_value = "set_type_123"

    response = client.post("/api/v2/memories/semantic/set_type", json=payload)
    assert response.status_code == 201
    assert response.json()["set_type_id"] == "set_type_123"

    mock_memmachine.create_semantic_set_type.assert_awaited_once()
    call_args = mock_memmachine.create_semantic_set_type.call_args[1]
    assert call_args["session_data"].session_key == "test_org/test_proj"
    assert call_args["metadata_tags"] == ["user_id", "session_id"]
    assert call_args["is_org_level"] is False
    assert call_args["name"] == "User Sessions"
    assert call_args["description"] == "Set type for user sessions"


def test_create_semantic_set_type_minimal(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id"],
    }

    mock_memmachine.create_semantic_set_type.return_value = "set_type_456"

    response = client.post("/api/v2/memories/semantic/set_type", json=payload)
    assert response.status_code == 201
    assert response.json()["set_type_id"] == "set_type_456"


def test_create_semantic_set_type_invalid_arg(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id"],
    }

    mock_memmachine.create_semantic_set_type.side_effect = ValueError("Invalid tags")
    response = client.post("/api/v2/memories/semantic/set_type", json=payload)
    assert response.status_code == 422
    assert "invalid argument" in response.json()["detail"]["message"]


def test_create_semantic_set_type_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id"],
    }

    mock_memmachine.create_semantic_set_type.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/set_type", json=payload)
    assert response.status_code == 500
    assert "Unable to create set type" in response.json()["detail"]["message"]


def test_list_semantic_set_types(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_set_type1 = MagicMock()
    mock_set_type1.id = "st_1"
    mock_set_type1.is_org_level = False
    mock_set_type1.tags = ["user_id"]
    mock_set_type1.name = "User Set"
    mock_set_type1.description = "User-scoped sets"

    mock_set_type2 = MagicMock()
    mock_set_type2.id = "st_2"
    mock_set_type2.is_org_level = True
    mock_set_type2.tags = []
    mock_set_type2.name = None
    mock_set_type2.description = None

    mock_memmachine.list_semantic_set_type.return_value = [
        mock_set_type1,
        mock_set_type2,
    ]

    response = client.post("/api/v2/memories/semantic/set_type/list", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["set_types"]) == 2
    assert data["set_types"][0]["id"] == "st_1"
    assert data["set_types"][0]["is_org_level"] is False
    assert data["set_types"][0]["tags"] == ["user_id"]
    assert data["set_types"][0]["name"] == "User Set"
    assert data["set_types"][1]["id"] == "st_2"
    assert data["set_types"][1]["is_org_level"] is True


def test_list_semantic_set_types_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_memmachine.list_semantic_set_type.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/set_type/list", json=payload)
    assert response.status_code == 500
    assert "Unable to list set types" in response.json()["detail"]["message"]


def test_delete_semantic_set_type(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_type_id": "st_123",
    }

    response = client.post("/api/v2/memories/semantic/set_type/delete", json=payload)
    assert response.status_code == 204

    mock_memmachine.delete_semantic_set_type.assert_awaited_once_with(
        set_type_id="st_123"
    )


def test_delete_semantic_set_type_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_type_id": "nonexistent",
    }

    mock_memmachine.delete_semantic_set_type.side_effect = ResourceNotFoundError(
        "Set type not found"
    )
    response = client.post("/api/v2/memories/semantic/set_type/delete", json=payload)
    assert response.status_code == 404
    assert "Set type not found" in response.json()["detail"]["message"]


def test_delete_semantic_set_type_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_type_id": "st_123",
    }

    mock_memmachine.delete_semantic_set_type.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/set_type/delete", json=payload)
    assert response.status_code == 500
    assert "Unable to delete set type" in response.json()["detail"]["message"]


def test_get_semantic_set_id(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id"],
        "is_org_level": False,
        "set_metadata": {"user_id": "user123"},
    }

    mock_memmachine.semantic_get_set_id.return_value = "mem_user_set_abc123"

    response = client.post("/api/v2/memories/semantic/set_id/get", json=payload)
    assert response.status_code == 200
    assert response.json()["set_id"] == "mem_user_set_abc123"

    mock_memmachine.semantic_get_set_id.assert_awaited_once()
    call_args = mock_memmachine.semantic_get_set_id.call_args[1]
    assert call_args["session_data"].session_key == "test_org/test_proj"
    assert call_args["metadata_tags"] == ["user_id"]
    assert call_args["is_org_level"] is False
    assert call_args["set_metadata"] == {"user_id": "user123"}


def test_get_semantic_set_id_with_set_metadata(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id"],
        "set_metadata": {"user_id": "user123"},
    }

    mock_memmachine.semantic_get_set_id.return_value = "mem_user_set_legacy"

    response = client.post("/api/v2/memories/semantic/set_id/get", json=payload)
    assert response.status_code == 200
    assert response.json()["set_id"] == "mem_user_set_legacy"

    call_args = mock_memmachine.semantic_get_set_id.call_args[1]
    assert call_args["set_metadata"] == {"user_id": "user123"}


def test_get_semantic_set_id_minimal(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": [],
    }

    mock_memmachine.semantic_get_set_id.return_value = "mem_project_set_def456"

    response = client.post("/api/v2/memories/semantic/set_id/get", json=payload)
    assert response.status_code == 200
    assert response.json()["set_id"] == "mem_project_set_def456"


def test_get_semantic_set_id_invalid_arg(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id"],
    }

    mock_memmachine.semantic_get_set_id.side_effect = ValueError("Invalid metadata")
    response = client.post("/api/v2/memories/semantic/set_id/get", json=payload)
    assert response.status_code == 422
    assert "invalid argument" in response.json()["detail"]["message"]


def test_get_semantic_set_id_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "metadata_tags": ["user_id"],
    }

    mock_memmachine.semantic_get_set_id.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/set_id/get", json=payload)
    assert response.status_code == 500
    assert "Unable to get set ID" in response.json()["detail"]["message"]


def test_list_semantic_set_ids(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_set1 = MagicMock()
    mock_set1.id = "mem_user_set_abc"
    mock_set1.is_org_level = False
    mock_set1.tags = ["user_id"]
    mock_set1.name = "User Set"
    mock_set1.description = None

    mock_set2 = MagicMock()
    mock_set2.id = "mem_project_set_def"
    mock_set2.is_org_level = True
    mock_set2.tags = []
    mock_set2.name = None
    mock_set2.description = None

    mock_memmachine.semantic_list_set_ids.return_value = [mock_set1, mock_set2]

    response = client.post("/api/v2/memories/semantic/set_id/list", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["sets"]) == 2
    assert data["sets"][0]["id"] == "mem_user_set_abc"
    assert data["sets"][0]["is_org_level"] is False
    assert data["sets"][0]["tags"] == ["user_id"]
    assert data["sets"][1]["id"] == "mem_project_set_def"


def test_list_semantic_set_ids_with_metadata_filter(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_metadata": {"user_id": "user123"},
    }

    mock_memmachine.semantic_list_set_ids.return_value = []

    response = client.post("/api/v2/memories/semantic/set_id/list", json=payload)
    assert response.status_code == 200

    call_args = mock_memmachine.semantic_list_set_ids.call_args[1]
    assert call_args["set_metadata"] == {"user_id": "user123"}


def test_list_semantic_set_ids_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_memmachine.semantic_list_set_ids.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/set_id/list", json=payload)
    assert response.status_code == 500
    assert "Unable to list sets" in response.json()["detail"]["message"]


def test_configure_semantic_set(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "mem_user_set_abc",
        "embedder_name": "openai-embed",
        "llm_name": "gpt-4",
    }

    response = client.post("/api/v2/memories/semantic/set/configure", json=payload)
    assert response.status_code == 204

    mock_memmachine.configure_semantic_set.assert_awaited_once()
    call_args = mock_memmachine.configure_semantic_set.call_args[1]
    assert call_args["set_id"] == "mem_user_set_abc"
    assert call_args["embedder_name"] == "openai-embed"
    assert call_args["llm_name"] == "gpt-4"


def test_configure_semantic_set_partial(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "mem_user_set_abc",
        "embedder_name": "openai-embed",
    }

    response = client.post("/api/v2/memories/semantic/set/configure", json=payload)
    assert response.status_code == 204

    call_args = mock_memmachine.configure_semantic_set.call_args[1]
    assert call_args["embedder_name"] == "openai-embed"
    assert call_args["llm_name"] is None


def test_configure_semantic_set_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "nonexistent",
        "embedder_name": "openai-embed",
    }

    mock_memmachine.configure_semantic_set.side_effect = ResourceNotFoundError(
        "Set not found"
    )
    response = client.post("/api/v2/memories/semantic/set/configure", json=payload)
    assert response.status_code == 404
    assert "Set not found" in response.json()["detail"]["message"]


def test_configure_semantic_set_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "mem_user_set_abc",
        "embedder_name": "openai-embed",
    }

    mock_memmachine.configure_semantic_set.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/set/configure", json=payload)
    assert response.status_code == 500
    assert "Unable to configure set" in response.json()["detail"]["message"]


# --- Semantic Category Tests ---


def test_get_semantic_category(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
    }

    mock_category = MagicMock()
    mock_category.id = "cat_123"
    mock_category.name = "preferences"
    mock_category.prompt = "Extract user preferences"
    mock_category.description = "Category for user preferences"

    mock_memmachine.semantic_get_category.return_value = mock_category

    response = client.post("/api/v2/memories/semantic/category/get", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "cat_123"
    assert data["name"] == "preferences"
    assert data["prompt"] == "Extract user preferences"
    assert data["description"] == "Category for user preferences"


def test_get_semantic_category_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "nonexistent",
    }

    mock_memmachine.semantic_get_category.return_value = None

    response = client.post("/api/v2/memories/semantic/category/get", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_semantic_category_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
    }

    mock_memmachine.semantic_get_category.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/category/get", json=payload)
    assert response.status_code == 500
    assert "Unable to get category" in response.json()["detail"]["message"]


def test_add_semantic_category(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "set_123",
        "category_name": "preferences",
        "prompt": "Extract user preferences",
        "description": "Category for user preferences",
    }

    mock_memmachine.semantic_add_category.return_value = "cat_456"

    response = client.post("/api/v2/memories/semantic/category", json=payload)
    assert response.status_code == 201
    assert response.json()["category_id"] == "cat_456"

    mock_memmachine.semantic_add_category.assert_awaited_once()
    call_args = mock_memmachine.semantic_add_category.call_args[1]
    assert call_args["set_id"] == "set_123"
    assert call_args["category_name"] == "preferences"
    assert call_args["prompt"] == "Extract user preferences"
    assert call_args["description"] == "Category for user preferences"


def test_add_semantic_category_minimal(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "set_123",
        "category_name": "preferences",
        "prompt": "Extract preferences",
    }

    mock_memmachine.semantic_add_category.return_value = "cat_789"

    response = client.post("/api/v2/memories/semantic/category", json=payload)
    assert response.status_code == 201
    assert response.json()["category_id"] == "cat_789"


def test_add_semantic_category_invalid_arg(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "set_123",
        "category_name": "preferences",
        "prompt": "Extract preferences",
    }

    mock_memmachine.semantic_add_category.side_effect = ValueError("Invalid category")
    response = client.post("/api/v2/memories/semantic/category", json=payload)
    assert response.status_code == 422
    assert "invalid argument" in response.json()["detail"]["message"]


def test_add_semantic_category_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "set_123",
        "category_name": "preferences",
        "prompt": "Extract preferences",
    }

    mock_memmachine.semantic_add_category.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/category", json=payload)
    assert response.status_code == 500
    assert "Unable to add category" in response.json()["detail"]["message"]


def test_add_semantic_category_template(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_type_id": "st_123",
        "category_name": "preferences",
        "prompt": "Extract user preferences",
        "description": "Template for user preferences",
    }

    mock_memmachine.semantic_add_category_template.return_value = "cat_template_456"

    response = client.post("/api/v2/memories/semantic/category/template", json=payload)
    assert response.status_code == 201
    assert response.json()["category_id"] == "cat_template_456"

    mock_memmachine.semantic_add_category_template.assert_awaited_once()
    call_args = mock_memmachine.semantic_add_category_template.call_args[1]
    assert call_args["set_type_id"] == "st_123"
    assert call_args["category_name"] == "preferences"
    assert call_args["prompt"] == "Extract user preferences"
    assert call_args["description"] == "Template for user preferences"


def test_add_semantic_category_template_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_type_id": "st_123",
        "category_name": "preferences",
        "prompt": "Extract preferences",
    }

    mock_memmachine.semantic_add_category_template.side_effect = Exception(
        "Database error"
    )
    response = client.post("/api/v2/memories/semantic/category/template", json=payload)
    assert response.status_code == 500
    assert "Unable to add category template" in response.json()["detail"]["message"]


def test_list_semantic_category_templates(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_type_id": "st_123",
    }

    mock_cat1 = MagicMock()
    mock_cat1.id = "cat_1"
    mock_cat1.name = "preferences"
    mock_cat1.origin_type = "set_type"
    mock_cat1.origin_id = "st_123"
    mock_cat1.inherited = False

    mock_cat2 = MagicMock()
    mock_cat2.id = "cat_2"
    mock_cat2.name = "facts"
    mock_cat2.origin_type = "set"
    mock_cat2.origin_id = "set_456"
    mock_cat2.inherited = True

    mock_memmachine.semantic_list_category_templates.return_value = [
        mock_cat1,
        mock_cat2,
    ]

    response = client.post(
        "/api/v2/memories/semantic/category/template/list", json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["categories"]) == 2
    assert data["categories"][0]["id"] == "cat_1"
    assert data["categories"][0]["name"] == "preferences"
    assert data["categories"][0]["origin_type"] == "set_type"
    assert data["categories"][0]["inherited"] is False
    assert data["categories"][1]["id"] == "cat_2"
    assert data["categories"][1]["inherited"] is True


def test_list_semantic_category_templates_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_type_id": "st_123",
    }

    mock_memmachine.semantic_list_category_templates.side_effect = Exception(
        "Database error"
    )
    response = client.post(
        "/api/v2/memories/semantic/category/template/list", json=payload
    )
    assert response.status_code == 500
    assert "Unable to list category templates" in response.json()["detail"]["message"]


def test_disable_semantic_category(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "set_123",
        "category_name": "preferences",
    }

    response = client.post("/api/v2/memories/semantic/category/disable", json=payload)
    assert response.status_code == 204

    mock_memmachine.semantic_disable_category.assert_awaited_once()
    call_args = mock_memmachine.semantic_disable_category.call_args[1]
    assert call_args["set_id"] == "set_123"
    assert call_args["category_name"] == "preferences"


def test_disable_semantic_category_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "set_123",
        "category_name": "nonexistent",
    }

    mock_memmachine.semantic_disable_category.side_effect = ResourceNotFoundError(
        "Category not found"
    )
    response = client.post("/api/v2/memories/semantic/category/disable", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]["message"]


def test_disable_semantic_category_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "set_id": "set_123",
        "category_name": "preferences",
    }

    mock_memmachine.semantic_disable_category.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/category/disable", json=payload)
    assert response.status_code == 500
    assert "Unable to disable category" in response.json()["detail"]["message"]


def test_get_semantic_category_set_ids(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
    }

    mock_memmachine.semantic_get_category_set_ids.return_value = [
        "set_1",
        "set_2",
        "set_3",
    ]

    response = client.post(
        "/api/v2/memories/semantic/category/set_ids/get", json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert data["set_ids"] == ["set_1", "set_2", "set_3"]


def test_get_semantic_category_set_ids_empty(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
    }

    mock_memmachine.semantic_get_category_set_ids.return_value = []

    response = client.post(
        "/api/v2/memories/semantic/category/set_ids/get", json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert data["set_ids"] == []


def test_get_semantic_category_set_ids_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
    }

    mock_memmachine.semantic_get_category_set_ids.side_effect = Exception(
        "Database error"
    )
    response = client.post(
        "/api/v2/memories/semantic/category/set_ids/get", json=payload
    )
    assert response.status_code == 500
    assert "Unable to get category set IDs" in response.json()["detail"]["message"]


def test_delete_semantic_category(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
    }

    response = client.post("/api/v2/memories/semantic/category/delete", json=payload)
    assert response.status_code == 204

    mock_memmachine.semantic_delete_category.assert_awaited_once()
    call_args = mock_memmachine.semantic_delete_category.call_args[1]
    assert call_args["category_id"] == "cat_123"


def test_delete_semantic_category_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "nonexistent",
    }

    mock_memmachine.semantic_delete_category.side_effect = ResourceNotFoundError(
        "Category not found"
    )
    response = client.post("/api/v2/memories/semantic/category/delete", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]["message"]


def test_delete_semantic_category_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
    }

    mock_memmachine.semantic_delete_category.side_effect = Exception("Database error")
    response = client.post("/api/v2/memories/semantic/category/delete", json=payload)
    assert response.status_code == 500
    assert "Unable to delete category" in response.json()["detail"]["message"]


# --- Semantic Tag Tests ---


def test_add_semantic_tag(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
        "tag_name": "food_preference",
        "tag_description": "User food preferences",
    }

    mock_memmachine.semantic_add_tag_to_category.return_value = "tag_456"

    response = client.post("/api/v2/memories/semantic/category/tag", json=payload)
    assert response.status_code == 201
    assert response.json()["tag_id"] == "tag_456"

    mock_memmachine.semantic_add_tag_to_category.assert_awaited_once()
    call_args = mock_memmachine.semantic_add_tag_to_category.call_args[1]
    assert call_args["category_id"] == "cat_123"
    assert call_args["tag_name"] == "food_preference"
    assert call_args["tag_description"] == "User food preferences"


def test_add_semantic_tag_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "nonexistent",
        "tag_name": "food_preference",
        "tag_description": "User food preferences",
    }

    mock_memmachine.semantic_add_tag_to_category.side_effect = ResourceNotFoundError(
        "Category not found"
    )
    response = client.post("/api/v2/memories/semantic/category/tag", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]["message"]


def test_add_semantic_tag_invalid_arg(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
        "tag_name": "food_preference",
        "tag_description": "User food preferences",
    }

    mock_memmachine.semantic_add_tag_to_category.side_effect = ValueError("Invalid tag")
    response = client.post("/api/v2/memories/semantic/category/tag", json=payload)
    assert response.status_code == 422
    assert "invalid argument" in response.json()["detail"]["message"]


def test_add_semantic_tag_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "category_id": "cat_123",
        "tag_name": "food_preference",
        "tag_description": "User food preferences",
    }

    mock_memmachine.semantic_add_tag_to_category.side_effect = Exception(
        "Database error"
    )
    response = client.post("/api/v2/memories/semantic/category/tag", json=payload)
    assert response.status_code == 500
    assert "Unable to add tag" in response.json()["detail"]["message"]


def test_delete_semantic_tag(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "tag_id": "tag_123",
    }

    response = client.post(
        "/api/v2/memories/semantic/category/tag/delete", json=payload
    )
    assert response.status_code == 204

    mock_memmachine.semantic_delete_tag.assert_awaited_once()
    call_args = mock_memmachine.semantic_delete_tag.call_args[1]
    assert call_args["tag_id"] == "tag_123"


def test_delete_semantic_tag_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "tag_id": "nonexistent",
    }

    mock_memmachine.semantic_delete_tag.side_effect = ResourceNotFoundError(
        "Tag not found"
    )
    response = client.post(
        "/api/v2/memories/semantic/category/tag/delete", json=payload
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]["message"]


def test_delete_semantic_tag_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "tag_id": "tag_123",
    }

    mock_memmachine.semantic_delete_tag.side_effect = Exception("Database error")
    response = client.post(
        "/api/v2/memories/semantic/category/tag/delete", json=payload
    )
    assert response.status_code == 500
    assert "Unable to delete tag" in response.json()["detail"]["message"]


# --- Episodic Memory Configuration Tests ---


def test_get_episodic_memory_config(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_session_info = MagicMock()
    mock_session_info.episode_memory_conf.enabled = True
    mock_session_info.episode_memory_conf.long_term_memory_enabled = True
    mock_session_info.episode_memory_conf.short_term_memory_enabled = False

    mock_memmachine.get_session.return_value = mock_session_info

    response = client.post("/api/v2/memory/episodic/config/get", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["long_term_memory_enabled"] is True
    assert data["short_term_memory_enabled"] is False

    mock_memmachine.get_session.assert_awaited_once_with(
        session_key="test_org/test_proj"
    )


def test_get_episodic_memory_config_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "nonexistent",
    }

    mock_memmachine.get_session.return_value = None

    response = client.post("/api/v2/memory/episodic/config/get", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_episodic_memory_config_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_memmachine.get_session.side_effect = Exception("Database error")
    response = client.post("/api/v2/memory/episodic/config/get", json=payload)
    assert response.status_code == 500
    assert (
        "Unable to get episodic memory config" in response.json()["detail"]["message"]
    )


def test_configure_episodic_memory(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "enabled": True,
        "long_term_memory_enabled": False,
        "short_term_memory_enabled": True,
    }

    response = client.post("/api/v2/memory/episodic/config", json=payload)
    assert response.status_code == 204

    mock_memmachine.update_session_episodic_config.assert_awaited_once()
    call_args = mock_memmachine.update_session_episodic_config.call_args[1]
    assert call_args["session_key"] == "test_org/test_proj"
    assert call_args["enabled"] is True
    assert call_args["long_term_memory_enabled"] is False
    assert call_args["short_term_memory_enabled"] is True


def test_configure_episodic_memory_partial(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "enabled": False,
    }

    response = client.post("/api/v2/memory/episodic/config", json=payload)
    assert response.status_code == 204

    call_args = mock_memmachine.update_session_episodic_config.call_args[1]
    assert call_args["enabled"] is False
    assert call_args["long_term_memory_enabled"] is None
    assert call_args["short_term_memory_enabled"] is None


def test_configure_episodic_memory_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "nonexistent",
        "enabled": False,
    }

    mock_memmachine.update_session_episodic_config.side_effect = SessionNotFoundError(
        "test_org/nonexistent"
    )
    response = client.post("/api/v2/memory/episodic/config", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]["message"]


def test_configure_episodic_memory_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "enabled": False,
    }

    mock_memmachine.update_session_episodic_config.side_effect = Exception(
        "Database error"
    )
    response = client.post("/api/v2/memory/episodic/config", json=payload)
    assert response.status_code == 500
    assert "Unable to configure episodic memory" in response.json()["detail"]["message"]


# --- Short-Term Memory Configuration Tests ---


def test_get_short_term_memory_config(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_session_info = MagicMock()
    mock_session_info.episode_memory_conf.short_term_memory_enabled = True

    mock_memmachine.get_session.return_value = mock_session_info

    response = client.post(
        "/api/v2/memory/episodic/short_term/config/get", json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True

    mock_memmachine.get_session.assert_awaited_once_with(
        session_key="test_org/test_proj"
    )


def test_get_short_term_memory_config_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "nonexistent",
    }

    mock_memmachine.get_session.return_value = None

    response = client.post(
        "/api/v2/memory/episodic/short_term/config/get", json=payload
    )
    assert response.status_code == 404


def test_get_short_term_memory_config_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_memmachine.get_session.side_effect = Exception("Database error")
    response = client.post(
        "/api/v2/memory/episodic/short_term/config/get", json=payload
    )
    assert response.status_code == 500
    assert (
        "Unable to get short-term memory config" in response.json()["detail"]["message"]
    )


def test_configure_short_term_memory(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "enabled": True,
    }

    response = client.post("/api/v2/memory/episodic/short_term/config", json=payload)
    assert response.status_code == 204

    mock_memmachine.update_session_episodic_config.assert_awaited_once()
    call_args = mock_memmachine.update_session_episodic_config.call_args[1]
    assert call_args["session_key"] == "test_org/test_proj"
    assert call_args["short_term_memory_enabled"] is True


def test_configure_short_term_memory_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "nonexistent",
        "enabled": False,
    }

    mock_memmachine.update_session_episodic_config.side_effect = SessionNotFoundError(
        "test_org/nonexistent"
    )
    response = client.post("/api/v2/memory/episodic/short_term/config", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]["message"]


def test_configure_short_term_memory_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "enabled": False,
    }

    mock_memmachine.update_session_episodic_config.side_effect = Exception(
        "Database error"
    )
    response = client.post("/api/v2/memory/episodic/short_term/config", json=payload)
    assert response.status_code == 500
    assert (
        "Unable to configure short-term memory" in response.json()["detail"]["message"]
    )


# --- Long-Term Memory Configuration Tests ---


def test_get_long_term_memory_config(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_session_info = MagicMock()
    mock_session_info.episode_memory_conf.long_term_memory_enabled = True

    mock_memmachine.get_session.return_value = mock_session_info

    response = client.post("/api/v2/memory/episodic/long_term/config/get", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True

    mock_memmachine.get_session.assert_awaited_once_with(
        session_key="test_org/test_proj"
    )


def test_get_long_term_memory_config_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "nonexistent",
    }

    mock_memmachine.get_session.return_value = None

    response = client.post("/api/v2/memory/episodic/long_term/config/get", json=payload)
    assert response.status_code == 404


def test_get_long_term_memory_config_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
    }

    mock_memmachine.get_session.side_effect = Exception("Database error")
    response = client.post("/api/v2/memory/episodic/long_term/config/get", json=payload)
    assert response.status_code == 500
    assert (
        "Unable to get long-term memory config" in response.json()["detail"]["message"]
    )


def test_configure_long_term_memory(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "enabled": True,
    }

    response = client.post("/api/v2/memory/episodic/long_term/config", json=payload)
    assert response.status_code == 204

    mock_memmachine.update_session_episodic_config.assert_awaited_once()
    call_args = mock_memmachine.update_session_episodic_config.call_args[1]
    assert call_args["session_key"] == "test_org/test_proj"
    assert call_args["long_term_memory_enabled"] is True


def test_configure_long_term_memory_not_found(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "nonexistent",
        "enabled": False,
    }

    mock_memmachine.update_session_episodic_config.side_effect = SessionNotFoundError(
        "test_org/nonexistent"
    )
    response = client.post("/api/v2/memory/episodic/long_term/config", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]["message"]


def test_configure_long_term_memory_error(client, mock_memmachine):
    payload = {
        "org_id": "test_org",
        "project_id": "test_proj",
        "enabled": False,
    }

    mock_memmachine.update_session_episodic_config.side_effect = Exception(
        "Database error"
    )
    response = client.post("/api/v2/memory/episodic/long_term/config", json=payload)
    assert response.status_code == 500
    assert (
        "Unable to configure long-term memory" in response.json()["detail"]["message"]
    )
