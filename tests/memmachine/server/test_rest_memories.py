from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from memmachine.server.app import app

"""
================================================================================
MemMachine REST API /v1/memories Endpoint Tests (Final Fix)

This version directly mocks the AsyncEpisodicMemory context manager to
resolve the 500 Internal Server Errors that occurred only during testing.
This is the most robust method for isolating the endpoint logic.
================================================================================
"""

# Create a single test client for all tests
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def mock_memory_managers(monkeypatch):
    """
    This fixture isolates the API from its dependencies.

    The key to this fix is patching `AsyncEpisodicMemory` where it is *used*
    (in the `app` module), not where it is defined. This ensures the mock
    is applied correctly during the test run.

    It also correctly configures a chain of mocks:
    - `MockAsyncEpisodicMemory` (replaces the class)
    - `mock_context_manager` (the return value of the class call)
    - `mock_inst` (the value yielded by the context manager)
    """
    import memmachine.server.app as app_module

    # 1. Mock the object that will be yielded by the context manager.
    #    This needs all the methods that are called on `inst` *inside*
    #    the `async with` block.
    mock_inst = MagicMock()
    mock_inst.add_memory_episode = AsyncMock(return_value=True)
    mock_inst.delete_data = AsyncMock()
    mock_inst.get_memory_context.return_value = MagicMock(group_id="g", session_id="s")
    mock_inst.query_memory = AsyncMock(return_value=([], [], ["EpisodicMemory"]))

    # 2. Create a mock async context manager that yields the mock instance.
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_inst

    # 3. Create a mock for the AsyncEpisodicMemory class itself.
    #    When `AsyncEpisodicMemory(inst)` is called in the app, this mock will
    #    be called instead, returning our mock context manager.
    MockAsyncEpisodicMemory = MagicMock(return_value=mock_context_manager)

    # 4. Mock the other dependencies.
    class DummyEpisodicMemoryManager:
        async def get_episodic_memory_instance(self, *args, **kwargs):
            return MagicMock()

    class DummyProfileMemory:
        async def add_persona_message(self, *args, **kwargs):
            pass

        async def semantic_search(self, *args, **kwargs):
            return []

    # 5. Apply all patches to the app module.
    monkeypatch.setattr(app_module, "episodic_memory", DummyEpisodicMemoryManager())
    monkeypatch.setattr(app_module, "profile_memory", DummyProfileMemory())
    # This is the crucial fix: patch the name in the module where it's looked
    # up.
    monkeypatch.setattr(app_module, "AsyncEpisodicMemory", MockAsyncEpisodicMemory)


# --- Base Payloads for DRY Tests ---


@pytest.fixture
def valid_post_payload():
    """Provides a valid, complete payload for POST requests."""
    return {
        "session": {
            "group_id": "group1",
            "agent_id": ["agent1", "agent2"],
            "user_id": ["user1", "user2"],
            "session_id": "session1",
        },
        "producer": "user1",
        "produced_for": "agent1",
        "episode_content": "A valid memory string.",
        "episode_type": "message",
        "metadata": {"source": "test-suite"},
    }


@pytest.fixture
def valid_query_payload():
    """Provides a valid, complete payload for query requests."""
    return {
        "session": {
            "group_id": "group1",
            "agent_id": ["agent1", "agent2"],
            "user_id": ["user1", "user2"],
            "session_id": "session1",
        },
        "query": "test",
    }


@pytest.fixture
def query_payload_without_session():
    """Provides a valid payload for query requests without session info."""
    return {
        "query": "test",
    }


@pytest.fixture
def valid_delete_payload():
    """Provides a valid payload for DELETE requests."""
    return {
        "session": {
            "group_id": "group1",
            "agent_id": ["agent1"],
            "user_id": ["user1"],
            "session_id": "session1",
        }
    }


@pytest.fixture
def valid_session_headers():
    """Provides valid headers representing a session."""
    return {
        "group-id": "group-hdr",
        "session-id": "session-hdr",
        "agent-id": "agent3,agent4",  # comma-separated values
        "user-id": "user3,user4",
    }


@pytest.fixture
def alias_session_headers():
    """Provides valid headers using alias names for session fields."""
    return {
        "group_id": "group-alias",
        "session_id": "session-alias",
        "agent_id": "agent5,agent6",
        "user_id": "user5,user6",
    }


@pytest.fixture
def valid_post_payload_without_session():
    """Provides a valid, complete payload for POST requests."""
    return {
        "producer": "user1",
        "produced_for": "agent1",
        "episode_content": "A valid memory string.",
        "episode_type": "message",
        "metadata": {"source": "test-suite"},
    }


# --- Tests for POST /v1/memories ---
# (No changes needed to the test functions themselves)


def test_post_memories_valid_string_content(valid_post_payload):
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


def test_post_memories_with_session_in_header(
    valid_post_payload_without_session, valid_session_headers
):
    response = client.post(
        "/v1/memories",
        json=valid_post_payload_without_session,
        headers=valid_session_headers,
    )
    assert response.status_code in (200, 201, 204)


def test_post_memories_with_session_in_alias_header(
    valid_post_payload_without_session, alias_session_headers
):
    response = client.post(
        "/v1/memories",
        json=valid_post_payload_without_session,
        headers=alias_session_headers,
    )
    assert response.status_code in (200, 201, 204)


def test_post_memories_without_session(valid_post_payload_without_session):
    response = client.post("/v1/memories", json=valid_post_payload_without_session)
    assert response.status_code == 200
    assert response.headers["session-id"] == "default"
    assert response.headers["group-id"] == "default"


def test_post_episodic_memories_valid_string_content(valid_post_payload):
    """
    Test episodic memory ingestion.
    """
    response = client.post("/v1/memories/episodic", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


def test_post_episodic_memories_with_session_in_header(
    valid_post_payload_without_session, valid_session_headers
):
    response = client.post(
        "/v1/memories/episodic",
        json=valid_post_payload_without_session,
        headers=valid_session_headers,
    )
    assert response.status_code in (200, 201, 204)


def test_post_episodic_memories_without_session(valid_post_payload_without_session):
    response = client.post(
        "/v1/memories/episodic", json=valid_post_payload_without_session
    )
    assert response.status_code == 200
    assert response.headers["session-id"] == "default"
    assert response.headers["group-id"] == "default"


def test_post_profile_memories_valid_string_content(valid_post_payload):
    """
    Test profile memory ingestion.
    """
    valid_post_payload["episode_type"] = "embedding"
    response = client.post("/v1/memories/profile", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


def test_post_profile_memories_with_session_in_header(
    valid_post_payload_without_session, valid_session_headers
):
    valid_post_payload_without_session["episode_type"] = "embedding"
    response = client.post(
        "/v1/memories/profile",
        json=valid_post_payload_without_session,
        headers=valid_session_headers,
    )
    assert response.status_code in (200, 201, 204)


def test_post_profile_memories_without_session(valid_post_payload_without_session):
    valid_post_payload_without_session["episode_type"] = "embedding"
    response = client.post(
        "/v1/memories/profile", json=valid_post_payload_without_session
    )
    assert response.status_code == 200
    assert response.headers["session-id"] == "default"
    assert response.headers["group-id"] == "default"


def test_post_memories_valid_list_content(valid_post_payload):
    valid_post_payload["episode_content"] = [0.1, 0.2, 0.3]
    valid_post_payload["episode_type"] = "embedding"
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


@pytest.mark.parametrize(
    "missing_field",
    ["session", "producer", "produced_for", "episode_content", "episode_type"],
)
def test_post_memories_missing_required_field(valid_post_payload, missing_field):
    del valid_post_payload[missing_field]
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code == 200, f"should not fail without {missing_field}"


@pytest.mark.parametrize(
    "missing_session_field", ["group_id", "agent_id", "user_id", "session_id"]
)
def test_post_memories_missing_nested_session_field(
    valid_post_payload, missing_session_field
):
    del valid_post_payload["session"][missing_session_field]
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code == 200, (
        f"Should not fail with missing {missing_session_field}"
    )


def test_post_memories_invalid_types(valid_post_payload):
    invalid_payload = {
        "session": "not-a-dict",
        "producer": 123,
        "produced_for": ["not-a-string"],
        "episode_content": {"wrong": "type"},
        "episode_type": False,
        "metadata": "not-a-dict",
    }
    response = client.post("/v1/memories", json=invalid_payload)
    assert response.status_code == 422


def test_post_memories_extra_field(valid_post_payload):
    valid_post_payload["unexpected_field"] = "should-be-accepted"
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


def test_post_memories_boundary_empty_values(valid_post_payload):
    valid_post_payload["session"]["group_id"] = ""
    valid_post_payload["session"]["agent_id"] = []
    valid_post_payload["session"]["user_id"] = []
    valid_post_payload["producer"] = ""
    valid_post_payload["episode_content"] = ""
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


def test_post_memories_null_metadata(valid_post_payload):
    del valid_post_payload["metadata"]
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


# --- Test memory query /v1/memories/search ---
def test_memory_search_valid(valid_query_payload):
    """
    Test query both episodic and profile memories.
    """
    response = client.post("/v1/memories/search", json=valid_query_payload)
    assert response.status_code in (200, 201, 204)
    rsp = response.json()["content"]
    assert len(rsp) == 2
    assert "episodic_memory" in rsp.keys()
    assert "profile_memory" in rsp.keys()


def test_memory_search_with_session_in_header(
    query_payload_without_session, valid_session_headers
):
    response = client.post(
        "/v1/memories/search",
        json=query_payload_without_session,
        headers=valid_session_headers,
    )
    assert response.status_code in (200, 201, 204)
    rsp = response.json()["content"]
    assert len(rsp) == 2
    assert "episodic_memory" in rsp.keys()
    assert "profile_memory" in rsp.keys()


def test_memory_search_without_session(query_payload_without_session):
    response = client.post("/v1/memories/search", json=query_payload_without_session)
    assert response.status_code == 200
    assert response.headers["session-id"] == "default"
    assert response.headers["group-id"] == "default"


# --- Test episodic memory query /v1/memories/episodic/search ---
def test_episodic_memory_search_valid(valid_query_payload):
    """
    Test episodic memory query.
    """
    response = client.post("/v1/memories/episodic/search", json=valid_query_payload)
    assert response.status_code in (200, 201, 204)
    rsp = response.json()["content"]
    assert len(rsp) == 1
    assert "episodic_memory" in rsp.keys()


def test_episodic_memory_search_with_session_in_header(
    query_payload_without_session, valid_session_headers
):
    response = client.post(
        "/v1/memories/episodic/search",
        json=query_payload_without_session,
        headers=valid_session_headers,
    )
    assert response.status_code in (200, 201, 204)
    rsp = response.json()["content"]
    assert len(rsp) == 1
    assert "episodic_memory" in rsp.keys()


def test_episodic_memory_search_without_session(query_payload_without_session):
    response = client.post(
        "/v1/memories/episodic/search", json=query_payload_without_session
    )
    assert response.status_code == 200
    assert response.headers["session-id"] == "default"
    assert response.headers["group-id"] == "default"


# --- Test profile memory query /v1/memories/profile/search ---
def test_profile_memory_search_valid(valid_query_payload):
    """
    Test profile memory query.
    """
    response = client.post("/v1/memories/profile/search", json=valid_query_payload)
    assert response.status_code in (200, 201, 204)
    rsp = response.json()["content"]
    assert len(rsp) == 1
    assert "profile_memory" in rsp.keys()


def test_profile_memory_search_with_session_in_header(
    query_payload_without_session, valid_session_headers
):
    response = client.post(
        "/v1/memories/profile/search",
        json=query_payload_without_session,
        headers=valid_session_headers,
    )
    assert response.status_code in (200, 201, 204)
    rsp = response.json()["content"]
    assert len(rsp) == 1
    assert "profile_memory" in rsp.keys()


def test_profile_memory_search_without_session(query_payload_without_session):
    response = client.post(
        "/v1/memories/profile/search", json=query_payload_without_session
    )
    assert response.status_code == 200
    assert response.headers["session-id"] == "default"
    assert response.headers["group-id"] == "default"


# --- Tests for DELETE /v1/memories ---


def test_delete_memories_valid(valid_delete_payload):
    response = client.request("DELETE", "/v1/memories", json=valid_delete_payload)
    assert response.status_code in (200, 201, 204)


def test_delete_memories_missing_session():
    response = client.request("DELETE", "/v1/memories", json={})
    assert response.status_code == 200
    assert response.headers["session-id"] == "default"
    assert response.headers["group-id"] == "default"


def test_delete_memories_with_session_in_header(valid_session_headers):
    response = client.request(
        "DELETE",
        "/v1/memories",
        json={},
        headers=valid_session_headers,
    )
    assert response.status_code in (200, 201, 204)


@pytest.mark.parametrize(
    "missing_session_field", ["group_id", "agent_id", "user_id", "session_id"]
)
def test_delete_memories_missing_nested_session_field(
    valid_delete_payload, missing_session_field
):
    del valid_delete_payload["session"][missing_session_field]
    response = client.request("DELETE", "/v1/memories", json=valid_delete_payload)
    assert response.status_code == 200, (
        f"Should not fail with missing session field: {missing_session_field}"
    )


def test_delete_memories_invalid_types():
    invalid_payload = {
        "session": {
            "group_id": 123,
            "agent_id": "not-a-list",
            "user_id": False,
            "session_id": None,
        }
    }
    response = client.request("DELETE", "/v1/memories", json=invalid_payload)
    assert response.status_code == 422


def test_delete_memories_extra_field(valid_delete_payload):
    valid_delete_payload["unexpected_field"] = "should-be-accepted"
    response = client.request("DELETE", "/v1/memories", json=valid_delete_payload)
    assert response.status_code in (200, 201, 204)
