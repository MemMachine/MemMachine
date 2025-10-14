import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from memmachine.server.app import app

"""
================================================================================
MemMachine REST API /v1/memories Endpoint Tests

This test suite provides comprehensive coverage for the /v1/memories endpoint,
including POST, DELETE, and query operations. It validates various scenarios:

- Valid payloads (string, list content for embeddings)
- Missing required fields (top-level and nested session fields)
- Invalid data types for fields
- Handling of extra, unexpected fields
- Boundary conditions for empty values and null metadata
- Rejection of null values for required session fields
- Robustness against malicious/junk input (path traversal, null byte injection, SQL injection)
- Numerical boundary conditions for 'episode_content' (embeddings), covering
  negative, zero, large, fractional numbers, and rejection of special floats (inf, nan).
- Rejection of unsupported HTTP methods (e.g., GET).

The tests utilize mocking of AsyncEpisodicMemory to isolate endpoint logic.
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


# --- Tests for POST /v1/memories ---
# (No changes needed to the test functions themselves)


def test_post_memories_valid_list_content(valid_post_payload):
    """Tests that POST requests with list content (embeddings) are rejected."""
    valid_post_payload["episode_content"] = [0.1, 0.2, 0.3]
    valid_post_payload["episode_type"] = "embedding"
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing_field",
    ["session", "producer", "produced_for", "episode_content", "episode_type"],
)
def test_post_memories_missing_required_field(valid_post_payload, missing_field):
    """Tests that POST requests fail when a top-level required field is missing."""
    del valid_post_payload[missing_field]
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code == 422, (
        f"Should fail with missing field: {missing_field}"
    )


@pytest.mark.parametrize(
    "missing_optional_session_field", ["group_id", "agent_id", "user_id"]
)
def test_post_memories_missing_optional_nested_session_field(
    valid_post_payload, missing_optional_session_field
):
    """Tests that POST requests succeed when an optional nested session field is missing."""
    payload = valid_post_payload.copy()
    del payload["session"][missing_optional_session_field]
    response = client.post("/v1/memories", json=payload)
    assert response.status_code == 200, (
        f"Should succeed with missing optional session field: {missing_optional_session_field}"
    )


def test_post_memories_missing_required_session_id(valid_post_payload):
    """Tests that POST requests fail when the required session_id is missing."""
    payload = valid_post_payload.copy()
    del payload["session"]["session_id"]
    response = client.post("/v1/memories", json=payload)
    assert response.status_code == 422


def test_post_memories_invalid_types(valid_post_payload):
    """Tests that POST requests fail when fields have invalid data types."""
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
    """Tests that POST requests succeed even with unexpected extra fields."""
    valid_post_payload["unexpected_field"] = "should-be-accepted"
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)

    @pytest.mark.parametrize(
        "field, value, expected_status",
        [
            ("producer", "", 422),
            ("producer", None, 422),
            ("producer", "invalid char!", 422),
            ("produced_for", "", 422),
            ("produced_for", None, 422),
            ("produced_for", "invalid char!", 422),
            ("episode_content", "", 422),
            ("episode_content", None, 422),
            ("episode_type", "", 422),
            ("episode_type", None, 422),
            ("episode_type", "some_unknown_type", 200),
        ],
    )
    def test_post_memories_invalid_fields(
        valid_post_payload, field, value, expected_status
    ):
        """Tests that POST requests fail when required fields have invalid values."""
        payload = valid_post_payload.copy()
        payload[field] = value
        response = client.post("/v1/memories", json=payload)
        assert response.status_code == expected_status, (
            f"Should fail with invalid value '{value}' for field '{field}'"
        )


def test_post_memories_null_metadata(valid_post_payload):
    """Tests that POST requests succeed when optional metadata is null."""
    valid_post_payload["metadata"] = None
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code in (200, 201, 204)


@pytest.mark.parametrize("optional_field", ["group_id", "agent_id", "user_id"])
def test_post_memories_none_optional_session_field(valid_post_payload, optional_field):
    """Tests that POST requests succeed when an optional nested session field is None."""
    payload = valid_post_payload.copy()
    payload["session"][optional_field] = None
    response = client.post("/v1/memories", json=payload)
    assert response.status_code == 200, (
        f"Should succeed with None optional session field: {optional_field}"
    )


def test_post_memories_none_required_session_id(valid_post_payload):
    """Tests that POST requests fail when the required session_id is None."""
    payload = valid_post_payload.copy()
    payload["session"]["session_id"] = None
    response = client.post("/v1/memories", json=payload)
    assert response.status_code == 422


def test_post_memories_all_none_session_fields(valid_post_payload):
    """Tests that setting all session fields to None fails validation."""
    valid_post_payload["session"] = {
        "group_id": None,
        "agent_id": None,
        "user_id": None,
        "session_id": None,
    }
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, malicious_value",
    [
        ("group_id", "../../../../../../../../../../../../../../../../../etc/passwd"),
        ("group_id", "valid_id\\u0000malicious_suffix"),
        ("group_id", "' OR '1'='1 --"),
        ("session_id", "../../../../../../../../../../../../../../../../../etc/passwd"),
        ("session_id", "valid_session\\u0000malicious_suffix"),
        ("session_id", "' OR '1'='1 --"),
        ("agent_id", ["agent1", "../../../etc/shadow"]),
        ("user_id", ["user1", "user2\\u0000evil"]),
    ],
)
def test_post_memories_malicious_input(valid_post_payload, field, malicious_value):
    """
    Tests that POST requests fail or handle safely when session fields contain malicious or junk input.
    Expected to fail with 422 if input validation is strict.
    """
    if field in ["agent_id", "user_id"]:
        valid_post_payload["session"][field] = malicious_value
    else:
        valid_post_payload["session"][field] = malicious_value
    response = client.post("/v1/memories", json=valid_post_payload)
    assert response.status_code == 422, (
        f"Should fail with malicious input in {field}: {malicious_value}"
    )


@pytest.mark.parametrize(
    "numerical_list, expected_status",
    [
        # All numerical lists should now be rejected (422) as episode_content is strictly 'str'
        ([-1.0, -0.5, -100.0], 422),  # Negative numbers
        ([0.0, 0.0], 422),  # Zero
        ([0.0000001, -0.0000001], 422),  # Small fractional numbers
        (
            [1.7976931348623157e308, 1.0e100],
            422,
        ),  # Large positive numbers (Python float max)
        ([-1.7976931348623157e308, -1.0e100], 422),  # Large negative numbers
        ([-1.0, 0.0, 1.0, 0.5, -0.5], 422),  # Mixed numbers
        ([1.0, 2.0, 3.0], 422),  # Integers as floats
        # Invalid numerical types/edge cases (expect 422 from Pydantic)
        ([float("inf")], 422),
        ([float("-inf")], 422),
        ([float("nan")], 422),
        (["not_a_number"], 422),  # Non-numerical item in list
        ([1, "invalid", 3], 422),  # Mixed valid and invalid
    ],
)
def test_post_memories_numerical_boundaries_episode_content(
    valid_post_payload, numerical_list, expected_status
):
    """
    Tests POST requests with various numerical boundary conditions for episode_content (embeddings).
    Checks for valid float lists and rejection of invalid float values like inf, -inf, nan.
    """
    valid_post_payload["episode_content"] = numerical_list
    valid_post_payload["episode_type"] = "embedding"
    response = client.post(
        "/v1/memories",
        content=json.dumps(valid_post_payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == expected_status, (
        f"Expected status {expected_status} for numerical list {numerical_list}, got {response.status_code}"
    )


@pytest.mark.parametrize(
    "endpoint, expected_keys",
    [
        ("/v1/memories/search", ["episodic_memory", "profile_memory"]),
        ("/v1/memories/episodic/search", ["episodic_memory"]),
        ("/v1/memories/profile/search", ["profile_memory"]),
    ],
)
def test_search_valid_content_to_endpoints(
    valid_query_payload, endpoint, expected_keys
):
    """Tests successful memory search across all relevant search endpoints."""
    response = client.post(endpoint, json=valid_query_payload)
    assert response.status_code in (200, 201, 204)
    rsp = response.json()["content"]
    assert len(rsp) == len(expected_keys)
    for key in expected_keys:
        assert key in rsp.keys()


@pytest.mark.parametrize("field_to_remove", ["session", "query"])
def test_search_memories_missing_required_field(valid_query_payload, field_to_remove):
    """Tests that search requests fail when a required field is missing."""
    del valid_query_payload[field_to_remove]
    response = client.post("/v1/memories/search", json=valid_query_payload)
    assert response.status_code == 422, (
        f"Should fail with missing field: {field_to_remove}"
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("query", ""),
        ("query", None),
        ("limit", -1),
        ("limit", "abc"),
        ("filter", "not-a-dict"),
    ],
)
def test_search_memories_invalid_field_values(valid_query_payload, field, value):
    """Tests that search requests fail when fields have invalid values."""
    valid_query_payload[field] = value
    response = client.post("/v1/memories/search", json=valid_query_payload)
    assert response.status_code == 422, (
        f"Should fail with invalid value '{value}' for field '{field}'"
    )


# --- Tests for DELETE /v1/memories ---


def test_delete_memories_valid(valid_delete_payload):
    """Tests successful memory deletion with a valid payload."""
    response = client.request("DELETE", "/v1/memories", json=valid_delete_payload)
    assert response.status_code in (200, 201, 204)


def test_delete_memories_missing_session():
    """Tests that DELETE requests fail when the session object is missing."""
    response = client.request("DELETE", "/v1/memories", json={})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing_optional_session_field", ["group_id", "agent_id", "user_id"]
)
def test_delete_memories_missing_optional_nested_session_field(
    valid_delete_payload, missing_optional_session_field
):
    """Tests that DELETE requests succeed when an optional nested session field is missing."""
    payload = valid_delete_payload.copy()
    del payload["session"][missing_optional_session_field]
    response = client.request("DELETE", "/v1/memories", json=payload)
    assert response.status_code == 200, (
        f"Should succeed with missing optional session field: {missing_optional_session_field}"
    )


def test_delete_memories_missing_required_session_id(valid_delete_payload):
    """Tests that DELETE requests fail when the required session_id is missing."""
    payload = valid_delete_payload.copy()
    del payload["session"]["session_id"]
    response = client.request("DELETE", "/v1/memories", json=payload)
    assert response.status_code == 422


def test_delete_memories_invalid_types():
    """Tests that DELETE requests fail when session fields have invalid types."""
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
    """Tests that DELETE requests succeed even with unexpected extra fields."""
    valid_delete_payload["unexpected_field"] = "should-be-accepted"
    response = client.request("DELETE", "/v1/memories", json=valid_delete_payload)
    assert response.status_code in (200, 201, 204)


@pytest.mark.parametrize("optional_field", ["group_id", "agent_id", "user_id"])
def test_delete_memories_none_optional_session_field(
    valid_delete_payload, optional_field
):
    """Tests that DELETE requests succeed when an optional nested session field is None."""
    payload = valid_delete_payload.copy()
    payload["session"][optional_field] = None
    response = client.request("DELETE", "/v1/memories", json=payload)
    assert response.status_code == 200, (
        f"Should succeed with None optional session field: {optional_field}"
    )


def test_delete_memories_none_required_session_id(valid_delete_payload):
    """Tests that DELETE requests fail when the required session_id is None."""
    payload = valid_delete_payload.copy()
    payload["session"]["session_id"] = None
    response = client.request("DELETE", "/v1/memories", json=payload)
    assert response.status_code == 422


def test_delete_memories_all_none_session_fields(valid_delete_payload):
    """Tests that setting all session fields to None fails validation."""
    valid_delete_payload["session"] = {
        "group_id": None,
        "agent_id": None,
        "user_id": None,
        "session_id": None,
    }
    response = client.request("DELETE", "/v1/memories", json=valid_delete_payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, malicious_value",
    [
        ("group_id", "../../../../../../../../../../../../../../../../../etc/passwd"),
        ("group_id", "valid_id\\u0000malicious_suffix"),
        ("group_id", "' OR '1'='1 --"),
        ("session_id", "../../../../../../../../../../../../../../../../../etc/passwd"),
        ("session_id", "valid_session\\u0000malicious_suffix"),
        ("session_id", "' OR '1'='1 --"),
        ("agent_id", ["agent1", "../../../etc/shadow"]),
        ("user_id", ["user1", "user2\\u0000evil"]),
    ],
)
def test_delete_memories_malicious_input(valid_delete_payload, field, malicious_value):
    """
    Tests that DELETE requests fail or handle safely when session fields contain malicious or junk input.
    Expected to fail with 422 if input validation is strict.
    """
    if field in ["agent_id", "user_id"]:
        valid_delete_payload["session"][field] = malicious_value
    else:
        valid_delete_payload["session"][field] = malicious_value
    response = client.request("DELETE", "/v1/memories", json=valid_delete_payload)
    assert response.status_code == 422, (
        f"Should fail with malicious input in {field}: {malicious_value}"
    )


@pytest.mark.parametrize(
    "field, value, expected_status",
    [
        ("session_id", "", 422),  # Should fail on empty string
        ("session_id", None, 422),  # Should fail on None
        ("group_id", "", 422),  # Should succeed on empty string
        ("group_id", None, 200),  # Should succeed on None
        ("agent_id", [], 422),  # Should fail on empty list
        ("agent_id", None, 200),  # Should succeed on None
        ("user_id", [], 422),  # Should fail on empty list
        ("user_id", None, 200),  # Should succeed on None
    ],
)
def test_delete_memories_boundary_values(
    valid_delete_payload, field, value, expected_status
):
    """Tests DELETE requests with various boundary conditions for session fields."""
    payload = valid_delete_payload.copy()
    payload["session"][field] = value
    response = client.request("DELETE", "/v1/memories", json=payload)
    assert response.status_code == expected_status, (
        f"Expected status {expected_status} for field '{field}' with value '{value}', got {response.status_code}"
    )


# --- Tests for Unsupported Methods ---


def test_get_memories_not_allowed():
    """
    Tests that GET requests to /v1/memories are rejected with 405 Method Not Allowed.
    """
    response = client.get("/v1/memories")
    assert response.status_code == 405
