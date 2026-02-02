from datetime import UTC, datetime

from typing import cast

import pytest
from dateutil.tz import tzoffset
from pydantic import ValidationError

from memmachine.common.api.spec import (
    DEFAULT_ORG_AND_PROJECT_ID,
    AddMemoriesResponse,
    AddMemoriesSpec,
    AddMemoryResult,
    CreateProjectSpec,
    DeleteEpisodicMemorySpec,
    DeleteProjectSpec,
    DeleteSemanticMemorySpec,
    InvalidNameError,
    ListMemoriesSpec,
    MemoryMessage,
    ProjectConfig,
    ProjectResponse,
    SearchMemoriesSpec,
    SearchResult,
    _is_valid_name,
)
from memmachine.common.episode_store.episode_model import EpisodeType
from memmachine.server.api_v2.router import RestError


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "abc123",
        "ABC:xyz-123",
        "你好世界",
        "テスト123",
        "안녕-세상",
        "名字:123",
        "中文-english_混合",
    ],
)
def test_validate_no_slash_valid(value):
    assert _is_valid_name(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "a/b",  # slash
        "/abc",  # slash
        "abc/",  # slash
        "hello world",  # space
        "name@123",  # symbol
        "value!",  # symbol
        "中文 test",  # space
        "",  # empty
    ],
)
def test_validate_no_slash_invalid(value):
    with pytest.raises(InvalidNameError) as exe_info:
        _is_valid_name(value)
    message = f"found: '{value}'"
    assert message in str(exe_info.value)


def assert_pydantic_errors(exc_info, expected_checks: dict[str, str]):
    errors = exc_info.value.errors()

    error_map = {e["loc"][-1]: e["type"] for e in errors}

    for field, expected_type in expected_checks.items():
        assert field in error_map, f"Expected error for field '{field}' not found."
        assert error_map[field] == expected_type, (
            f"Field '{field}': expected error type '{expected_type}', "
            f"but got '{error_map[field]}'"
        )
    assert len(error_map) == len(expected_checks), (
        f"Expected {len(expected_checks)} errors, but got {len(error_map)}."
    )


def test_project_config_defaults():
    config = ProjectConfig.model_validate({})
    assert config.reranker == ""
    assert config.embedder == ""


def test_create_project_spec_with_defaults():
    with pytest.raises(ValidationError) as exc_info:
        CreateProjectSpec.model_validate({})
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})

    spec = CreateProjectSpec.model_validate({"org_id": "org1", "project_id": "proj1"})
    assert spec.org_id == "org1"
    assert spec.project_id == "proj1"
    assert spec.description == ""
    assert spec.config is not None
    assert spec.config.reranker == ""
    assert spec.config.embedder == ""


def test_project_response_with_defaults():
    with pytest.raises(ValidationError) as exc_info:
        ProjectResponse.model_validate({})
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})

    response = ProjectResponse.model_validate({"org_id": "org1", "project_id": "proj1"})
    assert response.org_id == "org1"
    assert response.project_id == "proj1"
    assert response.description == ""
    assert response.config is not None
    assert response.config.reranker == ""
    assert response.config.embedder == ""


def test_get_project_spec():
    with pytest.raises(ValidationError) as exc_info:
        CreateProjectSpec.model_validate({})
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})


def test_delete_project_spec():
    with pytest.raises(ValidationError) as exc_info:
        DeleteProjectSpec.model_validate({})
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})


def test_memory_message_required_fields():
    with pytest.raises(ValidationError) as exc_info:
        MemoryMessage.model_validate({})
    assert_pydantic_errors(exc_info, {"content": "missing"})

    message = MemoryMessage.model_validate({"content": "This is a memory message."})
    assert message.content == "This is a memory message."
    assert message.producer == "user"
    assert message.produced_for == ""
    assert message.timestamp
    assert message.role == ""
    assert message.metadata == {}
    assert message.episode_type is None


def test_memory_message_episode_type_validation():
    message = MemoryMessage.model_validate(
        {"content": "hello", "episode_type": "message"}
    )
    assert message.episode_type == EpisodeType.MESSAGE

    with pytest.raises(ValidationError) as exc_info:
        MemoryMessage.model_validate(
            {"content": "hello", "episode_type": "not-a-real-type"}
        )
    assert any(e["loc"][-1] == "episode_type" for e in exc_info.value.errors())


def test_add_memory_spec():
    with pytest.raises(ValidationError) as exc_info:
        AddMemoriesSpec.model_validate({})
    assert_pydantic_errors(exc_info, {"messages": "missing"})

    message = MemoryMessage.model_validate(
        {"content": "Test content", "producer": "test_producer"}
    )
    spec = AddMemoriesSpec.model_validate({"messages": [message]})
    assert spec.types == []
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.messages == [message]

    with pytest.raises(ValidationError) as exc_info:
        AddMemoriesSpec.model_validate({"messages": []})
    assert_pydantic_errors(exc_info, {"messages": "too_short"})


def test_add_memory_result():
    with pytest.raises(ValidationError) as exc_info:
        AddMemoryResult.model_validate({})
    assert_pydantic_errors(exc_info, {"uid": "missing"})


def test_add_memory_response():
    with pytest.raises(ValidationError) as exc_info:
        AddMemoriesResponse.model_validate({})
    assert_pydantic_errors(exc_info, {"results": "missing"})

    result = AddMemoryResult.model_validate({"uid": "memory-123"})
    assert result.uid == "memory-123"
    response = AddMemoriesResponse.model_validate({"results": [result]})
    assert response.results == [result]

    response = AddMemoriesResponse.model_validate({"results": []})
    assert response.results == []


def test_search_memories_spec():
    with pytest.raises(ValidationError) as exc_info:
        SearchMemoriesSpec.model_validate({})
    assert_pydantic_errors(exc_info, {"query": "missing"})

    spec = SearchMemoriesSpec.model_validate({"query": "Find this"})
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.top_k == 10
    assert spec.query == "Find this"
    assert spec.filter == ""
    assert spec.types == []


def test_list_memories_spec():
    spec = ListMemoriesSpec.model_validate({})
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.page_size == 100
    assert spec.page_num == 0
    assert spec.filter == ""
    assert spec.type is None


def test_delete_episodic_memory_spec():
    with pytest.raises(ValidationError):
        DeleteEpisodicMemorySpec.model_validate({})

    spec = DeleteEpisodicMemorySpec.model_validate({"episodic_id": "ep-123"})
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.episodic_id == "ep-123"


def test_delete_semantic_memory_spec():
    with pytest.raises(ValidationError):
        DeleteSemanticMemorySpec.model_validate({})

    spec = DeleteSemanticMemorySpec.model_validate({"semantic_id": "sem-123"})
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.semantic_id == "sem-123"


def test_get_semantic_ids():
    spec = DeleteSemanticMemorySpec.model_validate({"semantic_ids": ["2", "1"]})
    assert spec.get_ids() == ["1", "2"]

    spec = DeleteSemanticMemorySpec.model_validate(
        {"semantic_id": "1", "semantic_ids": ["2", "1"]}
    )
    assert spec.get_ids() == ["1", "2"]


def test_get_episodic_ids():
    spec = DeleteEpisodicMemorySpec.model_validate({"episodic_ids": ["2", "1"]})
    assert spec.get_ids() == ["1", "2"]

    spec = DeleteEpisodicMemorySpec.model_validate(
        {"episodic_id": "1", "episodic_ids": ["2", "1"]}
    )
    assert spec.get_ids() == ["1", "2"]


def test_search_result_model():
    with pytest.raises(ValidationError) as exc_info:
        SearchResult.model_validate({})
    assert_pydantic_errors(exc_info, {"content": "missing"})

    # content is a typed model now; empty payload is valid
    result = SearchResult.model_validate({"status": 0, "content": {}})
    assert result.status == 0
    assert result.content.model_dump(mode="json", exclude_none=True) == {}


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


def test_timestamp_default_now():
    before = datetime.now(UTC)
    msg = MemoryMessage.model_validate({"content": "hello"})
    after = datetime.now(UTC)

    assert before <= msg.timestamp <= after
    assert msg.timestamp.tzinfo == UTC


def test_timestamp_datetime_with_tz():
    dt = datetime(2025, 5, 23, 10, 30, tzinfo=UTC)
    msg = MemoryMessage.model_validate({"content": "hello", "timestamp": dt})

    assert msg.timestamp == dt


def test_timestamp_datetime_without_tz():
    aware = datetime(2025, 5, 23, 10, 30, tzinfo=UTC)
    dt = aware.replace(tzinfo=None)
    msg = MemoryMessage.model_validate({"content": "hello", "timestamp": dt})

    assert msg.timestamp.tzinfo == UTC
    assert msg.timestamp.replace(tzinfo=None) == dt


def test_timestamp_unix_seconds():
    ts = 1_716_480_000  # unix seconds
    msg = MemoryMessage.model_validate({"content": "hello", "timestamp": ts})

    assert msg.timestamp == datetime.fromtimestamp(ts, tz=UTC)


def test_timestamp_unix_milliseconds():
    ts_ms = 1_716_480_000_000  # unix ms
    msg = MemoryMessage.model_validate({"content": "hello", "timestamp": ts_ms})

    assert msg.timestamp == datetime.fromtimestamp(ts_ms / 1000, tz=UTC)


def test_timestamp_iso_string():
    ts = "2025-05-23T10:30:00Z"
    msg = MemoryMessage.model_validate({"content": "hello", "timestamp": ts})

    assert msg.timestamp == datetime(2025, 5, 23, 10, 30, tzinfo=UTC)


def test_timestamp_common_datetime_string():
    ts = "2023-10-01T08:20:00-04:00"
    msg = MemoryMessage.model_validate({"content": "hello", "timestamp": ts})

    tz = tzoffset(None, -14400)
    assert msg.timestamp == datetime(2023, 10, 1, 8, 20, tzinfo=tz)


def test_timestamp_none_uses_now():
    before = datetime.now(UTC)
    msg = MemoryMessage.model_validate({"content": "hello", "timestamp": None})
    after = datetime.now(UTC)

    assert before <= msg.timestamp <= after


@pytest.mark.parametrize(
    "timestamp",
    [
        "05/23/2025T10:30:00Z",
        "10:30 2025-05-23",
        "now",
    ],
)
def test_timestamp_invalid_string(timestamp):
    with pytest.raises(
        ValidationError,
        match=r"Unsupported timestamp: " + timestamp,
    ):
        MemoryMessage.model_validate({"content": "hello", "timestamp": timestamp})


def test_timestamp_invalid_type():
    error_msg = "Unsupported timestamp: {'bad': 'value'}"
    with pytest.raises(ValidationError, match=error_msg):
        MemoryMessage.model_validate(
            {"content": "hello", "timestamp": {"bad": "value"}}
        )
