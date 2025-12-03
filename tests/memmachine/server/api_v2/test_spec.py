import pytest
from pydantic import ValidationError

from memmachine.main.memmachine import MemoryType
from memmachine.server.api_v2.spec import (
    DEFAULT_ORG_AND_PROJECT_ID,
    AddMemoriesResponse,
    AddMemoriesSpec,
    AddMemoryResult,
    CreateProjectSpec,
    DeleteEpisodicMemorySpec,
    DeleteProjectSpec,
    DeleteSemanticMemorySpec,
    ListMemoriesSpec,
    MemoryMessage,
    ProjectConfig,
    ProjectResponse,
    SearchMemoriesSpec,
    SearchResult,
)


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
    config = ProjectConfig()
    assert config.reranker == ""
    assert config.embedder == ""


def test_create_project_spec_with_defaults():
    with pytest.raises(ValidationError) as exc_info:
        CreateProjectSpec()
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})

    spec = CreateProjectSpec(org_id="org1", project_id="proj1")
    assert spec.org_id == "org1"
    assert spec.project_id == "proj1"
    assert spec.description == ""
    assert spec.config is not None
    assert spec.config.reranker == ""
    assert spec.config.embedder == ""


def test_project_response_with_defaults():
    with pytest.raises(ValidationError) as exc_info:
        ProjectResponse()
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})

    response = ProjectResponse(org_id="org1", project_id="proj1")
    assert response.org_id == "org1"
    assert response.project_id == "proj1"
    assert response.description == ""
    assert response.config is not None
    assert response.config.reranker == ""
    assert response.config.embedder == ""


def test_get_project_spec():
    with pytest.raises(ValidationError) as exc_info:
        CreateProjectSpec()
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})


def test_delete_project_spec():
    with pytest.raises(ValidationError) as exc_info:
        DeleteProjectSpec()
    assert_pydantic_errors(exc_info, {"org_id": "missing", "project_id": "missing"})


def test_memory_message_required_fields():
    with pytest.raises(ValidationError) as exc_info:
        MemoryMessage()
    assert_pydantic_errors(exc_info, {"content": "missing"})

    message = MemoryMessage(content="This is a memory message.")
    assert message.content == "This is a memory message."
    assert message.producer == "user"
    assert message.produced_for == ""
    assert message.timestamp
    assert message.role == ""
    assert message.metadata == {}


def test_add_memory_spec():
    with pytest.raises(ValidationError) as exc_info:
        AddMemoriesSpec()
    assert_pydantic_errors(exc_info, {"messages": "missing"})

    message = MemoryMessage(content="Test content", producer="test_producer")
    spec = AddMemoriesSpec(messages=[message])
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.messages == [message]

    with pytest.raises(ValidationError) as exc_info:
        AddMemoriesSpec(messages=[])
    assert_pydantic_errors(exc_info, {"messages": "too_short"})


def test_add_memory_result():
    with pytest.raises(ValidationError) as exc_info:
        AddMemoryResult()
    assert_pydantic_errors(exc_info, {"uid": "missing"})


def test_add_memory_response():
    with pytest.raises(ValidationError) as exc_info:
        AddMemoriesResponse()
    assert_pydantic_errors(exc_info, {"results": "missing"})

    result = AddMemoryResult(uid="memory-123")
    assert result.uid == "memory-123"
    response = AddMemoriesResponse(results=[result])
    assert response.results == [result]

    response = AddMemoriesResponse(results=[])
    assert response.results == []


def test_search_memories_spec():
    with pytest.raises(ValidationError) as exc_info:
        SearchMemoriesSpec()
    assert_pydantic_errors(exc_info, {"query": "missing"})

    spec = SearchMemoriesSpec(query="Find this")
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.top_k == 10
    assert spec.query == "Find this"
    assert spec.filter == ""
    assert spec.types == []


def test_list_memories_spec():
    spec = ListMemoriesSpec()
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.page_size == 100
    assert spec.page_num == 0
    assert spec.filter == ""
    assert spec.type == MemoryType.Episodic


def test_delete_episodic_memory_spec():
    with pytest.raises(ValidationError) as exc_info:
        DeleteEpisodicMemorySpec()
    assert_pydantic_errors(exc_info, {"episodic_id": "missing"})

    spec = DeleteEpisodicMemorySpec(episodic_id="ep-123")
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.episodic_id == "ep-123"


def test_delete_semantic_memory_spec():
    with pytest.raises(ValidationError) as exc_info:
        DeleteSemanticMemorySpec()
    assert_pydantic_errors(exc_info, {"semantic_id": "missing"})

    spec = DeleteSemanticMemorySpec(semantic_id="sem-123")
    assert spec.org_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.project_id == DEFAULT_ORG_AND_PROJECT_ID
    assert spec.semantic_id == "sem-123"


def test_search_result_model():
    with pytest.raises(ValidationError) as exc_info:
        SearchResult()
    assert_pydantic_errors(exc_info, {"content": "missing"})

    result = SearchResult(status=0, content={"key": "value"})
    assert result.status == 0
    assert result.content == {"key": "value"}
