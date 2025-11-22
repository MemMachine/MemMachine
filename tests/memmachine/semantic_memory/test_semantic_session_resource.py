"""Additional tests for SessionResourceRetriever and SessionIdManager."""

import pytest

from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    Resources,
    SemanticCategory,
    SemanticPrompt,
)
from memmachine.semantic_memory.semantic_session_resource import (
    IsolationType,
    SessionIdIsolationTypeChecker,
    SessionIdManager,
    SessionResourceRetriever,
)
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    MockEmbedder,
)


def test_semantic_session_id_manager_is_instance():
    assert isinstance(SessionIdManager(), SessionIdIsolationTypeChecker)


@pytest.fixture
def semantic_prompt() -> SemanticPrompt:
    return RawSemanticPrompt(
        update_prompt="update-prompt",
        consolidation_prompt="consolidation-prompt",
    )


@pytest.fixture
def profile_semantic_type(semantic_prompt: SemanticPrompt) -> SemanticCategory:
    return SemanticCategory(
        name="Profile",
        tags={"profile_tag"},
        prompt=semantic_prompt,
    )


@pytest.fixture
def session_semantic_type(semantic_prompt: SemanticPrompt) -> SemanticCategory:
    return SemanticCategory(
        name="Session",
        tags={"session_tag"},
        prompt=semantic_prompt,
    )


@pytest.fixture
def profile_resources(
    profile_semantic_type: SemanticCategory,
    mock_llm_model,
) -> Resources:
    return Resources(
        embedder=MockEmbedder(),
        language_model=mock_llm_model,
        semantic_categories=[profile_semantic_type],
    )


@pytest.fixture
def session_resources(
    session_semantic_type: SemanticCategory,
    mock_llm_model,
) -> Resources:
    return Resources(
        embedder=MockEmbedder(),
        language_model=mock_llm_model,
        semantic_categories=[session_semantic_type],
    )


class TestSessionIdManager:
    """Tests for SessionIdManager."""

    def test_generate_session_data_creates_valid_session(self):
        manager = SessionIdManager()

        session_data = manager.generate_session_data(
            user_id="user123",
            session_id="session456",
            role_id="role789",
        )

        assert session_data.user_profile_id == "mem_user_user123"
        assert session_data.session_id == "mem_session_session456"
        assert session_data.role_profile_id == "mem_role_role789"

    def test_generate_session_data_with_empty_strings(self):
        manager = SessionIdManager()

        session_data = manager.generate_session_data(
            user_id="",
            session_id="",
        )

        # Should return None
        assert session_data.user_profile_id is None
        assert session_data.session_id is None

    def test_is_session_id_recognizes_session_prefix(self):
        manager = SessionIdManager()

        assert manager.is_session_id("mem_session_abc123")
        assert manager.is_session_id("mem_session_")
        assert not manager.is_session_id("mem_user_abc123")
        assert not manager.is_session_id("random_id")
        assert not manager.is_session_id("")

    def test_is_producer_id_recognizes_user_prefix(self):
        manager = SessionIdManager()

        assert manager.is_producer_id("mem_user_user456")
        assert manager.is_producer_id("mem_user_")
        assert not manager.is_producer_id("mem_session_session789")
        assert not manager.is_producer_id("random_id")
        assert not manager.is_producer_id("")

    def test_set_id_isolation_type_returns_session(self):
        manager = SessionIdManager()

        isolation_type = manager.set_id_isolation_type("mem_session_xyz")

        assert isolation_type == IsolationType.SESSION

    def test_set_id_isolation_type_returns_user(self):
        manager = SessionIdManager()

        isolation_type = manager.set_id_isolation_type("mem_user_xyz")

        assert isolation_type == IsolationType.USER

    def test_set_id_isolation_type_returns_role(self):
        manager = SessionIdManager()

        isolation_type = manager.set_id_isolation_type("mem_role_xyz")

        assert isolation_type == IsolationType.ROLE

    def test_set_id_isolation_type_raises_on_invalid_id(self):
        manager = SessionIdManager()

        with pytest.raises(ValueError, match="Invalid id"):
            manager.set_id_isolation_type("invalid_id")


class TestSessionResourceRetriever:
    """Tests for SessionResourceRetriever."""

    def test_get_resources_for_session_id(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        manager = SessionIdManager()
        retriever = SessionResourceRetriever(
            session_id_manager=manager,
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
            },
        )

        resources = retriever.get_resources("mem_session_test123")

        assert resources == session_resources
        assert resources.semantic_categories[0].name == "Session"

    def test_get_resources_for_profile_id(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        manager = SessionIdManager()
        retriever = SessionResourceRetriever(
            session_id_manager=manager,
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
            },
        )

        resources = retriever.get_resources("mem_user_test456")

        assert resources == profile_resources
        assert resources.semantic_categories[0].name == "Profile"

    def test_get_resources_with_invalid_id_raises_error(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        manager = SessionIdManager()
        retriever = SessionResourceRetriever(
            session_id_manager=manager,
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
            },
        )

        with pytest.raises(ValueError, match="Invalid id"):
            retriever.get_resources("invalid_set_id")

    def test_get_resources_returns_different_resources_for_different_types(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        manager = SessionIdManager()
        retriever = SessionResourceRetriever(
            session_id_manager=manager,
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
            },
        )

        profile_res = retriever.get_resources("mem_user_user1")
        session_res = retriever.get_resources("mem_session_session1")

        assert profile_res != session_res
        assert profile_res.semantic_categories[0].name == "Profile"
        assert session_res.semantic_categories[0].name == "Session"

    def test_custom_resources_returns_none(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        # This tests the current TODO implementation
        manager = SessionIdManager()
        retriever = SessionResourceRetriever(
            session_id_manager=manager,
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
            },
        )

        # The _get_set_id_resources static method should return None
        custom_res = retriever._get_set_id_resources("any_set_id")
        assert custom_res is None


class TestIsolationType:
    """Tests for IsolationType enum."""

    def test_isolation_type_values(self):
        assert IsolationType.USER.value == "user_profile"
        assert IsolationType.ROLE.value == "role_profile"
        assert IsolationType.SESSION.value == "session"

    def test_isolation_type_enum_members(self):
        isolation_types = list(IsolationType)
        assert len(isolation_types) == 3
        assert IsolationType.USER in isolation_types
        assert IsolationType.ROLE in isolation_types
        assert IsolationType.SESSION in isolation_types
