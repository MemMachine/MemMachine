"""Additional tests for session id handling and resource retrieval."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    Resources,
    SemanticCategory,
    SemanticPrompt,
)
from memmachine.semantic_memory.semantic_session_manager import (
    IsolationType,
    SemanticSessionManager,
)
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    MockEmbedder,
    SimpleSessionResourceRetriever,
)


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


@pytest.fixture
def session_manager() -> SemanticSessionManager:
    return SemanticSessionManager(MagicMock())


@pytest.fixture
def raw_session_data():
    @dataclass
    class _SessionData:
        user_profile_id: str | None
        session_id: str | None
        role_profile_id: str | None

    return _SessionData(
        user_profile_id="user123",
        session_id="session456",
        role_profile_id="role789",
    )


class TestSessionIdGeneration:
    """Tests for generating and classifying semantic session ids."""

    def test_generate_session_data_creates_valid_session(
        self, session_manager: SemanticSessionManager, raw_session_data
    ):
        session_data = session_manager._generate_session_data(
            session_data=raw_session_data,
        )

        assert session_data.user_profile_id == "mem_user_user123"
        assert session_data.session_id == "mem_session_session456"
        assert session_data.role_profile_id == "mem_role_role789"

    def test_generate_session_data_with_empty_strings(
        self, session_manager: SemanticSessionManager
    ):
        @dataclass
        class _SessionData:
            user_profile_id: str | None
            session_id: str | None
            role_profile_id: str | None

        session_data = session_manager._generate_session_data(
            session_data=_SessionData(
                user_profile_id="",
                session_id="",
                role_profile_id=None,
            ),
        )

        # Should return None
        assert session_data.user_profile_id is None
        assert session_data.session_id is None

    def test_is_session_id_recognizes_session_prefix(self):
        assert SemanticSessionManager.is_session_id("mem_session_abc123")
        assert SemanticSessionManager.is_session_id("mem_session_")
        assert not SemanticSessionManager.is_session_id("mem_user_abc123")
        assert not SemanticSessionManager.is_session_id("random_id")
        assert not SemanticSessionManager.is_session_id("")

    def test_is_producer_id_recognizes_user_prefix(self):
        assert SemanticSessionManager.is_producer_id("mem_user_user456")
        assert SemanticSessionManager.is_producer_id("mem_user_")
        assert not SemanticSessionManager.is_producer_id("mem_session_session789")
        assert not SemanticSessionManager.is_producer_id("random_id")
        assert not SemanticSessionManager.is_producer_id("")

    def test_set_id_isolation_type_returns_session(self):
        isolation_type = SemanticSessionManager.set_id_isolation_type("mem_session_xyz")

        assert isolation_type == IsolationType.SESSION

    def test_set_id_isolation_type_returns_user(self):
        isolation_type = SemanticSessionManager.set_id_isolation_type("mem_user_xyz")

        assert isolation_type == IsolationType.USER

    def test_set_id_isolation_type_returns_role(self):
        isolation_type = SemanticSessionManager.set_id_isolation_type("mem_role_xyz")

        assert isolation_type == IsolationType.ROLE

    def test_set_id_isolation_type_raises_on_invalid_id(self):
        with pytest.raises(ValueError, match="Invalid id"):
            SemanticSessionManager.set_id_isolation_type("invalid_id")


class TestSessionResourceRetriever:
    """Tests for resource retrieval keyed by session set ids."""

    def test_get_resources_for_session_id(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        retriever = SimpleSessionResourceRetriever(
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
                IsolationType.ROLE: session_resources,
            }
        )

        resources = retriever.get_resources("mem_session_test123")

        assert resources == session_resources
        assert resources.semantic_categories[0].name == "Session"

    def test_get_resources_for_profile_id(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        retriever = SimpleSessionResourceRetriever(
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
                IsolationType.ROLE: session_resources,
            }
        )

        resources = retriever.get_resources("mem_user_test456")

        assert resources == profile_resources
        assert resources.semantic_categories[0].name == "Profile"

    def test_get_resources_with_invalid_id_raises_error(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        retriever = SimpleSessionResourceRetriever(
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
                IsolationType.ROLE: session_resources,
            }
        )

        with pytest.raises(ValueError, match="Invalid id"):
            retriever.get_resources("invalid_set_id")

    def test_get_resources_returns_different_resources_for_different_types(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        retriever = SimpleSessionResourceRetriever(
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
                IsolationType.ROLE: session_resources,
            }
        )

        profile_res = retriever.get_resources("mem_user_user1")
        session_res = retriever.get_resources("mem_session_session1")

        assert profile_res != session_res
        assert profile_res.semantic_categories[0].name == "Profile"
        assert session_res.semantic_categories[0].name == "Session"

    def test_unknown_type_defaults_raise(
        self,
        profile_resources: Resources,
        session_resources: Resources,
    ):
        retriever = SimpleSessionResourceRetriever(
            default_resources={
                IsolationType.USER: profile_resources,
                IsolationType.SESSION: session_resources,
                IsolationType.ROLE: session_resources,
            }
        )

        with pytest.raises(ValueError, match="Invalid id"):
            retriever.get_resources("any_set_id")


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
