"""Unit tests for the SemanticSessionManager using in-memory storage."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder import Embedder
from memmachine.episode_store.episode_storage import EpisodeStorage
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    Resources,
    SemanticCategory,
    SemanticPrompt,
)
from memmachine.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)
from memmachine.semantic_memory.semantic_session_resource import (
    IsolationType,
    SessionIdManager,
)
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    MockResourceRetriever,
)
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    SemanticStorage,
)

pytestmark = pytest.mark.asyncio


class SpyEmbedder(Embedder):
    """Deterministic embedder that tracks calls for assertions."""

    def __init__(self) -> None:
        self.ingest_calls: list[list[str]] = []
        self.search_calls: list[list[str]] = []

    async def ingest_embed(
        self,
        inputs: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        self.ingest_calls.append(list(inputs))
        return [self._vector(text) for text in inputs]

    async def search_embed(
        self,
        queries: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        self.search_calls.append(list(queries))
        return [self._vector(text) for text in queries]

    @property
    def model_id(self) -> str:
        return "spy-embedder"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        score_alpha = 1.0 if "alpha" in lowered else -1.0
        score_beta = 1.0 if "beta" in lowered else -1.0
        return [score_alpha, score_beta]


@pytest.fixture
def spy_embedder() -> SpyEmbedder:
    return SpyEmbedder()


@pytest.fixture
def semantic_prompt() -> SemanticPrompt:
    return RawSemanticPrompt(
        update_prompt="update-session-memory",
        consolidation_prompt="consolidate-session-memory",
    )


@pytest.fixture
def semantic_type(semantic_prompt: SemanticPrompt) -> SemanticCategory:
    return SemanticCategory(
        name="Profile",
        tags={"general"},
        prompt=semantic_prompt,
    )


@pytest.fixture
def resources(
    spy_embedder: SpyEmbedder,
    mock_llm_model,
    semantic_type: SemanticCategory,
) -> Resources:
    return Resources(
        embedder=spy_embedder,
        language_model=mock_llm_model,
        semantic_categories=[semantic_type],
    )


@pytest.fixture
def resource_retriever(resources: Resources) -> MockResourceRetriever:
    return MockResourceRetriever(resources)


@pytest_asyncio.fixture
async def semantic_service(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    resource_retriever: MockResourceRetriever,
):
    params = SemanticService.Params(
        semantic_storage=semantic_storage,
        episodic_storage=episode_storage,
        resource_retriever=resource_retriever,
        feature_update_interval_sec=0.05,
        feature_update_message_limit=10,
        feature_update_time_limit_sec=0.05,
    )
    service = SemanticService(params)
    yield service
    await service.stop()


@pytest.fixture
def session_id_manager() -> SessionIdManager:
    return SessionIdManager()


@pytest.fixture
def session_data(session_id_manager: SessionIdManager):
    return session_id_manager.generate_session_data(
        user_id="user-123",
        session_id="session-abc",
    )


@pytest_asyncio.fixture
async def session_manager(
    semantic_service: SemanticService,
) -> SemanticSessionManager:
    return SemanticSessionManager(
        semantic_service=semantic_service,
    )


@pytest.fixture
def mock_semantic_service() -> MagicMock:
    service = MagicMock(spec=SemanticService)
    service.add_message_to_sets = AsyncMock()
    service.search = AsyncMock(return_value=[])
    service.number_of_uningested = AsyncMock(return_value=0)
    service.add_new_feature = AsyncMock(return_value=101)
    service.get_feature = AsyncMock(return_value="feature")
    service.update_feature = AsyncMock()
    service.get_set_features = AsyncMock(return_value=["features"])
    return service


@pytest_asyncio.fixture
async def mock_session_manager(
    mock_semantic_service: MagicMock,
    semantic_storage: SemanticStorage,
) -> SemanticSessionManager:
    return SemanticSessionManager(
        semantic_service=mock_semantic_service,
    )


async def test_add_message_records_history_and_uningested_counts(
    session_manager: SemanticSessionManager,
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    session_data,
):
    # Given a session with both session and profile identifiers
    history_id = await episode_storage.add_history(
        content="Alpha memory",
        session_key="session_id",
        producer_id="profile_id",
        producer_role="dev",
    )
    await session_manager.add_message(
        session_data=session_data, episode_ids=[history_id]
    )

    # When inspecting storage for each isolation level
    profile_id = session_data.user_profile_id()
    session_id = session_data.session_id()
    profile_messages = await semantic_storage.get_history_messages(
        set_ids=[profile_id],
        is_ingested=False,
    )
    session_messages = await semantic_storage.get_history_messages(
        set_ids=[session_id],
        is_ingested=False,
    )

    # Then the history is recorded for both set ids and marked as uningested
    assert len(history_id) > 0
    assert list(profile_messages) == [history_id]
    assert list(session_messages) == [history_id]
    assert await semantic_service.number_of_uningested([profile_id]) == 1
    assert await semantic_service.number_of_uningested([session_id]) == 1


async def test_search_returns_relevant_features(
    session_manager: SemanticSessionManager,
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
    session_data,
):
    # Given semantic features stored for both profile and session
    profile_id = session_data.user_profile_id()
    session_id = session_data.session_id()
    await semantic_service.add_new_feature(
        set_id=profile_id,
        category_name="Profile",
        feature="alpha_fact",
        value="Alpha enjoys calm chats",
        tag="facts",
    )
    await semantic_service.add_new_feature(
        set_id=session_id,
        category_name="Profile",
        feature="beta_fact",
        value="Beta prefers debates",
        tag="facts",
    )

    # When searching with an alpha-focused query
    matches = await session_manager.search(
        message="Why does alpha stay calm?",
        session_data=session_data,
    )

    # Then only the alpha feature is returned and embedder search was invoked
    assert spy_embedder.search_calls == [["Why does alpha stay calm?"]]
    assert len(matches) == 1
    assert matches[0].feature_name == "alpha_fact"
    assert matches[0].set_id in {profile_id, session_id}


async def test_add_feature_applies_requested_isolation(
    session_manager: SemanticSessionManager,
    semantic_storage: SemanticStorage,
    spy_embedder: SpyEmbedder,
    session_data,
):
    # Given a profile-scoped feature request
    feature_id = await session_manager.add_feature(
        session_data=session_data,
        memory_type=IsolationType.USER,
        category_name="Profile",
        feature="tone",
        value="Alpha casual",
        tag="writing_style",
    )

    # When retrieving features for each set id
    profile_id = session_data.user_profile_id()
    session_id = session_data.session_id()
    profile_features = await semantic_storage.get_feature_set(set_ids=[profile_id])
    session_features = await semantic_storage.get_feature_set(set_ids=[session_id])

    # Then only the profile receives the new feature and embeddings were generated
    assert feature_id == profile_features[0].metadata.id
    assert profile_features[0].feature_name == "tone"
    assert profile_features[0].value == "Alpha casual"
    assert session_features == []
    assert spy_embedder.ingest_calls == [["Alpha casual"]]


async def test_delete_feature_set_removes_filtered_entries(
    session_manager: SemanticSessionManager,
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data,
):
    # Given profile and session features with distinct tags
    profile_id = session_data.user_profile_id()
    session_id = session_data.session_id()
    await semantic_service.add_new_feature(
        set_id=profile_id,
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="profile_tag",
    )
    await semantic_service.add_new_feature(
        set_id=session_id,
        category_name="Profile",
        feature="session_note",
        value="Remember to ask about projects",
        tag="session_tag",
    )

    # When deleting only the profile-tagged features
    await session_manager.delete_feature_set(
        session_data=session_data,
        memory_type=[IsolationType.USER],
        tags=["profile_tag"],
    )

    # Then profile features are cleared while session-scoped entries remain
    profile_features = await semantic_storage.get_feature_set(set_ids=[profile_id])
    session_features = await semantic_storage.get_feature_set(set_ids=[session_id])

    assert profile_features == []
    assert len(session_features) == 1
    assert session_features[0].feature_name == "session_note"


async def test_add_message_uses_all_isolations(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    history_id = "abc"
    await mock_session_manager.add_message(
        session_data=session_data, episode_ids=[history_id]
    )

    mock_semantic_service.add_message_to_sets.assert_awaited_once_with(
        history_id,
        [session_data.session_id(), session_data.user_profile_id()],
    )


async def test_add_message_with_session_only_isolation(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    await mock_session_manager.add_message(
        episode_ids=["abc"],
        session_data=session_data,
        memory_type=[IsolationType.SESSION],
    )

    mock_semantic_service.add_message_to_sets.assert_awaited_once()
    args, kwargs = mock_semantic_service.add_message_to_sets.await_args
    assert kwargs == {}
    assert args[1] == [session_data.session_id()]


async def test_search_passes_set_ids_and_filters(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    mock_semantic_service.search.return_value = ["result"]

    result = await mock_session_manager.search(
        message="Find alpha info",
        session_data=session_data,
        memory_type=[IsolationType.USER],
        tag_names=["facts"],
        feature_names=["alpha_fact"],
        limit=5,
        load_citations=True,
    )

    mock_semantic_service.search.assert_awaited_once()
    kwargs = mock_semantic_service.search.await_args.kwargs
    assert kwargs["set_ids"] == [session_data.user_profile_id()]
    assert kwargs["tag_names"] == ["facts"]
    assert kwargs["feature_names"] == ["alpha_fact"]
    assert kwargs["limit"] == 5
    assert kwargs["load_citations"] is True
    assert result == ["result"]


async def test_number_of_uningested_messages_delegates(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    mock_semantic_service.number_of_uningested.return_value = 7

    count = await mock_session_manager.number_of_uningested_messages(
        session_data=session_data,
        memory_type=[IsolationType.SESSION],
    )

    mock_semantic_service.number_of_uningested.assert_awaited_once_with(
        set_ids=[session_data.session_id()],
    )
    assert count == 7


async def test_add_feature_translates_to_single_set(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    feature_id = await mock_session_manager.add_feature(
        session_data=session_data,
        memory_type=IsolationType.USER,
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
        metadata={"source": "test"},
        citations=[1, 2],
    )

    mock_semantic_service.add_new_feature.assert_awaited_once_with(
        set_id=session_data.user_profile_id(),
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
        metadata={"source": "test"},
        citations=[1, 2],
    )
    assert feature_id == 101


async def test_get_feature_proxies_call(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
):
    result = await mock_session_manager.get_feature(42, load_citations=True)

    mock_semantic_service.get_feature.assert_awaited_once_with(42, load_citations=True)
    assert result == "feature"


async def test_update_feature_forwards_arguments(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
):
    await mock_session_manager.update_feature(
        17,
        category_name="Profile",
        feature="tone",
        value="calm",
        tag="writing_style",
        metadata={"updated": "true"},
    )

    mock_semantic_service.update_feature.assert_awaited_once_with(
        17,
        category_name="Profile",
        feature="tone",
        value="calm",
        tag="writing_style",
        metadata={"updated": "true"},
    )


async def test_get_set_features_wraps_opts(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    result = await mock_session_manager.get_set_features(
        session_data=session_data,
        memory_type=[IsolationType.USER],
        category_names=["Profile"],
        tag_names=["facts"],
        feature_names=["alpha_fact"],
    )

    mock_semantic_service.get_set_features.assert_awaited_once()
    opts = mock_semantic_service.get_set_features.await_args.args[0]
    assert opts.set_ids == [session_data.user_profile_id()]
    assert opts.category_names == ["Profile"]
    assert opts.tags == ["facts"]
    assert opts.feature_names == ["alpha_fact"]
    assert result == ["features"]
