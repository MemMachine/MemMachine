"""Unit tests for the SemanticService using an in-memory storage backend."""

import pytest
import pytest_asyncio

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import EpisodeEntry, EpisodeStorage
from memmachine.common.filter.filter_parser import parse_filter
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    Resources,
    SemanticCategory,
    SemanticPrompt,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    MockResourceRetriever,
)

pytestmark = pytest.mark.asyncio


class SpyEmbedder(Embedder):
    """Test double that records calls and produces deterministic embeddings."""

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
        update_prompt="update-semantic-memory",
        consolidation_prompt="consolidate-semantic-memory",
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
):
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
    resource_retriever: MockResourceRetriever,
    episode_storage: EpisodeStorage,
) -> SemanticService:
    params = SemanticService.Params(
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        resource_retriever=resource_retriever,
        feature_update_interval_sec=0.05,
        feature_update_message_limit=10,
        feature_update_time_limit_sec=0.05,
    )
    service = SemanticService(params)
    yield service
    await service.stop()


async def add_history(history_storage: EpisodeStorage, content: str):
    episodes = await history_storage.add_episodes(
        session_key="session_id",
        episodes=[
            EpisodeEntry(
                content=content,
                producer_id="profile_id",
                producer_role="dev",
            )
        ],
    )
    return episodes[0].uid


async def test_add_new_feature_stores_entry(
    semantic_service: SemanticService,
    resource_retriever: MockResourceRetriever,
    spy_embedder: SpyEmbedder,
):
    # Given a fresh semantic service
    feature_id = await semantic_service.add_new_feature(
        set_id="user-123",
        category_name="Profile",
        feature="tone",
        value="Alpha voice",
        tag="writing_style",
        metadata={"source": "test"},
    )

    # When retrieving the stored features
    features = await semantic_service.get_set_features(
        filter_expr=parse_filter("set_id IN ('user-123')"),
    )

    # Then the feature is persisted with embeddings recorded
    assert resource_retriever.seen_ids == ["user-123"]
    assert spy_embedder.ingest_calls == [["Alpha voice"]]
    assert len(features) == 1
    feature = features[0]
    assert feature.metadata.id == feature_id
    assert feature.set_id == "user-123"
    assert feature.feature_name == "tone"
    assert feature.value == "Alpha voice"
    assert feature.tag == "writing_style"


async def test_get_set_features_filters_by_tag(
    semantic_service: SemanticService,
):
    # Given multiple features under a single set
    await semantic_service.add_new_feature(
        set_id="user-42",
        category_name="Profile",
        feature="tone",
        value="Alpha friendly",
        tag="writing_style",
    )
    await semantic_service.add_new_feature(
        set_id="user-42",
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="personal_info",
    )

    # When filtering on a specific tag
    filtered = await semantic_service.get_set_features(
        filter_expr=parse_filter("set_id IN ('user-42') AND tag IN ('writing_style')"),
    )

    # Then only matching features are returned
    assert len(filtered) == 1
    assert filtered[0].feature_name == "tone"
    assert filtered[0].tag == "writing_style"


async def test_update_feature_changes_value_and_embedding(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
):
    # Given an existing feature
    feature_id = await semantic_service.add_new_feature(
        set_id="user-7",
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )

    # When updating the value
    await semantic_service.update_feature(
        feature_id,
        set_id="user-7",
        category_name="Profile",
        value="Alpha energetic",
        tag="writing_style",
    )

    # Then the feature reflects the new value and re-embeds
    feature = await semantic_service.get_feature(feature_id, load_citations=False)
    assert feature is not None
    assert feature.value == "Alpha energetic"
    assert spy_embedder.ingest_calls == [["Alpha calm"], ["Alpha energetic"]]


async def test_delete_features_removes_selected_entries(
    semantic_service: SemanticService,
):
    # Given two stored features
    to_remove = await semantic_service.add_new_feature(
        set_id="user-55",
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )
    to_keep = await semantic_service.add_new_feature(
        set_id="user-55",
        category_name="Profile",
        feature="hobby",
        value="Gardening",
        tag="personal_info",
    )

    # When deleting one feature by id
    await semantic_service.delete_features([to_remove])

    # Then the targeted feature is gone and the other remains
    assert await semantic_service.get_feature(to_remove, load_citations=False) is None
    assert await semantic_service.get_feature(to_keep, load_citations=False) is not None


async def test_delete_feature_set_applies_filters(
    semantic_service: SemanticService,
):
    # Given two features with different tags
    await semantic_service.add_new_feature(
        set_id="user-88",
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )
    await semantic_service.add_new_feature(
        set_id="user-88",
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="personal_info",
    )

    # When deleting by tag filter
    filter_str = "set_id in ('user-88') AND tag in ('writing_style')"
    filter_expr = parse_filter(filter_str)
    assert filter_expr is not None

    await semantic_service.delete_feature_set(
        filter_expr=filter_expr,
    )

    # Then only the non-matching feature remains
    filter_str = "set_id in ('user-88')"
    filter_expr = parse_filter(filter_str)
    assert filter_expr is not None

    remaining = await semantic_service.get_set_features(
        filter_expr=filter_expr,
    )
    assert len(remaining) == 1
    assert remaining[0].feature_name == "favorite_color"


async def test_add_messages_tracks_uningested_counts(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
):
    # Given a stored history message
    history_id = await add_history(
        history_storage=episode_storage,
        content="Alpha memory",
    )

    # When associating the message to a set
    await semantic_service.add_messages(set_id="user-21", history_ids=[history_id])

    # Then the set reports one uningested message
    assert await semantic_service.number_of_uningested(["user-21"]) == 1

    # When the message is marked ingested
    await semantic_storage.mark_messages_ingested(
        set_id="user-21",
        history_ids=[history_id],
    )

    # Then the uningested count drops to zero
    assert await semantic_service.number_of_uningested(["user-21"]) == 0


async def test_add_message_to_sets_supports_multiple_targets(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
):
    # Given a history entry
    history_id = await add_history(
        history_storage=episode_storage,
        content="Alpha shared memory",
    )

    # When linking the message to multiple sets
    await semantic_service.add_message_to_sets(
        history_id=history_id,
        set_ids=["user-a", "user-b"],
    )

    # Then all sets report the pending ingestion
    assert await semantic_service.number_of_uningested(["user-a"]) == 1
    assert await semantic_service.number_of_uningested(["user-b"]) == 1


async def test_search_returns_matching_features(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
):
    # Given a set with two features
    await semantic_service.add_new_feature(
        set_id="user-search",
        category_name="Profile",
        feature="alpha_fact",
        value="Alpha prefers calm conversations.",
        tag="facts",
    )
    await semantic_service.add_new_feature(
        set_id="user-search",
        category_name="Profile",
        feature="beta_fact",
        value="Beta enjoys debates.",
        tag="facts",
    )

    # When searching with an alpha-focused query
    results = await semantic_service.search(
        set_ids=["user-search"],
        query="Why does alpha prefer quiet chats?",
        min_distance=0.5,
    )

    # Then only the matching feature is returned using the query embedding
    assert spy_embedder.search_calls == [["Why does alpha prefer quiet chats?"]]
    assert len(results) == 1
    assert results[0].feature_name == "alpha_fact"
