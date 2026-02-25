"""Unit tests for entity type passthrough in DeclarativeMemory.add_episodes().

These tests verify that derivative nodes inherit entity_types from
episode user_metadata per design decision D2 (no LLM call in episodic
path -- entity type is taken directly from user_metadata).
"""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import pytest

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder.embedder import Embedder
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store.data_types import (
    Edge,
    Node,
    OrderedPropertyValue,
)
from memmachine.common.vector_graph_store.vector_graph_store import VectorGraphStore
from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    DeclarativeMemoryParams,
    Episode,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CapturingVectorGraphStore(VectorGraphStore):
    """VectorGraphStore that captures add_nodes calls for assertions."""

    def __init__(self) -> None:
        self.added_nodes_calls: list[tuple[str, list[Node]]] = []
        self.added_edges_calls: list[tuple[str, str, str, list[Edge]]] = []

    async def add_nodes(self, *, collection: str, nodes: Iterable[Node]) -> None:
        self.added_nodes_calls.append((collection, list(nodes)))

    async def add_edges(
        self,
        *,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        self.added_edges_calls.append(
            (relation, source_collection, target_collection, list(edges))
        )

    # ---- stubs for abstract methods not exercised by add_episodes ----

    async def search_similar_nodes(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return []  # pragma: no cover

    async def search_related_nodes(
        self,
        *,
        relation: str,
        other_collection: str,
        this_collection: str,
        this_node_uid: str,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        edge_property_filter: FilterExpr | None = None,
        node_property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return []  # pragma: no cover

    async def search_directional_nodes(
        self,
        *,
        collection: str,
        by_properties: Iterable[str],
        starting_at: Iterable[OrderedPropertyValue | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return []  # pragma: no cover

    async def search_matching_nodes(
        self,
        *,
        collection: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return []  # pragma: no cover

    async def get_nodes(
        self, *, collection: str, node_uids: Iterable[str]
    ) -> list[Node]:
        return []  # pragma: no cover

    async def delete_nodes(self, *, collection: str, node_uids: Iterable[str]) -> None:
        pass  # pragma: no cover

    async def delete_all_data(self) -> None:
        pass  # pragma: no cover

    async def close(self) -> None:
        pass  # pragma: no cover


class _FakeEmbedder(Embedder):
    """Embedder that returns deterministic 2-d embeddings."""

    async def ingest_embed(
        self,
        inputs: Iterable[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [[float(len(s)), -float(len(s))] for s in inputs]

    async def search_embed(
        self,
        queries: Iterable[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [[float(len(q)), -float(len(q))] for q in queries]

    @property
    def model_id(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class _FakeReranker(Reranker):
    """Reranker that returns descending scores."""

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return [1.0 / (i + 1) for i in range(len(candidates))]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> _CapturingVectorGraphStore:
    return _CapturingVectorGraphStore()


@pytest.fixture
def declarative_memory(store: _CapturingVectorGraphStore) -> DeclarativeMemory:
    return DeclarativeMemory(
        DeclarativeMemoryParams(
            session_id="test_session",
            vector_graph_store=store,
            embedder=_FakeEmbedder(),
            reranker=_FakeReranker(),
        ),
    )


def _get_derivative_nodes(
    store: _CapturingVectorGraphStore,
) -> list[Node]:
    """Return derivative nodes from captured add_nodes calls."""
    for collection, nodes in store.added_nodes_calls:
        if collection.startswith("Derivative_"):
            return nodes
    return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derivative_inherits_entity_type_from_episode_metadata(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """Episode with entity_type in user_metadata → derivative gets entity_types=["Event"]."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="The meeting happened yesterday.",
            user_metadata={"entity_type": "Event"},
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 1
    assert derivative_nodes[0].entity_types == ["Event"]


@pytest.mark.asyncio
async def test_derivative_empty_entity_types_when_no_entity_type_in_metadata(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """Episode without entity_type key → derivative gets entity_types=[]."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="Hello world.",
            user_metadata={"other_key": "value"},
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 1
    assert derivative_nodes[0].entity_types == []


@pytest.mark.asyncio
async def test_derivative_empty_entity_types_when_metadata_is_none(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """Episode with user_metadata=None → derivative gets entity_types=[]."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="Hello world.",
            user_metadata=None,
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 1
    assert derivative_nodes[0].entity_types == []


@pytest.mark.asyncio
async def test_derivative_empty_entity_types_when_metadata_is_empty_dict(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """Episode with user_metadata={} → derivative gets entity_types=[]."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="Some text.",
            user_metadata={},
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 1
    assert derivative_nodes[0].entity_types == []


@pytest.mark.asyncio
async def test_multiple_episodes_mixed_entity_types(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """Multiple episodes: one with entity_type, one without → correct propagation."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="I met Bob in Paris.",
            user_metadata={"entity_type": "Person"},
        ),
        Episode(
            uid="ep2",
            timestamp=now + timedelta(seconds=1),
            source="Bob",
            content_type=ContentType.MESSAGE,
            content="No metadata here.",
            user_metadata={"unrelated": True},
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 2

    # Derivatives are ordered same as episodes (sorted by timestamp, uid).
    ep1_derivative = derivative_nodes[0]
    ep2_derivative = derivative_nodes[1]

    assert ep1_derivative.entity_types == ["Person"]
    assert ep2_derivative.entity_types == []


@pytest.mark.asyncio
async def test_derivative_ignores_non_string_entity_type(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """Episode with entity_type as non-string (e.g. int) → derivative gets entity_types=[]."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="Bad entity type.",
            user_metadata={"entity_type": 42},
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 1
    assert derivative_nodes[0].entity_types == []


@pytest.mark.asyncio
async def test_derivative_ignores_empty_string_entity_type(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """Episode with entity_type="" → derivative gets entity_types=[] (empty string ignored)."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="Alice",
            content_type=ContentType.MESSAGE,
            content="Empty entity type.",
            user_metadata={"entity_type": ""},
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 1
    assert derivative_nodes[0].entity_types == []


@pytest.mark.asyncio
async def test_text_episode_entity_type_passthrough(
    declarative_memory: DeclarativeMemory,
    store: _CapturingVectorGraphStore,
) -> None:
    """TEXT content type episode also passes entity_type to derivative."""
    now = datetime.now(tz=UTC)
    episodes = [
        Episode(
            uid="ep1",
            timestamp=now,
            source="document",
            content_type=ContentType.TEXT,
            content="The Eiffel Tower is in Paris.",
            user_metadata={"entity_type": "Location"},
        ),
    ]
    await declarative_memory.add_episodes(episodes)

    derivative_nodes = _get_derivative_nodes(store)
    assert len(derivative_nodes) == 1
    assert derivative_nodes[0].entity_types == ["Location"]
