"""Unit tests for the Embedder base class."""

from typing import Any

import pytest

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder.embedder import Embedder


class MockEmbedder(Embedder):
    """Mock implementation of Embedder for testing."""

    def __init__(self, batch_size: int | None = None) -> None:
        """Initialize the mock embedder."""
        super().__init__(batch_size=batch_size)
        self.ingest_calls: list[list[Any]] = []
        self.search_calls: list[list[Any]] = []

    async def _ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        self.ingest_calls.append(inputs)
        # Return dummy embeddings of dimension 2
        return [[0.1, 0.2] for _ in inputs]

    async def _search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        self.search_calls.append(queries)
        # Return dummy embeddings of dimension 2
        return [[0.3, 0.4] for _ in queries]

    @property
    def model_id(self) -> str:
        return "mock-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


@pytest.mark.asyncio
async def test_embedder_batching() -> None:
    """Test that the embedder correctly batches inputs."""
    batch_size = 2
    embedder = MockEmbedder(batch_size=batch_size)

    # Test ingest_embed batching
    inputs = ["a", "b", "c", "d", "e"]
    # Expected batches: ["a", "b"], ["c", "d"], ["e"]

    embeddings = await embedder.ingest_embed(inputs)

    assert len(embeddings) == 5
    assert len(embedder.ingest_calls) == 3
    assert embedder.ingest_calls[0] == ["a", "b"]
    assert embedder.ingest_calls[1] == ["c", "d"]
    assert embedder.ingest_calls[2] == ["e"]

    # Test search_embed batching
    queries = ["q1", "q2", "q3"]
    # Expected batches: ["q1", "q2"], ["q3"]

    search_embeddings = await embedder.search_embed(queries)

    assert len(search_embeddings) == 3
    assert len(embedder.search_calls) == 2
    assert embedder.search_calls[0] == ["q1", "q2"]
    assert embedder.search_calls[1] == ["q3"]


@pytest.mark.asyncio
async def test_embedder_no_batching() -> None:
    """Test that the embedder works correctly without batching."""
    embedder = MockEmbedder(batch_size=None)

    inputs = ["a", "b", "c"]
    embeddings = await embedder.ingest_embed(inputs)

    assert len(embeddings) == 3
    assert len(embedder.ingest_calls) == 1
    assert embedder.ingest_calls[0] == ["a", "b", "c"]
