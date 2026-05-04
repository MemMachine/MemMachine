"""Tests for USearchVectorSearchEngine."""

import math
from pathlib import Path

import pytest

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
    USearchVectorSearchEngine,
)

NDIM = 3


def _normalize(v: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in v))
    return [x / magnitude for x in v]


async def _search_one(engine, vector, limit=10, **kwargs):
    """Helper: search a single vector, return the one SearchResult."""
    results = await engine.search([vector], limit=limit, **kwargs)
    return results[0]


# -- Construction --


class TestConstruction:
    def test_supported_metrics(self):
        for metric in (
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.DOT,
        ):
            USearchVectorSearchEngine(num_dimensions=NDIM, similarity_metric=metric)

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            USearchVectorSearchEngine(
                num_dimensions=NDIM, similarity_metric=SimilarityMetric.MANHATTAN
            )


# -- Add --


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_single(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1.0, 0.0, 0.0]})
        result = await _search_one(engine, [1.0, 0.0, 0.0], limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_add_batch(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({10: [1, 0, 0], 20: [0, 1, 0], 30: [0, 0, 1]})
        result = await _search_one(engine, [1, 0, 0], limit=3)
        assert {m.key for m in result.matches} == {10, 20, 30}

    @pytest.mark.asyncio
    async def test_remove_then_add_replaces_key(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1.0, 0.0, 0.0]})
        await engine.remove([1])
        await engine.add({1: [0.0, 1.0, 0.0]})
        result = await _search_one(engine, _normalize([0, 1, 0]), limit=1)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_add_empty(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches == []


# -- Remove --


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_existing(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0], 2: [0, 1, 0]})
        await engine.remove([1])
        result = await _search_one(engine, [1, 0, 0], limit=2)
        keys = {m.key for m in result.matches}
        assert 1 not in keys
        assert 2 in keys

    @pytest.mark.asyncio
    async def test_remove_missing_is_ignored(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0]})
        await engine.remove([99, 100])
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_remove_all(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]})
        await engine.remove([1, 2, 3])
        result = await _search_one(engine, [1, 0, 0], limit=3)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_remove_empty_iterable(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0]})
        await engine.remove([])
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches[0].key == 1


# -- Search: Cosine --


class TestSearchCosine:
    @pytest.mark.asyncio
    async def test_basic_knn(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add(
            {
                1: _normalize([1, 0, 0]),
                2: _normalize([0, 1, 0]),
                3: _normalize([1, 1, 0]),
            }
        )
        result = await _search_one(engine, _normalize([1, 0, 0]), limit=3)
        assert len(result.matches) == 3
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        assert result.matches[1].score == pytest.approx(1.0 / math.sqrt(2), abs=0.01)

    @pytest.mark.asyncio
    async def test_scores_are_cosine_similarity(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        v1 = _normalize([1, 0, 0])
        v2 = _normalize([0, 1, 0])
        await engine.add({1: v1, 2: v2})
        result = await _search_one(engine, v1, limit=2)
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        assert result.matches[1].score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_scores_ordered_best_first(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add(
            {
                1: _normalize([1, 0, 0]),
                2: _normalize([0, 1, 0]),
                3: _normalize([1, 1, 0]),
            }
        )
        result = await _search_one(engine, _normalize([1, 0, 0]), limit=3)
        for i in range(len(result.matches) - 1):
            assert result.matches[i].score >= result.matches[i + 1].score

    @pytest.mark.asyncio
    async def test_k_larger_than_index(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=10)
        assert len(result.matches) == 1

    @pytest.mark.asyncio
    async def test_search_empty_index(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        result = await _search_one(engine, [1, 0, 0], limit=5)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_batched_search(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: _normalize([1, 0, 0]), 2: _normalize([0, 1, 0])})
        results = await engine.search(
            [_normalize([1, 0, 0]), _normalize([0, 1, 0])], limit=1
        )
        assert len(results) == 2
        assert results[0].matches[0].key == 1
        assert results[1].matches[0].key == 2


# -- Search: Euclidean --


class TestSearchEuclidean:
    @pytest.mark.asyncio
    async def test_scores_are_euclidean_distance(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.EUCLIDEAN
        )
        await engine.add({1: [0, 0, 0], 2: [3, 4, 0]})
        result = await _search_one(engine, [0, 0, 0], limit=2)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(0.0, abs=0.01)
        assert result.matches[1].score == pytest.approx(5.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_scores_ordered_best_first(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.EUCLIDEAN
        )
        await engine.add({1: [0, 0, 0], 2: [1, 0, 0], 3: [3, 4, 0]})
        result = await _search_one(engine, [0, 0, 0], limit=3)
        for i in range(len(result.matches) - 1):
            assert result.matches[i].score <= result.matches[i + 1].score


# -- Search: Dot product --


class TestSearchDot:
    @pytest.mark.asyncio
    async def test_scores_are_inner_product(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.DOT
        )
        v1 = _normalize([1, 0, 0])
        v2 = _normalize([0, 1, 0])
        await engine.add({1: v1, 2: v2})
        result = await _search_one(engine, v1, limit=2)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        assert result.matches[1].score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_scores_ordered_best_first(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.DOT
        )
        v1 = _normalize([1, 0, 0])
        v2 = _normalize([1, 1, 0])
        v3 = _normalize([0, 1, 0])
        await engine.add({1: v1, 2: v2, 3: v3})
        result = await _search_one(engine, v1, limit=3)
        for i in range(len(result.matches) - 1):
            assert result.matches[i].score >= result.matches[i + 1].score


# -- Persistence --


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path: Path):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: _normalize([1, 0, 0]), 2: _normalize([0, 1, 0])})

        path = str(tmp_path / "test.idx")
        await engine.save(path)

        engine2 = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine2.load(path)

        result = await _search_one(engine2, _normalize([1, 0, 0]), limit=2)
        assert {m.key for m in result.matches} == {1, 2}
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)


# -- SearchResult types --


class TestSearchResultTypes:
    @pytest.mark.asyncio
    async def test_keys_are_ints(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({42: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert isinstance(result.matches[0].key, int)

    @pytest.mark.asyncio
    async def test_scores_are_floats(self):
        engine = USearchVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({42: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert isinstance(result.matches[0].score, float)
