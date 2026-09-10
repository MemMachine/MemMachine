"""Tests for USearchVectorSearchEngine."""

import math
from pathlib import Path

import pytest

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
    def test_construction(self):
        USearchVectorSearchEngine(num_dimensions=NDIM)


# -- Add --


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_single(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [1.0, 0.0, 0.0]})
        result = await _search_one(engine, [1.0, 0.0, 0.0], limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_add_batch(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({10: [1, 0, 0], 20: [0, 1, 0], 30: [0, 0, 1]})
        result = await _search_one(engine, [1, 0, 0], limit=3)
        assert {m.key for m in result.matches} == {10, 20, 30}

    @pytest.mark.asyncio
    async def test_remove_then_add_replaces_key(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [1.0, 0.0, 0.0]})
        await engine.remove([1])
        await engine.add({1: [0.0, 1.0, 0.0]})
        result = await _search_one(engine, _normalize([0, 1, 0]), limit=1)
        assert result.matches[0].key == 1
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_add_empty(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches == []


# -- Remove --


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_existing(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [1, 0, 0], 2: [0, 1, 0]})
        await engine.remove([1])
        result = await _search_one(engine, [1, 0, 0], limit=2)
        keys = {m.key for m in result.matches}
        assert 1 not in keys
        assert 2 in keys

    @pytest.mark.asyncio
    async def test_remove_missing_is_ignored(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [1, 0, 0]})
        await engine.remove([99, 100])
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_remove_all(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]})
        await engine.remove([1, 2, 3])
        result = await _search_one(engine, [1, 0, 0], limit=3)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_remove_empty_iterable(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [1, 0, 0]})
        await engine.remove([])
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches[0].key == 1


# -- Search: Cosine --


# -- Search: non-unit input --


class TestNonUnitVectors:
    """Vectors are scaled to unit length on the way in, so scores are cosines.

    Every other test here passes unit vectors, under which the scaling is a
    no-op. These pass magnitudes and pin that the score is the cosine of the
    originals rather than their raw inner product.
    """

    @pytest.mark.asyncio
    async def test_stored_magnitude_does_not_reach_the_score(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [3.0, 4.0, 0.0]})

        result = await _search_one(engine, [1.0, 0.0, 0.0], limit=1)
        # Raw inner product would be 3.0; the cosine is 0.6.
        assert result.matches[0].cosine_similarity == pytest.approx(0.6, abs=1e-3)

    @pytest.mark.asyncio
    async def test_query_magnitude_does_not_reach_the_score(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: _normalize([1.0, 0.0, 0.0])})

        result = await _search_one(engine, [50.0, 0.0, 0.0], limit=1)
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_allowlist_scores_are_cosines(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [3.0, 4.0, 0.0], 2: [0.0, 0.0, 9.0]})

        result = await _search_one(engine, [2.0, 0.0, 0.0], limit=2, allowlist=[1, 2])
        scores = {m.key: m.cosine_similarity for m in result.matches}
        assert scores[1] == pytest.approx(0.6, abs=1e-3)
        assert scores[2] == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_scores_stay_within_the_cosine_range(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({key: _normalize([1.0, 0.0, 0.0]) for key in range(20)})

        result = await _search_one(
            engine, _normalize([1.0, 0.0, 0.0]), limit=20, allowlist=list(range(20))
        )
        assert all(-1.0 <= m.cosine_similarity <= 1.0 for m in result.matches)


# -- Search: allowlist --


class TestSearchAllowlist:
    @pytest.mark.asyncio
    async def test_allowlist_restricts_results(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add(
            {
                1: _normalize([1, 0, 0]),
                2: _normalize([0, 1, 0]),
                3: _normalize([0, 0, 1]),
            }
        )
        result = await _search_one(
            engine, _normalize([1, 0, 0]), limit=3, allowlist=[2, 3]
        )
        assert {m.key for m in result.matches} == {2, 3}

    @pytest.mark.asyncio
    async def test_allowlist_excludes_best_match(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add(
            {
                1: _normalize([1, 0, 0]),
                2: _normalize([0, 1, 0]),
                3: _normalize([0, 0, 1]),
            }
        )
        result = await _search_one(
            engine, _normalize([1, 0, 0]), limit=1, allowlist=[2, 3]
        )
        assert len(result.matches) == 1
        assert result.matches[0].key in {2, 3}

    @pytest.mark.asyncio
    async def test_empty_allowlist_returns_nothing(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: _normalize([1, 0, 0])})
        result = await _search_one(engine, _normalize([1, 0, 0]), limit=1, allowlist=[])
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_missing_allowlist_keys_ignored(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: _normalize([1, 0, 0])})
        result = await _search_one(
            engine, _normalize([1, 0, 0]), limit=2, allowlist=[1, 99]
        )
        assert [m.key for m in result.matches] == [1]


class TestSearchCosine:
    @pytest.mark.asyncio
    async def test_basic_knn(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
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
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)
        assert result.matches[1].cosine_similarity == pytest.approx(
            1.0 / math.sqrt(2), abs=0.01
        )

    @pytest.mark.asyncio
    async def test_scores_are_cosine_similarity(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        v1 = _normalize([1, 0, 0])
        v2 = _normalize([0, 1, 0])
        await engine.add({1: v1, 2: v2})
        result = await _search_one(engine, v1, limit=2)
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)
        assert result.matches[1].cosine_similarity == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_scores_ordered_best_first(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add(
            {
                1: _normalize([1, 0, 0]),
                2: _normalize([0, 1, 0]),
                3: _normalize([1, 1, 0]),
            }
        )
        result = await _search_one(engine, _normalize([1, 0, 0]), limit=3)
        for i in range(len(result.matches) - 1):
            assert (
                result.matches[i].cosine_similarity
                >= result.matches[i + 1].cosine_similarity
            )

    @pytest.mark.asyncio
    async def test_k_larger_than_index(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=10)
        assert len(result.matches) == 1

    @pytest.mark.asyncio
    async def test_search_empty_index(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        result = await _search_one(engine, [1, 0, 0], limit=5)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_batched_search(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: _normalize([1, 0, 0]), 2: _normalize([0, 1, 0])})
        results = await engine.search(
            [_normalize([1, 0, 0]), _normalize([0, 1, 0])], limit=1
        )
        assert len(results) == 2
        assert results[0].matches[0].key == 1
        assert results[1].matches[0].key == 2


# -- Persistence --


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path: Path):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: _normalize([1, 0, 0]), 2: _normalize([0, 1, 0])})

        path = str(tmp_path / "test.idx")
        await engine.save(path)

        engine2 = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine2.load(path)

        result = await _search_one(engine2, _normalize([1, 0, 0]), limit=2)
        assert {m.key for m in result.matches} == {1, 2}
        assert result.matches[0].key == 1
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_save_leaves_no_temp_file(self, tmp_path: Path):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: _normalize([1, 0, 0])})

        path = tmp_path / "test.idx"
        await engine.save(str(path))

        assert path.exists()
        assert not (tmp_path / "test.idx.tmp").exists()

    @pytest.mark.asyncio
    async def test_load_clears_stale_temp_file(self, tmp_path: Path):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({1: _normalize([1, 0, 0]), 2: _normalize([0, 1, 0])})

        path = tmp_path / "test.idx"
        await engine.save(str(path))

        # A temp file left behind by a previously interrupted save.
        stale_temp = tmp_path / "test.idx.tmp"
        stale_temp.write_text("STALE")

        engine2 = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine2.load(str(path))

        assert not stale_temp.exists()
        result = await _search_one(engine2, _normalize([1, 0, 0]), limit=2)
        assert {m.key for m in result.matches} == {1, 2}


# -- SearchResult types --


class TestSearchResultTypes:
    @pytest.mark.asyncio
    async def test_keys_are_ints(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({42: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert isinstance(result.matches[0].key, int)

    @pytest.mark.asyncio
    async def test_scores_are_floats(self):
        engine = USearchVectorSearchEngine(num_dimensions=NDIM)
        await engine.add({42: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert isinstance(result.matches[0].cosine_similarity, float)
