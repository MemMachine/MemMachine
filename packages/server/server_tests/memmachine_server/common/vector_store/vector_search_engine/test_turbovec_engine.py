"""Tests for TurboVecVectorSearchEngine.

turbovec stores TurboQuant-compressed vectors, so scores are approximate: a
self-match lands near (and may slightly exceed) the exact value, and orthogonal
pairs land near zero rather than exactly zero. Tests therefore assert ranking
and membership (robust under quantization) and only loosely assert score
magnitudes.
"""

import math
from pathlib import Path

import pytest

pytest.importorskip("turbovec")

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store.vector_search_engine.turbovec_engine import (
    TurboVecVectorSearchEngine,
)

# turbovec requires the dimensionality to be a positive multiple of 8.
NDIM = 8

# Tolerance for quantized scores (a self-match lands ~1.0, e.g. 1.0003).
QUANT_ABS = 0.05


def _normalize(v: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in v))
    return [x / magnitude for x in v]


def _one_hot(index: int, value: float = 1.0) -> list[float]:
    """A length-NDIM vector with `value` at `index`, zeros elsewhere."""
    vector = [0.0] * NDIM
    vector[index] = value
    return vector


async def _search_one(engine, vector, limit=10, **kwargs):
    """Helper: search a single vector, return the one SearchResult."""
    results = await engine.search([vector], limit=limit, **kwargs)
    return results[0]


def _make_engine(metric=SimilarityMetric.COSINE, **kwargs):
    return TurboVecVectorSearchEngine(
        num_dimensions=NDIM, similarity_metric=metric, **kwargs
    )


# -- Construction --


class TestConstruction:
    def test_supported_metrics(self):
        for metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            _make_engine(metric)

    def test_unsupported_metric_raises(self):
        for metric in (SimilarityMetric.EUCLIDEAN, SimilarityMetric.MANHATTAN):
            with pytest.raises(NotImplementedError, match="does not support"):
                _make_engine(metric)

    def test_valid_bit_widths(self):
        for bit_width in (2, 3, 4):
            _make_engine(bit_width=bit_width)

    def test_invalid_bit_width_raises(self):
        with pytest.raises(ValueError, match="bit_width"):
            _make_engine(bit_width=5)

    def test_dimensions_must_be_multiple_of_8(self):
        with pytest.raises(ValueError, match="multiple of 8"):
            TurboVecVectorSearchEngine(
                num_dimensions=7, similarity_metric=SimilarityMetric.COSINE
            )


# -- Add --


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_single(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_add_batch(self):
        engine = _make_engine()
        await engine.add(
            {
                10: _normalize(_one_hot(0)),
                20: _normalize(_one_hot(1)),
                30: _normalize(_one_hot(2)),
            }
        )
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        assert {m.key for m in result.matches} == {10, 20, 30}

    @pytest.mark.asyncio
    async def test_add_empty(self):
        engine = _make_engine()
        await engine.add({})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=1)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_remove_then_add_replaces_key(self):
        # The store's upsert path is remove-then-re-add of the same key.
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        await engine.remove([1])
        await engine.add({1: _normalize(_one_hot(1))})
        result = await _search_one(engine, _normalize(_one_hot(1)), limit=1)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)

    @pytest.mark.asyncio
    async def test_repeated_reupsert_of_same_key(self):
        engine = _make_engine()
        for index in range(NDIM):
            await engine.remove([7])
            await engine.add({7: _normalize(_one_hot(index))})
        result = await _search_one(engine, _normalize(_one_hot(NDIM - 1)), limit=1)
        assert result.matches[0].key == 7


# -- Remove --


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_existing(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})
        await engine.remove([1])
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=2)
        keys = {m.key for m in result.matches}
        assert 1 not in keys
        assert 2 in keys

    @pytest.mark.asyncio
    async def test_remove_missing_is_ignored(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        await engine.remove([99, 100])
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_remove_all(self):
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize(_one_hot(2)),
            }
        )
        await engine.remove([1, 2, 3])
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_remove_empty_iterable(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        await engine.remove([])
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=1)
        assert result.matches[0].key == 1


# -- Search: Cosine --


class TestSearchCosine:
    @pytest.mark.asyncio
    async def test_basic_knn(self):
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize([1, 1, 0, 0, 0, 0, 0, 0]),
            }
        )
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        assert len(result.matches) == 3
        # Self-match first (~1.0), then the 45-degree vector, then orthogonal.
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)
        assert result.matches[1].key == 3
        assert result.matches[2].key == 2

    @pytest.mark.asyncio
    async def test_orthogonal_scores_near_zero(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=2)
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)
        assert result.matches[1].score == pytest.approx(0.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_scores_ordered_best_first(self):
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize([1, 1, 0, 0, 0, 0, 0, 0]),
            }
        )
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        for i in range(len(result.matches) - 1):
            assert result.matches[i].score >= result.matches[i + 1].score

    @pytest.mark.asyncio
    async def test_k_larger_than_index(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=10)
        assert len(result.matches) == 1

    @pytest.mark.asyncio
    async def test_search_empty_index(self):
        engine = _make_engine()
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=5)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_limit_zero_returns_empty(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=0)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_empty_query_list(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        results = await engine.search([], limit=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_batched_search(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})
        results = await engine.search(
            [_normalize(_one_hot(0)), _normalize(_one_hot(1))], limit=1
        )
        assert len(results) == 2
        assert results[0].matches[0].key == 1
        assert results[1].matches[0].key == 2


# -- Search: Dot product --


class TestSearchDot:
    @pytest.mark.asyncio
    async def test_self_match_highest(self):
        engine = _make_engine(SimilarityMetric.DOT)
        v1 = _normalize(_one_hot(0))
        v2 = _normalize(_one_hot(1))
        await engine.add({1: v1, 2: v2})
        result = await _search_one(engine, v1, limit=2)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)
        assert result.matches[1].score == pytest.approx(0.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_scores_ordered_best_first(self):
        engine = _make_engine(SimilarityMetric.DOT)
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize([1, 1, 0, 0, 0, 0, 0, 0]),
                3: _normalize(_one_hot(1)),
            }
        )
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        for i in range(len(result.matches) - 1):
            assert result.matches[i].score >= result.matches[i + 1].score


# -- Search: allowed_keys filtering (turbovec's overfetch path) --


class TestSearchFiltered:
    @pytest.mark.asyncio
    async def test_allowed_keys_restricts_results(self):
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize(_one_hot(2)),
                4: _normalize(_one_hot(3)),
            }
        )
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=4, allowed_keys={2, 3}
        )
        assert {m.key for m in result.matches} == {2, 3}

    @pytest.mark.asyncio
    async def test_allowed_keys_excludes_best_match(self):
        # The closest vector (key 1) is disallowed, so it must be skipped even
        # though it would otherwise rank first.
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize(_one_hot(2)),
            }
        )
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=1, allowed_keys={2, 3}
        )
        assert len(result.matches) == 1
        assert result.matches[0].key in {2, 3}

    @pytest.mark.asyncio
    async def test_allowed_keys_empty_returns_nothing(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=2, allowed_keys=set()
        )
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_allowed_keys_found_beyond_overfetch_window(self):
        # Only one key in a larger index is allowed; the overfetch loop must
        # widen its fetch window until it surfaces that single key.
        engine = _make_engine()
        vectors = {
            index: _normalize(_one_hot(index % NDIM, value=1.0 + index))
            for index in range(2 * NDIM)
        }
        await engine.add(vectors)
        target = 2 * NDIM - 1
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=1, allowed_keys={target}
        )
        assert [m.key for m in result.matches] == [target]


# -- get_vectors --


class TestGetVectors:
    @pytest.mark.asyncio
    async def test_get_vectors_raises(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        with pytest.raises(NotImplementedError, match="cannot be retrieved"):
            await engine.get_vectors([1])


# -- Persistence --


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path: Path):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})

        path = str(tmp_path / "test.idx")
        await engine.save(path)

        engine2 = _make_engine()
        await engine2.load(path)

        result = await _search_one(engine2, _normalize(_one_hot(0)), limit=2)
        assert {m.key for m in result.matches} == {1, 2}
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)

    @pytest.mark.asyncio
    async def test_load_replaces_existing_index(self, tmp_path: Path):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        path = str(tmp_path / "test.idx")
        await engine.save(path)

        engine2 = _make_engine()
        await engine2.add({2: _normalize(_one_hot(1))})
        await engine2.load(path)

        result = await _search_one(engine2, _normalize(_one_hot(0)), limit=5)
        keys = {m.key for m in result.matches}
        assert keys == {1}


# -- SearchResult types --


class TestSearchResultTypes:
    @pytest.mark.asyncio
    async def test_keys_are_ints(self):
        engine = _make_engine()
        await engine.add({42: _normalize(_one_hot(0))})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=1)
        assert isinstance(result.matches[0].key, int)

    @pytest.mark.asyncio
    async def test_scores_are_floats(self):
        engine = _make_engine()
        await engine.add({42: _normalize(_one_hot(0))})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=1)
        assert isinstance(result.matches[0].score, float)
