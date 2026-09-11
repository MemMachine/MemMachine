"""Tests for TurboVecVectorSearchEngine.

turbovec stores TurboQuant-compressed vectors, so scores are approximate: a
self-match lands near the exact value, and orthogonal pairs land near zero
rather than exactly zero. Tests therefore assert ranking and membership (robust
under quantization) and only loosely assert score magnitudes. The one exact
bound is the cosine range, which the engine clamps.
"""

import math
from pathlib import Path

import pytest

pytest.importorskip("turbovec")

from memmachine_server.common.vector_store.vector_search_engine.turbovec_engine import (
    TurboVecVectorSearchEngine,
)

NDIM = 8

QUANT_ABS = 0.05


def _normalize(v: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in v))
    return [x / magnitude for x in v]


def _one_hot(index: int, value: float = 1.0) -> list[float]:
    """A length-NDIM vector with `value` at `index`, zeros elsewhere."""
    vector = [0.0] * NDIM
    vector[index] = value
    return vector


def _entry_names(directory: Path) -> list[str]:
    """Names of everything in `directory`. Sync: the tests calling it are not."""
    return [entry.name for entry in directory.iterdir()]


def _spread(seed: int) -> list[float]:
    """A deterministic unit vector, distinct per `seed`."""
    return _normalize(
        [math.cos(seed * (axis + 1) * 0.7 + axis) for axis in range(NDIM)]
    )


async def _search_one(engine, vector, limit=10, **kwargs):
    """Helper: search a single vector, return the one SearchResult."""
    results = await engine.search([vector], limit=limit, **kwargs)
    return results[0]


def _make_engine(**kwargs):
    return TurboVecVectorSearchEngine(num_dimensions=NDIM, **kwargs)


# -- Construction --


class TestConstruction:
    def test_valid_bit_widths(self):
        for bit_width in (2, 3, 4):
            _make_engine(bit_width=bit_width)

    def test_invalid_bit_width_raises(self):
        with pytest.raises(ValueError, match="bit_width"):
            _make_engine(bit_width=5)

    def test_unaligned_dimensions_are_accepted(self):
        TurboVecVectorSearchEngine(num_dimensions=7)


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
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        await engine.remove([1])
        await engine.add({1: _normalize(_one_hot(1))})
        result = await _search_one(engine, _normalize(_one_hot(1)), limit=1)
        assert result.matches[0].key == 1
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=QUANT_ABS)

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
        assert result.matches[0].key == 1
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=QUANT_ABS)
        assert result.matches[1].key == 3
        assert result.matches[2].key == 2

    @pytest.mark.asyncio
    async def test_orthogonal_scores_near_zero(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=2)
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=QUANT_ABS)
        assert result.matches[1].cosine_similarity == pytest.approx(0.0, abs=0.1)

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
            assert (
                result.matches[i].cosine_similarity
                >= result.matches[i + 1].cosine_similarity
            )

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


# -- Search: allowed_keys filtering --


class TestSearchAllowedKeys:
    @pytest.mark.asyncio
    async def test_allowed_keys_restrict_results(self):
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
            engine, _normalize(_one_hot(0)), limit=4, allowed_keys=[2, 3]
        )
        assert {m.key for m in result.matches} == {2, 3}

    @pytest.mark.asyncio
    async def test_allowed_keys_exclude_best_match(self):
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize(_one_hot(2)),
            }
        )
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=1, allowed_keys=[2, 3]
        )
        assert len(result.matches) == 1
        assert result.matches[0].key in {2, 3}

    @pytest.mark.asyncio
    async def test_empty_allowed_keys_return_nothing(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=1, allowed_keys=[]
        )
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_absent_allowed_keys_are_ignored(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=2, allowed_keys=[1, 12345]
        )
        assert [m.key for m in result.matches] == [1]

    @pytest.mark.asyncio
    async def test_low_ranked_allowed_key_found(self):
        engine = _make_engine()
        vectors = {index: _spread(index) for index in range(2 * NDIM)}
        await engine.add(vectors)

        target = 2 * NDIM - 1
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=1, allowed_keys=[target]
        )
        assert [m.key for m in result.matches] == [target]


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
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=QUANT_ABS)

    @pytest.mark.asyncio
    async def test_save_leaves_no_temp_file(self, tmp_path: Path):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})

        path = tmp_path / "test.idx"
        await engine.save(str(path))

        assert _entry_names(tmp_path) == ["test.idx"]

    @pytest.mark.asyncio
    async def test_checkpoint_carries_adds_and_mass_removals(self, tmp_path: Path):
        engine = _make_engine()
        await engine.add({key: _spread(key) for key in range(1, 2001)})

        path = str(tmp_path / "test.idx")
        await engine.save(path)

        doomed = [key for key in range(1, 2001) if key % 4 != 0]
        await engine.remove(doomed)
        await engine.add({key: _spread(key) for key in range(10_001, 10_033)})
        await engine.save(path)

        engine2 = _make_engine()
        await engine2.load(path)

        expected = set(range(4, 2001, 4)) | set(range(10_001, 10_033))
        result = await _search_one(engine2, _spread(1), limit=len(expected) + 10)
        assert {match.key for match in result.matches} == expected

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


class TestUnalignedDimensions:
    @pytest.mark.asyncio
    async def test_search_at_an_unaligned_width(self):
        engine = TurboVecVectorSearchEngine(num_dimensions=5)
        await engine.add(
            {
                1: [1.0, 0.0, 0.0, 0.0, 0.0],
                2: [0.0, 1.0, 0.0, 0.0, 0.0],
            }
        )

        results = await engine.search([[1.0, 0.0, 0.0, 0.0, 0.0]], limit=2)
        matches = results[0].matches
        assert [match.key for match in matches] == [1, 2]
        assert matches[0].cosine_similarity == pytest.approx(1.0, abs=QUANT_ABS)
        assert matches[1].cosine_similarity == pytest.approx(0.0, abs=QUANT_ABS)

    @pytest.mark.asyncio
    async def test_save_and_load_at_an_unaligned_width(self, tmp_path: Path):
        def make_engine():
            return TurboVecVectorSearchEngine(num_dimensions=5)

        engine = make_engine()
        await engine.add({1: [1.0, 0.0, 0.0, 0.0, 0.0]})
        path = str(tmp_path / "test.idx")
        await engine.save(path)

        loaded = make_engine()
        await loaded.load(path)

        results = await loaded.search([[1.0, 0.0, 0.0, 0.0, 0.0]], limit=1)
        assert [match.key for match in results[0].matches] == [1]

    @pytest.mark.asyncio
    async def test_wrong_width_vector_is_rejected(self):
        engine = TurboVecVectorSearchEngine(num_dimensions=5)
        with pytest.raises(ValueError, match="must have 5 dimensions"):
            await engine.add({1: [1.0, 0.0, 0.0]})


class TestNonUnitVectors:
    """Vectors are normalized on the way in, so scores are cosines.

    Every other test here feeds unit vectors, under which normalization is a
    no-op. These use SHORT vectors rather than long ones on purpose: an
    over-long raw inner product is hidden by the clamp onto [-1, 1], while an
    under-long one lands inside the range and is visible.
    """

    @pytest.mark.asyncio
    async def test_a_short_stored_vector_still_self_matches(self):
        engine = _make_engine()
        await engine.add({1: _one_hot(0, value=0.02)})

        result = await _search_one(engine, _one_hot(0), limit=1)
        # Without normalization the raw inner product is 0.02.
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=QUANT_ABS)

    @pytest.mark.asyncio
    async def test_a_short_query_vector_still_self_matches(self):
        engine = _make_engine()
        await engine.add({1: _one_hot(0)})

        result = await _search_one(engine, _one_hot(0, value=0.02), limit=1)
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=QUANT_ABS)

    @pytest.mark.asyncio
    async def test_magnitude_does_not_outrank_direction(self):
        """A raw inner product would rank the longer vector first."""
        aligned_but_short = _one_hot(0, value=0.02)
        misaligned_but_unit = _normalize([1.0, 1.0] + [0.0] * (NDIM - 2))
        engine = _make_engine()
        await engine.add({1: aligned_but_short, 2: misaligned_but_unit})

        # Raw: 0.02 for key 1 against ~0.707 for key 2, so key 2 would win.
        # By cosine: 1.0 against ~0.707, so key 1 wins.
        result = await _search_one(engine, _one_hot(0), limit=2)
        assert [match.key for match in result.matches] == [1, 2]

    @pytest.mark.asyncio
    async def test_score_is_scale_invariant(self):
        """The same direction at any length must score the same."""
        engine = _make_engine()
        await engine.add({1: _one_hot(1), 2: _one_hot(1, value=0.02)})

        result = await _search_one(engine, _one_hot(1), limit=2)
        scores = {match.key: match.cosine_similarity for match in result.matches}
        assert scores[1] == pytest.approx(scores[2], abs=QUANT_ABS)


class TestScoreRange:
    @pytest.mark.asyncio
    async def test_cosine_scores_stay_within_range(self):
        engine = _make_engine()
        vectors = {key: _spread(key) for key in range(1, 51)}
        await engine.add(vectors)

        for vector in vectors.values():
            result = await _search_one(engine, vector, limit=5)
            for match in result.matches:
                assert -1.0 <= match.cosine_similarity <= 1.0


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
        assert isinstance(result.matches[0].cosine_similarity, float)
