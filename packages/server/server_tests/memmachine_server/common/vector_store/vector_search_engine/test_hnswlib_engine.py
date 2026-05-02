"""Tests for HnswlibVectorSearchEngine."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

# hnswlib ships sdist only and hardcodes -march=native, so a wheel built on one
# GitHub-hosted runner CPU can SIGILL on another (uv caches built wheels keyed
# by OS + lockfile, not CPU microarch). Skip on CI before importing hnswlib so
# the SIGILL doesn't fire during pytest collection.
if os.getenv("CI") == "true":
    pytest.skip(
        "hnswlib build cache mismatches CPU across GitHub-hosted runners",
        allow_module_level=True,
    )

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store.vector_search_engine.hnswlib_engine import (
    HnswlibVectorSearchEngine,
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
            HnswlibVectorSearchEngine(num_dimensions=NDIM, similarity_metric=metric)

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            HnswlibVectorSearchEngine(
                num_dimensions=NDIM, similarity_metric=SimilarityMetric.MANHATTAN
            )


# -- Add --


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_single(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1.0, 0.0, 0.0]})
        result = await _search_one(engine, [1.0, 0.0, 0.0], limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_add_batch(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({10: [1, 0, 0], 20: [0, 1, 0], 30: [0, 0, 1]})
        result = await _search_one(engine, [1, 0, 0], limit=3)
        assert {m.key for m in result.matches} == {10, 20, 30}

    @pytest.mark.asyncio
    async def test_remove_then_add_replaces_key(self):
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_add_beyond_initial_capacity(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM,
            similarity_metric=SimilarityMetric.COSINE,
            initial_capacity=2,
        )
        await engine.add(
            {
                1: [1, 0, 0],
                2: [0, 1, 0],
                3: [0, 0, 1],
                4: [1, 1, 0],
                5: [0, 1, 1],
            }
        )
        result = await _search_one(engine, [1, 0, 0], limit=5)
        assert len(result.matches) == 5


# -- Remove --


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_existing(self):
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0]})
        await engine.remove([99, 100])
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_remove_all(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]})
        await engine.remove([1, 2, 3])
        result = await _search_one(engine, [1, 0, 0], limit=3)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_remove_empty_iterable(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0]})
        await engine.remove([])
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert result.matches[0].key == 1

    @pytest.mark.asyncio
    async def test_remove_excludes_from_search(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: _normalize([1, 0, 0]), 2: _normalize([0, 1, 0])})
        await engine.remove([1])
        result = await _search_one(engine, _normalize([1, 0, 0]), limit=2)
        assert 1 not in {m.key for m in result.matches}


# -- Search: Cosine --


class TestSearchCosine:
    @pytest.mark.asyncio
    async def test_basic_knn(self):
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=10)
        assert len(result.matches) == 1

    @pytest.mark.asyncio
    async def test_search_empty_index(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        result = await _search_one(engine, [1, 0, 0], limit=5)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_batched_search(self):
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.EUCLIDEAN
        )
        await engine.add({1: [0, 0, 0], 2: [3, 4, 0]})
        result = await _search_one(engine, [0, 0, 0], limit=2)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(0.0, abs=0.01)
        assert result.matches[1].score == pytest.approx(5.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_scores_ordered_best_first(self):
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
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
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({1: _normalize([1, 0, 0]), 2: _normalize([0, 1, 0])})

        path = str(tmp_path / "test.hnswlib")
        await engine.save(path)

        engine2 = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine2.load(path)

        result = await _search_one(engine2, _normalize([1, 0, 0]), limit=2)
        assert {m.key for m in result.matches} == {1, 2}
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_save_and_load_with_deletions(self, tmp_path: Path):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add(
            {
                1: _normalize([1, 0, 0]),
                2: _normalize([0, 1, 0]),
                3: _normalize([0, 0, 1]),
            }
        )
        await engine.remove([2])

        path = str(tmp_path / "test.hnswlib")
        await engine.save(path)

        engine2 = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine2.load(path)

        result = await _search_one(engine2, _normalize([1, 0, 0]), limit=3)
        assert {m.key for m in result.matches} == {1, 3}


# -- allow_replace_deleted=True (slot reclamation) --


def _rand_unit(rng, dim: int) -> list[float]:
    v = rng.standard_normal(dim).astype("float32")
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-10)).tolist()


def _no_corruption(engine: HnswlibVectorSearchEngine) -> None:
    """Verify label_lookup_ is internally consistent.

    Corruption indicators:
      - knn_query returns a label that get_items can't fetch (zombie).
      - A label in get_ids_list() raises "Label not found" from get_items
        (would indicate a label_lookup_ entry pointing nowhere or wrong).

    NOT corruption: hnswlib's "Cannot return ... Probably ef or M is too
    small" when k exceeds reachable live count — that's a known knn_query
    capacity limit, not a label-table issue.
    """
    import contextlib

    idx = engine._index
    label_set = {int(x) for x in idx.get_ids_list()}
    if not label_set or idx.element_count == 0:
        return

    # Probe with a small k that even a heavily-tombstoned index can fill.
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((3, engine._num_dimensions)).astype("float32")
    k = min(3, idx.element_count)
    try:
        labels, _ = idx.knn_query(probe, k=k)
        for row in labels:
            for lab in row:
                lab_i = int(lab)
                try:
                    idx.get_items([lab_i])
                except RuntimeError as exc:
                    raise AssertionError(
                        f"zombie label {lab_i}: returned by knn but get_items raised: {exc}"
                    ) from exc
    except RuntimeError as exc:
        # "Cannot return ... Probably ef or M is too small" is a capacity
        # problem, not corruption. Anything else (e.g., "Label not found"
        # surfacing from inside knn) IS corruption.
        if "Cannot return" not in str(exc):
            raise AssertionError(f"unexpected knn_query failure: {exc}") from exc

    # For live (non-tombstoned) labels, get_items must succeed. For
    # tombstoned labels, get_items raises with a "deleted"/"not found" msg
    # and that's expected. The corruption signature would be a label that
    # appears in get_ids_list() but get_items raises for a reason OTHER
    # than tombstoning — i.e., the label_lookup_ entry doesn't actually
    # resolve to a real slot. Since hnswlib uses one shared message, we
    # can't distinguish here; this loop is informational only.
    for lab in list(label_set)[:50]:
        with contextlib.suppress(RuntimeError):
            idx.get_items([int(lab)])


class TestReplaceDeletedSlotReclamation:
    @pytest.mark.asyncio
    async def test_basic_parity_with_no_replace(self):
        """Same workload, both modes produce searchable correct results."""
        rng = np.random.default_rng(42)
        for allow in (False, True):
            engine = HnswlibVectorSearchEngine(
                num_dimensions=8,
                similarity_metric=SimilarityMetric.COSINE,
                allow_replace_deleted=allow,
            )
            vecs = {i: _rand_unit(rng, 8) for i in range(20)}
            await engine.add(vecs)
            result = await _search_one(engine, vecs[0], limit=5)
            assert result.matches[0].key == 0
            _no_corruption(engine)

    @pytest.mark.asyncio
    async def test_remove_then_readd_same_label_in_place(self):
        """Re-adding a deleted label must preserve the label and update vector."""
        rng = np.random.default_rng(1)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=8,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
        )
        v_old = _rand_unit(rng, 8)
        v_new = _rand_unit(rng, 8)
        await engine.add({42: v_old})
        await engine.remove([42])
        await engine.add({42: v_new})

        result = await _search_one(engine, v_new, limit=1)
        assert result.matches[0].key == 42
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        _no_corruption(engine)

    @pytest.mark.asyncio
    async def test_batch_remove_then_readd_same_labels_no_corruption(self):
        """Case C scenario: batch remove + batch re-add of same labels.

        Naive (T,T) corrupts label_lookup_ here. The smart-partition
        wrapper must keep all labels intact via the in-place upsert path.
        """
        rng = np.random.default_rng(2)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=16,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
        )
        vecs = {i: _rand_unit(rng, 16) for i in range(50)}
        await engine.add(vecs)

        await engine.remove(list(range(50)))
        new_vecs = {i: _rand_unit(rng, 16) for i in range(50)}
        await engine.add(new_vecs)

        # Every label must be lookup-able with its NEW vector.
        for i in range(50):
            result = await _search_one(engine, new_vecs[i], limit=1)
            assert result.matches[0].key == i, f"label {i} missing or wrong"
            assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        _no_corruption(engine)

    @pytest.mark.asyncio
    async def test_accumulated_tombstones_then_single_readd(self):
        """Case D scenario: many tombstones present, then re-add one label.

        Naive (T,T) creates an orphan that arms a future corruption.
        The smart-partition wrapper takes the in-place path for the re-add.
        """
        rng = np.random.default_rng(3)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=16,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
        )
        vecs = {i: _rand_unit(rng, 16) for i in range(20)}
        await engine.add(vecs)

        # Mark 10 deleted, leaving labels 0-9 tombstoned, 10-19 live.
        await engine.remove(list(range(10)))

        # Re-add label 5 with new vector.
        v_new = _rand_unit(rng, 16)
        await engine.add({5: v_new})

        result = await _search_one(engine, v_new, limit=1)
        assert result.matches[0].key == 5
        # Other tombstoned labels should remain non-returnable in queries.
        # And label 5's position in the lookup should be live & queryable.
        _no_corruption(engine)

    @pytest.mark.asyncio
    async def test_slot_reclamation_actually_happens(self):
        """Tombstoned slots get reused by brand-new labels (the point of T,T)."""
        rng = np.random.default_rng(4)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=8,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
            initial_capacity=200,
        )

        await engine.add({i: _rand_unit(rng, 8) for i in range(100)})
        count_before = engine._index.element_count

        await engine.remove(list(range(50)))
        await engine.add({100 + i: _rand_unit(rng, 8) for i in range(50)})

        # Slots reused: count should not have grown.
        count_after = engine._index.element_count
        assert count_after == count_before, (
            f"expected slot reuse to keep element_count flat, got {count_before} → {count_after}"
        )
        _no_corruption(engine)

        # All 100 currently-intended labels must be queryable.
        for lab in [50, 75, 99, 100, 125, 149]:
            result = await _search_one(engine, _rand_unit(rng, 8), limit=100)
            keys = {m.key for m in result.matches}
            assert lab in keys, f"label {lab} missing from search results"

    @pytest.mark.asyncio
    async def test_no_reclamation_when_disabled(self):
        """With allow_replace_deleted=False, deleted slots accumulate."""
        rng = np.random.default_rng(5)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=8,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=False,
            initial_capacity=200,
        )

        await engine.add({i: _rand_unit(rng, 8) for i in range(100)})
        count_before = engine._index.element_count

        await engine.remove(list(range(50)))
        await engine.add({100 + i: _rand_unit(rng, 8) for i in range(50)})

        count_after = engine._index.element_count
        assert count_after == count_before + 50, (
            f"expected new appends without reuse, got {count_before} → {count_after}"
        )

    @pytest.mark.asyncio
    async def test_heavy_churn_no_corruption(self):
        """Many delete/insert cycles must not produce zombies or missing labels."""
        rng = np.random.default_rng(6)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=16,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
            initial_capacity=500,
        )

        next_label = 0
        live: dict[int, list[float]] = {}
        for _ in range(200):
            new_vecs = {next_label + i: _rand_unit(rng, 16) for i in range(10)}
            next_label += 10
            await engine.add(new_vecs)
            live.update(new_vecs)

            if len(live) > 50:
                victims = rng.choice(list(live.keys()), size=10, replace=False)
                await engine.remove([int(x) for x in victims])
                for v in victims:
                    live.pop(int(v), None)

            if next_label % 50 == 0:
                _no_corruption(engine)

        _no_corruption(engine)

        # Every live label must be findable.
        sample = list(live.keys())[:20]
        for lab in sample:
            v = live[lab]
            result = await _search_one(engine, v, limit=1)
            assert result.matches[0].key == lab, f"label {lab} not findable"

    @pytest.mark.asyncio
    async def test_save_load_preserves_replace_state(self, tmp_path: Path):
        """Save/load round trip preserves correct (T,T) behavior.

        Loading an index reconstructs label_lookup_ from per-slot stored
        labels and (when allow_replace_deleted=True) deleted_elements from
        tombstone bits. _known_labels must be rebuilt from get_ids_list().
        """
        rng = np.random.default_rng(7)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=16,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
            initial_capacity=100,
        )
        vecs = {i: _rand_unit(rng, 16) for i in range(40)}
        await engine.add(vecs)
        await engine.remove([5, 15, 25])

        path = str(tmp_path / "engine.bin")
        await engine.save(path)

        engine2 = HnswlibVectorSearchEngine(
            num_dimensions=16,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
        )
        await engine2.load(path)

        # _known_labels rebuilt from get_ids_list (includes tombstoned).
        assert 5 in engine2._known_labels
        assert 15 in engine2._known_labels
        assert 0 in engine2._known_labels

        # Re-add a deleted label in the loaded engine — known path → unmark + in-place.
        v_new = _rand_unit(rng, 16)
        await engine2.add({5: v_new})
        result = await _search_one(engine2, v_new, limit=1)
        assert result.matches[0].key == 5
        _no_corruption(engine2)

        # Insert a brand-new label — should reuse one of the remaining tombstones.
        count_before = engine2._index.element_count
        await engine2.add({200: _rand_unit(rng, 16)})
        count_after = engine2._index.element_count
        assert count_after == count_before, "expected slot reuse after load"
        _no_corruption(engine2)

    @pytest.mark.asyncio
    async def test_row_id_reuse_pattern(self):
        """SQLite autoincrement may reassign a row_id to a new uuid.

        Engine sees: delete(X), then later add(X) with a different vector.
        With (T,T) and the smart-partition wrapper, X is still in
        _known_labels after the delete, so the add takes the in-place path.
        """
        rng = np.random.default_rng(8)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=8,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=True,
        )
        v_orig = _rand_unit(rng, 8)
        v_reused = _rand_unit(rng, 8)
        await engine.add({100: v_orig})
        await engine.remove([100])
        await engine.add({100: v_reused})  # same row_id, different content

        result = await _search_one(engine, v_reused, limit=1)
        assert result.matches[0].key == 100
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        _no_corruption(engine)


# -- allow_replace_deleted=False (id reuse via in-place upsert path) --


class TestAllowReplaceDeletedFalse:
    """When allow_replace_deleted=False, hnswlib routes id reuse through the
    inner addPoint path (hnswalg.h:1158-1175): existing labels (live OR
    tombstoned) get an in-place updatePoint; the corruption-prone smart-swap
    path at lines 980-990 is never taken because replace_deleted=False on
    add_items. The engine MUST stay corruption-free for:
      - repeated add() on the same id without remove (in-place update)
      - add → remove → add on the same id (unmark + in-place update)
      - heavy churn mixing the above
    """

    @pytest.mark.asyncio
    async def test_repeat_add_same_label_no_remove(self):
        """50 adds of the same label → 1 slot; final vector is the last one."""
        rng = np.random.default_rng(101)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=8,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=False,
        )
        last_v: list[float] | None = None
        for _ in range(50):
            last_v = _rand_unit(rng, 8)
            await engine.add({99: last_v})

        assert engine._index.element_count == 1
        assert last_v is not None
        result = await _search_one(engine, last_v, limit=1)
        assert result.matches[0].key == 99
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        _no_corruption(engine)

    @pytest.mark.asyncio
    async def test_remove_then_readd_same_label_reuses_slot(self):
        """add → remove → add of same label reuses the slot via in-place upsert.

        Note: this is DIFFERENT from brand-new label slot reclamation, which
        requires allow_replace_deleted=True. Here the slot is "reused" only
        because addPoint finds the label already in label_lookup_, unmarks it,
        and updatePoints in place — see hnswalg.h:1169-1174.
        """
        rng = np.random.default_rng(102)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=8,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=False,
        )
        v_old = _rand_unit(rng, 8)
        v_new = _rand_unit(rng, 8)
        await engine.add({7: v_old})
        count_before = engine._index.element_count
        await engine.remove([7])
        await engine.add({7: v_new})
        count_after = engine._index.element_count
        assert count_after == count_before, (
            f"slot not reused for same-label re-add: {count_before} → {count_after}"
        )
        result = await _search_one(engine, v_new, limit=1)
        assert result.matches[0].key == 7
        assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        _no_corruption(engine)

    @pytest.mark.asyncio
    async def test_batch_add_with_existing_and_new_keys(self):
        """A single add() batch mixing existing (in-place) and brand-new keys
        must place each correctly without corruption."""
        rng = np.random.default_rng(103)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=8,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=False,
        )
        first = {i: _rand_unit(rng, 8) for i in range(20)}
        await engine.add(first)
        count_before = engine._index.element_count

        # Mix: keys 0-9 existing (update in place), keys 20-29 brand new.
        updated = {i: _rand_unit(rng, 8) for i in range(10)}
        added = {20 + i: _rand_unit(rng, 8) for i in range(10)}
        await engine.add({**updated, **added})
        count_after = engine._index.element_count
        # Existing 20 + 10 new = 30 slots; updates take no new slot.
        assert count_after == count_before + 10, (
            f"expected only 10 new slots, got {count_before} → {count_after}"
        )

        # Updated keys carry the new vectors.
        for k, v in updated.items():
            result = await _search_one(engine, v, limit=1)
            assert result.matches[0].key == k
            assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        # Brand-new keys queryable.
        for k, v in added.items():
            result = await _search_one(engine, v, limit=1)
            assert result.matches[0].key == k
            assert result.matches[0].score == pytest.approx(1.0, abs=0.01)
        # Untouched keys 10-19 still queryable.
        for k in range(10, 20):
            result = await _search_one(engine, first[k], limit=1)
            assert result.matches[0].key == k
        _no_corruption(engine)

    @pytest.mark.asyncio
    async def test_heavy_churn_no_corruption(self):
        """add/remove/re-add cycles, mixing id-reuse and new ids; no zombies."""
        rng = np.random.default_rng(104)
        engine = HnswlibVectorSearchEngine(
            num_dimensions=16,
            similarity_metric=SimilarityMetric.COSINE,
            allow_replace_deleted=False,
            initial_capacity=256,
        )
        next_label = 0
        live: dict[int, list[float]] = {}
        for cycle in range(50):
            new = {next_label + i: _rand_unit(rng, 16) for i in range(10)}
            next_label += 10
            await engine.add(new)
            live.update(new)

            if len(live) >= 5:
                victims = rng.choice(list(live.keys()), size=5, replace=False)
                await engine.remove([int(x) for x in victims])
                for v in victims:
                    live.pop(int(v), None)

            # Re-add an existing live id (id reuse without delete).
            if live:
                k = int(next(iter(live.keys())))
                v = _rand_unit(rng, 16)
                live[k] = v
                await engine.add({k: v})

            if cycle % 10 == 0:
                _no_corruption(engine)

        _no_corruption(engine)

        sample = list(live.keys())[:30]
        for k in sample:
            v = live[k]
            result = await _search_one(engine, v, limit=1)
            assert result.matches[0].key == k, f"label {k} not findable"
            assert result.matches[0].score == pytest.approx(1.0, abs=0.01)


# -- SearchResult types --


class TestSearchResultTypes:
    @pytest.mark.asyncio
    async def test_keys_are_ints(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({42: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert isinstance(result.matches[0].key, int)

    @pytest.mark.asyncio
    async def test_scores_are_floats(self):
        engine = HnswlibVectorSearchEngine(
            num_dimensions=NDIM, similarity_metric=SimilarityMetric.COSINE
        )
        await engine.add({42: [1, 0, 0]})
        result = await _search_one(engine, [1, 0, 0], limit=1)
        assert isinstance(result.matches[0].score, float)
