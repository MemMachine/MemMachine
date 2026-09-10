"""hnswlib HNSW implementation of VectorSearchEngine."""

import asyncio
import contextlib
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import ClassVar, override

import hnswlib  # ty: ignore[unresolved-import]  # C extension, no py.typed
import numpy as np
import numpy.typing as npt

from memmachine_server.common.rw_locks import AsyncRWLock

from .index_persistence import atomic_index_write, clear_stale_index_temp
from .utils import top_k_matches, unit_normalize
from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class HnswlibVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by hnswlib HNSW."""

    _SPACE: ClassVar[str] = "cosine"

    _DEFAULT_M: ClassVar[int] = 16
    _DEFAULT_EF_CONSTRUCTION: ClassVar[int] = 128
    _DEFAULT_EF_SEARCH: ClassVar[int] = 128

    _DEFAULT_INITIAL_CAPACITY: ClassVar[int] = 1024
    _RESIZE_FACTOR: ClassVar[float] = 2.0

    def __init__(
        self,
        *,
        num_dimensions: int,
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
        initial_capacity: int = _DEFAULT_INITIAL_CAPACITY,
        allow_replace_deleted: bool = True,
    ) -> None:
        """
        Initialize.

        When allow_replace_deleted=True (default),
        tombstoned slots from remove() are reclaimed by future inserts of brand-new labels,
        bounding index growth under churn, but mixed operations workloads may be many times slower.
        The add() path partitions inputs into known (in-place upsert)
        and new (slot reuse via replace_deleted=True) to sidestep an upstream label_lookup_ corruption bug;
        see https://github.com/nmslib/hnswlib/blob/v0.8.0/hnswlib/hnswalg.h#L981-L987.
        Set to False if you'd rather pay memory growth than the per-replacement updatePoint cost.
        """
        self._num_dimensions = num_dimensions

        self._ef_search = ef_search
        self._allow_replace_deleted = allow_replace_deleted

        self._index = hnswlib.Index(
            space=HnswlibVectorSearchEngine._SPACE, dim=num_dimensions
        )
        self._index.init_index(
            max_elements=initial_capacity,
            ef_construction=ef_construction,
            M=m,
            allow_replace_deleted=allow_replace_deleted,
        )
        self._index.set_ef(ef_search)

        # Mirrors hnswlib's label_lookup_ keys (live + tombstoned).
        # Used to partition adds into known (in-place upsert) vs new (slot reuse).
        # May contain stale entries after a brand-new add evicts a previously tombstoned label;
        # staleness causes a re-add of the evicted label to take the in-place upsert path,
        # resulting in appending a new slot instead of reusing a tombstone,
        # which does not affect correctness (but may be suboptimal).
        self._known_labels: set[int] = set()

        self._lock = AsyncRWLock()

    @staticmethod
    def _distance_to_cosine_similarity(distance: float) -> float:
        """Convert an hnswlib cosine distance to a cosine similarity."""
        return 1.0 - distance

    def _ensure_capacity(self, needed: int) -> None:
        """Grow the index if there isn't room for `needed` more elements."""
        current_max = self._index.get_max_elements()
        current_count = self._index.element_count

        if current_count + needed <= current_max:
            return

        new_max = max(
            current_max + needed,
            int(current_max * HnswlibVectorSearchEngine._RESIZE_FACTOR),
        )
        self._index.resize_index(new_max)

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_add, vectors)

    def _sync_add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not self._allow_replace_deleted:
            keys_array = np.array(list(vectors.keys()), dtype=np.int64)
            vectors_array = np.array(list(vectors.values()), dtype=np.float32)
            self._ensure_capacity(len(keys_array))
            self._index.add_items(vectors_array, keys_array)
            self._known_labels.update(vectors.keys())
            return

        # Smart-partition path: route known labels through in-place upsert
        # and brand-new labels through slot reuse.
        # Mixing replace_deleted=True with any label currently in hnswlib's label_lookup_ corrupts the lookup map.
        known_pairs = [(k, v) for k, v in vectors.items() if k in self._known_labels]
        new_pairs = [(k, v) for k, v in vectors.items() if k not in self._known_labels]

        if known_pairs:
            for key, _ in known_pairs:
                # Tombstoned labels must be unmarked first.
                with contextlib.suppress(RuntimeError):
                    self._index.unmark_deleted(int(key))

            # Reserve worst-case capacity.
            self._ensure_capacity(len(known_pairs))
            keys_array = np.fromiter(
                (k for k, _ in known_pairs), dtype=np.int64, count=len(known_pairs)
            )
            vectors_array = np.array([v for _, v in known_pairs], dtype=np.float32)
            self._index.add_items(vectors_array, keys_array, replace_deleted=False)

        if new_pairs:
            self._ensure_capacity(len(new_pairs))
            keys_array = np.fromiter(
                (k for k, _ in new_pairs), dtype=np.int64, count=len(new_pairs)
            )
            vectors_array = np.array([v for _, v in new_pairs], dtype=np.float32)
            self._index.add_items(vectors_array, keys_array, replace_deleted=True)
            self._known_labels.update(k for k, _ in new_pairs)

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        *,
        limit: int,
        allowlist: Collection[int] | None = None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if (
            self._index.element_count == 0
            or not vectors
            or (allowlist is not None and not allowlist)
        ):
            return [SearchResult(matches=[]) for _ in vectors]

        async with self._lock.read_lock():
            return await asyncio.to_thread(self._sync_search, vectors, limit, allowlist)

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
        allowlist: Collection[int] | None,
    ) -> list[SearchResult]:
        # hnswlib normalizes what it stores under the cosine space, so a
        # gathered vector is already unit; only the query is left to scale.
        query = unit_normalize(np.array(vectors, dtype=np.float32))

        if allowlist is not None:
            return self._sync_search_allowlist(query, limit, allowlist)

        effective_limit = min(limit, self._index.element_count)
        if effective_limit <= 0:
            return [SearchResult(matches=[]) for _ in vectors]

        return self._binary_search_query(query, effective_limit)

    def _sync_search_allowlist(
        self,
        unit_queries: npt.NDArray[np.float32],
        limit: int,
        allowlist: Collection[int],
    ) -> list[SearchResult]:
        """Exact: gather the allowed vectors and score them directly.

        Both sides are unit vectors by now, so the inner product
        `top_k_matches` ranks by is the cosine similarity.
        """
        present_keys, matrix = self._sync_gather_vectors(allowlist)
        if not present_keys:
            return [SearchResult(matches=[]) for _ in unit_queries]

        return [
            SearchResult(matches=top_k_matches(query, present_keys, matrix, limit))
            for query in unit_queries
        ]

    def _binary_search_query(
        self,
        query: np.ndarray,
        k: int,
    ) -> list[SearchResult]:
        # Fast path: hnswlib filled k for the entire batch in one call.
        result = self._try_knn_query(query, k)
        if result is not None:
            return self._build_search_results_from_knn(result)

        # Slow path: each query vector gets its own search and its own max fillable k.
        # The bottom-level graph is not guaranteed to be connected,
        # so a single probe's candidate set is not a valid stand-in
        # for other query vectors' candidate sets.
        results: list[SearchResult] = []
        for query_vector in query:
            single = query_vector[np.newaxis, :]
            single_result = self._try_knn_query(single, k)

            if single_result is None:
                # k is already known to fail for this query, so search strictly below it.
                single_result = self._largest_fillable_knn_query(single, k - 1)
                if single_result is None:
                    results.append(SearchResult(matches=[]))
                    continue

            results.extend(self._build_search_results_from_knn(single_result))

        return results

    def _try_knn_query(
        self,
        query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Run knn_query, returning None if hnswlib raises."""
        try:
            return self._index.knn_query(query, k=k, num_threads=1)
        except RuntimeError:
            return None

    def _build_search_results_from_knn(
        self,
        result: tuple[np.ndarray, np.ndarray],
    ) -> list[SearchResult]:
        all_labels, all_distances = result
        results: list[SearchResult] = []
        for labels, distances in zip(all_labels, all_distances, strict=True):
            matches = [
                SearchMatch(
                    key=int(label),
                    cosine_similarity=HnswlibVectorSearchEngine._distance_to_cosine_similarity(
                        float(dist)
                    ),
                )
                for label, dist in zip(labels, distances, strict=True)
                if int(label) >= 0
            ]
            results.append(SearchResult(matches=matches))
        return results

    def _largest_fillable_knn_query(
        self,
        query: np.ndarray,
        limit: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Binary search for the largest k in [1, limit] that hnswlib can fill, for a single query.

        Returns the knn_query result at that k, or None if even k=1 fails.
        """
        low, high = 1, limit
        best: tuple[np.ndarray, np.ndarray] | None = None
        while low <= high:
            mid = (low + high) // 2
            result = self._try_knn_query(query, mid)
            if result is not None:
                best = result
                low = mid + 1
            else:
                high = mid - 1

        return best

    def _sync_gather_vectors(
        self, keys: Iterable[int]
    ) -> tuple[list[int], npt.NDArray[np.float32]]:
        """Gather stored vectors by key as a float32 matrix; missing keys drop."""
        keys = list(set(keys))
        empty = np.empty((0, self._num_dimensions), dtype=np.float32)
        if not keys:
            return [], empty

        try:
            return keys, np.asarray(
                self._index.get_items(keys, return_type="numpy"), dtype=np.float32
            )
        except RuntimeError:
            # Fallback: a missing key was requested. Fetch one by one.
            present_keys: list[int] = []
            rows: list[np.ndarray] = []
            for key in keys:
                with contextlib.suppress(RuntimeError):
                    rows.append(
                        np.asarray(
                            self._index.get_items([key], return_type="numpy"),
                            dtype=np.float32,
                        )[0]
                    )
                    present_keys.append(key)
            if not present_keys:
                return [], empty
            return present_keys, np.vstack(rows)

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_remove, keys)

    def _sync_remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            with contextlib.suppress(RuntimeError):
                self._index.mark_deleted(key)

    @override
    async def save(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_save, path)

    def _sync_save(self, path: str) -> None:
        with atomic_index_write(path) as temp_path:
            self._index.save_index(temp_path)
        # Reconcile _known_labels with hnswlib's label_lookup_.
        if self._allow_replace_deleted:
            self._known_labels = {int(k) for k in self._index.get_ids_list()}

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_load, path)

    def _sync_load(self, path: str) -> None:
        clear_stale_index_temp(path)
        self._index = hnswlib.Index(
            space=HnswlibVectorSearchEngine._SPACE, dim=self._num_dimensions
        )
        self._index.load_index(path, allow_replace_deleted=self._allow_replace_deleted)
        self._index.set_ef(self._ef_search)
        # Reconcile _known_labels with hnswlib's label_lookup_.
        if self._allow_replace_deleted:
            self._known_labels = {int(k) for k in self._index.get_ids_list()}
