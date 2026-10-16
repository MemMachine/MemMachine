"""USearch HNSW implementation of VectorSearchEngine."""

import asyncio
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import ClassVar, override

import numpy as np
from usearch.index import Index, MetricKind

from memmachine_server.common.rw_locks import AsyncRWLock

from .index_persistence import atomic_index_write, clear_stale_index_temp
from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class USearchVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by USearch HNSW."""

    _METRIC_KIND: ClassVar[MetricKind] = MetricKind.Cos

    _DEFAULT_M: ClassVar[int] = 16
    _DEFAULT_EF_CONSTRUCTION: ClassVar[int] = 128
    _DEFAULT_EF_SEARCH: ClassVar[int] = 128

    _OVERFETCH_BASE: ClassVar[int] = 4

    def __init__(
        self,
        *,
        num_dimensions: int,
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
    ) -> None:
        """Initialize."""
        self._index = Index(
            ndim=num_dimensions,
            metric=self._METRIC_KIND,
            dtype="f32",
            connectivity=m,
            expansion_add=ef_construction,
            expansion_search=ef_search,
        )

        self._lock = AsyncRWLock()

    @staticmethod
    def _distance_to_cosine_similarity(distance: float) -> float:
        """Convert a USearch cosine distance to a cosine similarity."""
        return 1.0 - distance

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_add, vectors)

    def _sync_add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        keys_array = np.array(list(vectors.keys()), dtype=np.int64)
        vectors_array = np.array(list(vectors.values()), dtype=np.float32)
        self._index.add(keys_array, vectors_array)

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        *,
        limit: int,
        allowed_keys: Container[int] | None = None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if self._index.size == 0 or not vectors:
            return [SearchResult(matches=[]) for _ in vectors]

        async with self._lock.read_lock():
            return await asyncio.to_thread(
                self._sync_search, vectors, limit, allowed_keys
            )

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
        allowed_keys: Container[int] | None,
    ) -> list[SearchResult]:
        query = np.array(vectors, dtype=np.float32)
        num_queries = query.shape[0]

        overfetch_factor = (
            1 if allowed_keys is None else USearchVectorSearchEngine._OVERFETCH_BASE
        )

        final_results: list[SearchResult | None] = [None] * num_queries
        pending_indices = list(range(num_queries))

        while pending_indices:
            pending_query = query[pending_indices]
            fetch_limit = min(limit * overfetch_factor, self._index.size)

            results = self._index.search(pending_query, fetch_limit)
            all_keys = np.atleast_2d(results.keys)
            all_distances = np.atleast_2d(results.distances)

            still_pending: list[int] = []
            for batch_idx, original_idx in enumerate(pending_indices):
                matches: list[SearchMatch] = []
                for key, dist in zip(
                    all_keys[batch_idx], all_distances[batch_idx], strict=True
                ):
                    int_key = int(key)
                    if int_key < 0:
                        continue

                    if allowed_keys is not None and int_key not in allowed_keys:
                        continue

                    matches.append(
                        SearchMatch(
                            key=int_key,
                            cosine_similarity=self._distance_to_cosine_similarity(
                                float(dist)
                            ),
                        )
                    )
                    if len(matches) >= limit:
                        break

                if len(matches) >= limit or fetch_limit >= self._index.size:
                    final_results[original_idx] = SearchResult(matches=matches)
                else:
                    still_pending.append(original_idx)

            pending_indices = still_pending
            overfetch_factor *= USearchVectorSearchEngine._OVERFETCH_BASE

        return [r if r is not None else SearchResult(matches=[]) for r in final_results]

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_remove, keys)

    def _sync_remove(self, keys: Iterable[int]) -> None:
        index = self._index
        for key in keys:
            index.remove(int(key))

    @override
    async def save(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_save, path)

    def _sync_save(self, path: str) -> None:
        with atomic_index_write(path) as temp_path:
            self._index.save(temp_path)

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_load, path)

    def _sync_load(self, path: str) -> None:
        clear_stale_index_temp(path)
        self._index.load(path)
