"""USearch HNSW implementation of VectorSearchEngine."""

import asyncio
import math
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import ClassVar, override

import numpy as np
from usearch.index import Index, MetricKind

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.rw_locks import AsyncRWLock

from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class USearchVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by USearch HNSW."""

    _METRIC_MAP: ClassVar[dict[SimilarityMetric, MetricKind]] = {
        SimilarityMetric.COSINE: MetricKind.Cos,
        SimilarityMetric.EUCLIDEAN: MetricKind.L2sq,
        SimilarityMetric.DOT: MetricKind.IP,
    }

    _DEFAULT_M: ClassVar[int] = 16
    _DEFAULT_EF_CONSTRUCTION: ClassVar[int] = 128
    _DEFAULT_EF_SEARCH: ClassVar[int] = 128

    _OVERFETCH_BASE: ClassVar[int] = 4

    def __init__(
        self,
        *,
        num_dimensions: int,
        similarity_metric: SimilarityMetric,
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
    ) -> None:
        """Initialize."""
        usearch_metric = self._METRIC_MAP.get(similarity_metric)
        if usearch_metric is None:
            supported = ", ".join(
                similarity_metric.value for similarity_metric in self._METRIC_MAP
            )
            raise ValueError(
                f"USearch does not support {similarity_metric.value!r}. Supported: {supported}"
            )

        self._index = Index(
            ndim=num_dimensions,
            metric=usearch_metric,
            dtype="f32",
            connectivity=m,
            expansion_add=ef_construction,
            expansion_search=ef_search,
        )
        self._similarity_metric = similarity_metric

        self._lock = AsyncRWLock()

    def _distance_to_score(self, distance: float) -> float:
        """Convert a USearch distance to a pure metric score."""
        match self._similarity_metric:
            case SimilarityMetric.COSINE | SimilarityMetric.DOT:
                return 1.0 - distance
            case SimilarityMetric.EUCLIDEAN:
                return math.sqrt(max(0.0, distance))
            case _:
                raise NotImplementedError(self._similarity_metric)

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
                            score=self._distance_to_score(float(dist)),
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
    async def get_vectors(self, keys: Iterable[int]) -> dict[int, list[float]]:
        keys = list(keys)
        if not keys:
            return {}

        keys_array = np.array(keys, dtype=np.int64)
        async with self._lock.read_lock():
            vectors = await asyncio.to_thread(self._index.get, keys_array)

        result: dict[int, list[float]] = {}
        for i, key in enumerate(keys):
            vector = vectors[i] if vectors is not None else None
            if vector is not None:
                result[key] = list(vector)

        return result

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
            await asyncio.to_thread(self._index.save, path)

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._index.load, path)
