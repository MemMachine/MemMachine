"""USearch HNSW implementation of VectorSearchEngine."""

import asyncio
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import ClassVar, override

import numpy as np
from usearch.index import Index, MetricKind

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.rw_locks import AsyncRWLock

from .utils import exact_search_results
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
        allowlist: Collection[int] | None = None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if self._index.size == 0 or not vectors:
            return [SearchResult(matches=[]) for _ in vectors]

        async with self._lock.read_lock():
            if allowlist is not None:
                return await asyncio.to_thread(
                    self._sync_search_allowlist, vectors, limit, allowlist
                )
            return await asyncio.to_thread(self._sync_search, vectors, limit)

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
    ) -> list[SearchResult]:
        results = self._index.search(
            np.array(vectors, dtype=np.float32), min(limit, self._index.size)
        )
        return [
            SearchResult(
                matches=[
                    SearchMatch(
                        key=int(key), score=self._distance_to_score(float(dist))
                    )
                    for key, dist in zip(keys, distances, strict=True)
                    if int(key) >= 0
                ]
            )
            for keys, distances in zip(
                np.atleast_2d(results.keys),
                np.atleast_2d(results.distances),
                strict=True,
            )
        ]

    def _sync_search_allowlist(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
        allowlist: Collection[int],
    ) -> list[SearchResult]:
        stored_keys, stored_vectors = self._stored_vectors(allowlist)
        return exact_search_results(
            np.array(vectors, dtype=np.float32),
            stored_keys,
            stored_vectors,
            self._similarity_metric,
            limit,
        )

    def _stored_vectors(self, keys: Collection[int]) -> tuple[list[int], np.ndarray]:
        """The keys among `keys` that hold a vector, and their vectors in that order."""
        keys = list(keys)
        if not keys:
            return [], np.empty((0, self._index.ndim), dtype=np.float32)
        vectors = self._index.get(np.array(keys, dtype=np.int64))
        # For an array of keys, usearch answers with a tuple: a vector, or
        # None, per key.
        assert isinstance(vectors, tuple)
        stored = [
            (key, vector)
            for key, vector in zip(keys, vectors, strict=True)
            if vector is not None
        ]
        return [key for key, _ in stored], np.array(
            [vector for _, vector in stored], dtype=np.float32
        ).reshape(len(stored), self._index.ndim)

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
