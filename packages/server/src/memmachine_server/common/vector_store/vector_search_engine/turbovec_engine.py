"""turbovec (TurboQuant) implementation of VectorSearchEngine."""

import asyncio
import math
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import ClassVar, override

import numpy as np
from turbovec import IdMapIndex

from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class TurboVecVectorSearchEngine(VectorSearchEngine):
    """
    Vector search engine backed by turbovec.

    turbovec indexes a dimensionality that is a multiple of 8, so a vector of
    any other width is zero-padded up to one. Padding is exact -- a zero
    coordinate adds nothing to an inner product and nothing to a norm -- so
    the padded index answers as the unpadded one would, and any embedding
    width the other engines accept works here too.
    """

    _VALID_BIT_WIDTHS: ClassVar[frozenset[int]] = frozenset({2, 3, 4})
    _DEFAULT_BIT_WIDTH: ClassVar[int] = 4
    _OVERFETCH_BASE: ClassVar[int] = 4

    def __init__(
        self,
        *,
        num_dimensions: int,
        bit_width: int = _DEFAULT_BIT_WIDTH,
    ) -> None:
        """Initialize."""
        if bit_width not in self._VALID_BIT_WIDTHS:
            raise ValueError(
                f"turbovec bit_width must be one of "
                f"{sorted(self._VALID_BIT_WIDTHS)}, got {bit_width}"
            )

        self._num_dimensions = num_dimensions
        self._padded_dimensions = math.ceil(num_dimensions / 8) * 8
        self._index = IdMapIndex(dim=self._padded_dimensions, bit_width=bit_width)

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        await asyncio.to_thread(self._sync_add, vectors)

    def _sync_add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        keys = np.array(list(vectors.keys()), dtype=np.uint64)
        array = self._prepare_vectors(list(vectors.values()))
        self._index.add_with_ids(array, keys)

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        *,
        limit: int,
        allowed_keys: Container[int] | None = None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if not vectors or limit <= 0:
            return [SearchResult(matches=[]) for _ in vectors]
        return await asyncio.to_thread(self._sync_search, vectors, limit, allowed_keys)

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
        allowed_keys: Container[int] | None,
    ) -> list[SearchResult]:
        query = self._prepare_vectors(vectors)
        num_queries = query.shape[0]
        size = len(self._index)
        if size == 0:
            return [SearchResult(matches=[]) for _ in vectors]

        # turbovec has a native allowlist search, but it needs the allowed ids
        # enumerated and `allowed_keys` only answers membership. So, like the
        # engines beside it, this one fetches unrestricted, drops what the
        # filter rejects, and widens the fetch until `limit` survive or the
        # whole index has been scanned.
        overfetch_factor = 1 if allowed_keys is None else self._OVERFETCH_BASE
        final_results: list[SearchResult | None] = [None] * num_queries
        pending_indices = list(range(num_queries))
        while pending_indices:
            pending_query = query[pending_indices]
            fetch_limit = min(limit * overfetch_factor, size)
            inner_products, ids = self._index.search(pending_query, fetch_limit)
            still_pending: list[int] = []
            for batch_index, original_index in enumerate(pending_indices):
                matches: list[SearchMatch] = []
                for inner_product, key in zip(
                    inner_products[batch_index], ids[batch_index], strict=True
                ):
                    int_key = int(key)
                    if allowed_keys is not None and int_key not in allowed_keys:
                        continue
                    matches.append(
                        SearchMatch(
                            key=int_key,
                            cosine_similarity=self._to_cosine_similarity(inner_product),
                        )
                    )
                    if len(matches) >= limit:
                        break
                if len(matches) >= limit or fetch_limit >= size:
                    final_results[original_index] = SearchResult(matches=matches)
                else:
                    still_pending.append(original_index)
            pending_indices = still_pending
            overfetch_factor *= self._OVERFETCH_BASE
        return [
            result if result is not None else SearchResult(matches=[])
            for result in final_results
        ]

    @staticmethod
    def _to_cosine_similarity(inner_product: float) -> float:
        """Clamp a quantized inner product onto the cosine range."""
        return min(1.0, max(-1.0, float(inner_product)))

    def _prepare_vectors(self, vectors: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.zeros((len(vectors), self._padded_dimensions), dtype=np.float32)
        try:
            array[:, : self._num_dimensions] = vectors
        except ValueError as error:
            raise ValueError(
                f"vectors must have {self._num_dimensions} dimensions"
            ) from error
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return array / norms

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        await asyncio.to_thread(self._sync_remove, keys)

    def _sync_remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            self._index.remove(key)

    @override
    async def save(self, path: str) -> None:
        await asyncio.to_thread(self._sync_save, path)

    def _sync_save(self, path: str) -> None:
        self._index.sync(path)

    @override
    async def load(self, path: str) -> None:
        self._index = await asyncio.to_thread(self._sync_load, path)

    def _sync_load(self, path: str) -> IdMapIndex:
        return IdMapIndex.load(path)
