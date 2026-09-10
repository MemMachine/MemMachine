"""Shared utilities for vector search engine implementations."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .vector_search_engine import SearchMatch


def unit_normalize(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Scale each row to unit length, leaving zero rows as they are."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def inner_products(
    query_vector: Sequence[float], matrix: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inner product of `query_vector` with each row of `matrix`."""
    return matrix @ np.asarray(query_vector, dtype=np.float32)


def top_k_matches(
    query_vector: Sequence[float],
    keys: Sequence[int],
    matrix: npt.NDArray[np.float32],
    limit: int,
) -> list[SearchMatch]:
    """The best `limit` rows of `matrix` as matches keyed by `keys`.

    `query_vector` and every row of `matrix` must be a unit vector, so that
    the inner product this ranks by is their cosine similarity. Scores are
    clamped onto the cosine range, which quantized storage can otherwise
    carry a fraction outside.
    """
    similarities = inner_products(query_vector, matrix)
    k = min(limit, similarities.shape[0])
    if k <= 0:
        return []
    top = np.argpartition(-similarities, k - 1)[:k]
    top = top[np.argsort(-similarities[top])]
    return [
        SearchMatch(
            key=keys[index],
            cosine_similarity=min(1.0, max(-1.0, float(similarities[index]))),
        )
        for index in top
    ]
