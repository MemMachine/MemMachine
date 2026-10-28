"""Exact search over gathered vectors, shared by the engine implementations."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import SearchMatch, SearchResult


def exact_search_results(
    query_vectors: npt.NDArray[np.float32],
    keys: Sequence[int],
    vectors: npt.NDArray[np.float32],
    similarity_metric: SimilarityMetric,
    limit: int,
) -> list[SearchResult]:
    """
    The best `limit` of `vectors` for each query vector, scored exactly.

    `vectors` holds one row per key in `keys`, in the same order. Scores
    are the metric's own, as `SearchMatch` defines them, and each result
    is ordered best first.
    """
    if not keys:
        return [SearchResult(matches=[]) for _ in query_vectors]

    match similarity_metric:
        case SimilarityMetric.COSINE:
            scores = _unit_rows(query_vectors) @ _unit_rows(vectors).T
        case SimilarityMetric.DOT:
            scores = query_vectors @ vectors.T
        case SimilarityMetric.EUCLIDEAN:
            scores = np.stack(
                [
                    np.linalg.norm(vectors - query_vector, axis=1)
                    for query_vector in query_vectors
                ]
            )
        case _:
            raise NotImplementedError(similarity_metric)

    # Rank by ascending order of `ranks`, best first.
    ranks = -scores if similarity_metric.higher_is_better else scores
    k = min(limit, len(keys))
    results: list[SearchResult] = []
    for query_scores, query_ranks in zip(scores, ranks, strict=True):
        top = np.argpartition(query_ranks, k - 1)[:k]
        top = top[np.argsort(query_ranks[top])]
        results.append(
            SearchResult(
                matches=[
                    SearchMatch(score=float(query_scores[index]), key=keys[index])
                    for index in top
                ]
            )
        )
    return results


def _unit_rows(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Each row scaled to unit length; a zero row stays zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
