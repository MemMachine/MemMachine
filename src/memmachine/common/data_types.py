"""Common data types for MemMachine."""

from datetime import datetime
from enum import Enum

PropertyValue = bool | int | float | str | datetime
"""Type for stored property values."""

FilterValue = bool | int | float | str | datetime | list[int] | list[str]
"""Type for filter expression values (includes list types for IN clauses)."""

OrderedValue = int | float | str | datetime
"""Type for values that can be ordered/sorted."""


class SimilarityMetric(Enum):
    """Similarity metrics supported by embedding operations. All scores are normalized to [0, 1], where higher is more similar."""

    """
    COSINE
    with cosine_similarity as the raw cosine similarity in [-1, 1]:
    score formula: S = (1 + cosine_similarity) / 2, where cosine_similarity is the raw cosine in [-1, 1].
    inverse formula: cosine_similarity = 2 * S - 1
    """
    COSINE = "cosine"

    """
    DOT
    with dot_product as the raw dot product in (-inf, inf):
    score formula: S = 1 / (1 + exp(-2 * arcsinh(dot_product)))
    inverse formula: dot_product = (2 * S - 1) / (2 * sqrt(S * (1 - S)))
    """
    DOT = "dot"

    """
    EUCLIDEAN
    with euclidean_distance as the raw Euclidean distance in [0, inf):
    score formula: S = 1 / (1 + euclidean_distance)
    inverse formula: euclidean_distance = 1 / S - 1
    """
    EUCLIDEAN = "euclidean"

    """
    MANHATTAN
    with manhattan_distance as the raw Manhattan distance in [0, inf):
    score formula: S = 1 / (1 + manhattan_distance)
    inverse formula: manhattan_distance = 1 / S - 1
    """
    MANHATTAN = "manhattan"


class ExternalServiceAPIError(Exception):
    """Raised when an API error occurs for an external service."""
