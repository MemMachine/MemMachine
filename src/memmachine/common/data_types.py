"""Common data types for MemMachine."""

from enum import Enum

FilterablePropertyValue = bool | int | str


class SimilarityMetric(Enum):
    """Similarity metrics supported by embedding operations."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class ExternalServiceAPIError(Exception):
    """Raised when an API error occurs for an external service."""
