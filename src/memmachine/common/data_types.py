"""Common data types for MemMachine."""

from datetime import datetime
from enum import Enum

AttributeValue = bool | int | float | str | datetime

FilterValue = bool | int | float | str | datetime | list[int] | list[str]


class SimilarityMetric(Enum):
    """Similarity metrics supported by embedding operations."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class ExternalServiceAPIError(Exception):
    """Raised when an API error occurs for an external service."""
