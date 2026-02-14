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
    """Similarity metrics supported by embedding operations."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class ExternalServiceAPIError(Exception):
    """Raised when an API error occurs for an external service."""
