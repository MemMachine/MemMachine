"""Shared utilities for vector store implementations."""

import math
import re
from collections.abc import Iterable, Sequence
from uuid import UUID

# Matched with fullmatch: `$` also matches before a trailing newline.
_IDENTIFIER_RE = re.compile(r"[a-z0-9_]+")
_IDENTIFIER_MAX_BYTES = 32


def validate_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.fullmatch(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )


def require_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the partition key is a valid identifier."""
    if not validate_identifier(partition_key):
        raise ValueError(
            f"Partition key {partition_key!r} must match [a-z0-9_]+ and be at most "
            "32 bytes"
        )


def require_distinct_record_uuids(record_uuids: Iterable[UUID]) -> None:
    """Raise ValueError if a record UUID occurs more than once."""
    seen: set[UUID] = set()
    for record_uuid in record_uuids:
        if record_uuid in seen:
            raise ValueError(f"Record UUID {record_uuid} occurs more than once")
        seen.add(record_uuid)


def require_valid_query_vector(query_vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a query vector has the store's dimensions and finite coordinates."""
    require_dimensions(query_vector, dimensions)
    if not all(map(math.isfinite, query_vector)):
        position = next(
            position
            for position, coordinate in enumerate(query_vector)
            if not math.isfinite(coordinate)
        )
        raise ValueError(
            f"Query vector coordinate {position} is not finite: "
            f"{query_vector[position]}"
        )


def require_valid_min_cosine_similarity(min_cosine_similarity: float | None) -> None:
    """Raise ValueError if a minimum cosine similarity is not finite; None means no minimum."""
    if min_cosine_similarity is not None and not math.isfinite(min_cosine_similarity):
        raise ValueError(
            f"Minimum cosine similarity is not finite: {min_cosine_similarity}"
        )


def require_valid_limit(limit: int) -> None:
    """Raise ValueError unless a query limit is positive."""
    if not limit > 0:
        raise ValueError(f"Limit is not positive: {limit}")


def require_dimensions(vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a vector has the store's dimensions."""
    if len(vector) != dimensions:
        raise ValueError(
            f"Vector has {len(vector)} dimensions; the store has {dimensions}"
        )
