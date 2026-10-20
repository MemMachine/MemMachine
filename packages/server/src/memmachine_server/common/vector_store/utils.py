"""Shared utilities for vector store implementations."""

import math
import re
from collections.abc import Mapping, Sequence

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)
from memmachine_server.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine_server.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine_server.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine_server.common.filter.filter_parser import (
    Or as FilterOr,
)

# Matched with fullmatch: `$` also matches before a trailing newline.
_IDENTIFIER_RE = re.compile(r"[a-z0-9_]+")
_IDENTIFIER_MAX_BYTES = 32
_PARTITION_KEY_MAX_BYTES = 255


def is_valid_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.fullmatch(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )


def validate_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the partition key is non-empty, at most 255 bytes, and free of NUL."""
    # PostgreSQL text cannot hold NUL, and the registry may live there.
    if (
        not partition_key
        or "\x00" in partition_key
        or len(partition_key.encode()) > _PARTITION_KEY_MAX_BYTES
    ):
        raise ValueError(
            f"Partition key {partition_key!r} must be non-empty, at most "
            f"{_PARTITION_KEY_MAX_BYTES} bytes, and free of NUL characters"
        )


def validate_declared_types(
    properties: Mapping[str, PropertyValue],
    indexed_properties: Mapping[str, type[PropertyValue]],
) -> None:
    """Raise unless every declared property holds a value of its declared type."""
    for key, value in properties.items():
        declared_type = indexed_properties.get(key)
        # `bool` is an `int` at runtime and is its own property type here.
        if declared_type is not None and type(value) is not declared_type:
            raise ValueError(
                f"Property {key!r} is declared as "
                f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared_type]}, "
                f"got {type(value).__name__} {value!r}."
            )


def validate_query_vector(query_vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a query vector has the store's dimensions and finite coordinates."""
    validate_dimensions(query_vector, dimensions)
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


def validate_score_threshold(score_threshold: float | None) -> None:
    """Raise ValueError if a score threshold is not finite; None means no threshold."""
    if score_threshold is not None and not math.isfinite(score_threshold):
        raise ValueError(f"Score threshold is not finite: {score_threshold}")


def validate_limit(limit: int) -> None:
    """Raise ValueError unless a query limit is positive."""
    if not limit > 0:
        raise ValueError(f"Limit is not positive: {limit}")


def validate_dimensions(vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a vector has the store's dimensions."""
    if len(vector) != dimensions:
        raise ValueError(
            f"Vector has {len(vector)} dimensions; the store has {dimensions}"
        )


def is_valid_filter(expr: FilterExpr) -> bool:
    """Return whether all field names in the filter tree are valid identifiers."""
    if isinstance(expr, (FilterComparison, FilterIn, FilterIsNull)):
        return is_valid_identifier(expr.field)
    if isinstance(expr, FilterNot):
        return is_valid_filter(expr.expr)
    if isinstance(expr, (FilterAnd, FilterOr)):
        return is_valid_filter(expr.left) and is_valid_filter(expr.right)
    raise TypeError(f"Unsupported filter expression type: {type(expr)}")
