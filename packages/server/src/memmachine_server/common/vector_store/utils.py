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


def validate_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.fullmatch(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )


def require_declared_types(
    properties: Mapping[str, PropertyValue],
    indexed_properties_schema: Mapping[str, type[PropertyValue]],
) -> None:
    """Raise unless every declared property holds a value of its declared type."""
    for key, value in properties.items():
        declared_type = indexed_properties_schema.get(key)
        # `bool` is an `int` at runtime and is its own property type here.
        if declared_type is not None and type(value) is not declared_type:
            raise ValueError(
                f"Property {key!r} is declared as "
                f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared_type]}, "
                f"got {type(value).__name__} {value!r}."
            )


def require_valid_query_vector(query_vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a query vector has the collection's dimensions and finite coordinates."""
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


def require_valid_score_threshold(score_threshold: float | None) -> None:
    """Raise ValueError if a score threshold is not finite; None means no threshold."""
    if score_threshold is not None and not math.isfinite(score_threshold):
        raise ValueError(f"Score threshold is not finite: {score_threshold}")


def require_valid_limit(limit: int) -> None:
    """Raise ValueError unless a query limit is positive."""
    if not limit > 0:
        raise ValueError(f"Limit is not positive: {limit}")


def require_dimensions(vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a vector has the collection's dimensions."""
    if len(vector) != dimensions:
        raise ValueError(
            f"Vector has {len(vector)} dimensions; the collection has {dimensions}"
        )


def validate_filter(expr: FilterExpr) -> bool:
    """Return whether all field names in the filter tree are valid identifiers."""
    if isinstance(expr, (FilterComparison, FilterIn, FilterIsNull)):
        return validate_identifier(expr.field)
    if isinstance(expr, FilterNot):
        return validate_filter(expr.expr)
    if isinstance(expr, (FilterAnd, FilterOr)):
        return validate_filter(expr.left) and validate_filter(expr.right)
    raise TypeError(f"Unsupported filter expression type: {type(expr)}")
