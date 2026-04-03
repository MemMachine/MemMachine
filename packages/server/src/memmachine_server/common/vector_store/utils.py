"""Shared utilities for vector store implementations."""

import re

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

_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]+$")
_IDENTIFIER_MAX_BYTES = 32


def validate_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.match(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
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
