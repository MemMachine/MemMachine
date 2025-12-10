"""SQLAlchemy utilities for FilterExpr."""

import logging
from typing import Any

from sqlalchemy import ColumnElement
from sqlalchemy.orm import InstrumentedAttribute, MappedColumn

from memmachine.common.data_types import FilterablePropertyValue
from memmachine.common.filter.filter_parser import Comparison

logger = logging.getLogger(__name__)


def _normalize_metadata_value(
    value: FilterablePropertyValue | list[FilterablePropertyValue],
) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def _ensure_scalar_value(
    value: FilterablePropertyValue | list[FilterablePropertyValue], op: str
) -> FilterablePropertyValue:
    if isinstance(value, list):
        raise TypeError(f"'{op}' comparison cannot accept list values")
    return value


def _ensure_list_value(
    value: FilterablePropertyValue | list[FilterablePropertyValue],
) -> list[FilterablePropertyValue]:
    if not isinstance(value, list):
        raise TypeError("IN comparison requires a list of values")
    return value


def parse_sql_filter(
    column: MappedColumn[Any] | InstrumentedAttribute[Any] | None,
    is_metadata: bool,
    expr: Comparison,
) -> ColumnElement[bool] | None:
    """Parse a FilterExpr comparison into an SQLAlchemy boolean expression."""
    if column is None:
        logger.warning("Unsupported feature filter field: %s", expr.field)
        return None

    op = expr.op
    normalize = _normalize_metadata_value if is_metadata else lambda v: v

    match op:
        case "is_null":
            return column.is_(None)
        case "is_not_null":
            return column.is_not(None)
        case "in":
            values = _ensure_list_value(expr.value)
            if is_metadata:
                values = [normalize(v) for v in values]
            return column.in_(values)
        case ">" | "<" | ">=" | "<=" | "=":
            value = normalize(_ensure_scalar_value(expr.value, op))
            return {
                ">": column > value,
                "<": column < value,
                ">=": column >= value,
                "<=": column <= value,
                "=": column == value,
            }[op]
        case _:
            raise ValueError(f"Unsupported operator: {expr.op}")
