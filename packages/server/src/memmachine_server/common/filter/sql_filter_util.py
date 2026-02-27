"""SQLAlchemy utilities for FilterExpr."""

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import ColumnElement, and_, false, or_

from memmachine_server.common.data_types import FilterValue
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)

logger = logging.getLogger(__name__)

FieldResolver = Callable[[str], tuple[Any, bool]]


def _cast_json_column(
    column: ColumnElement[Any],
    value: FilterValue,
) -> ColumnElement[Any]:
    """Cast a JSON path column to the appropriate SQL type based on *value*.

    Note: datetime values are converted to ISO strings for comparison.
    """
    if isinstance(value, bool):
        return column.as_boolean()
    if isinstance(value, int):
        return column.as_integer()
    if isinstance(value, float):
        return column.as_float()
    return column.as_string()


_COMPARISON_OPS = {
    "=": lambda col, val: col == val,
    "!=": lambda col, val: col != val,
    ">": lambda col, val: col > val,
    "<": lambda col, val: col < val,
    ">=": lambda col, val: col >= val,
    "<=": lambda col, val: col <= val,
}


def _compile_leaf(
    expr: IsNull | In | Comparison,
    resolve_field: FieldResolver,
) -> ColumnElement[bool] | None:
    column, is_json = resolve_field(expr.field)
    if column is None:
        logger.warning("Unsupported filter field: %s", expr.field)
        return None

    if isinstance(expr, IsNull):
        if is_json:
            return column.as_string().is_(None)
        return column.is_(None)

    if isinstance(expr, In):
        if not expr.values:
            return false()
        if is_json:
            return _cast_json_column(column, expr.values[0]).in_(expr.values)
        return column.in_(expr.values)

    op_fn = _COMPARISON_OPS.get(expr.op)
    if op_fn is None:
        raise ValueError(f"Unsupported operator: {expr.op}")

    if is_json:
        return op_fn(_cast_json_column(column, expr.value), expr.value)
    return op_fn(column, expr.value)


def compile_sql_filter(
    expr: FilterExpr,
    resolve_field: FieldResolver,
) -> ColumnElement[bool] | None:
    """Compile a FilterExpr tree into an SQLAlchemy boolean expression."""
    if isinstance(expr, (IsNull, In, Comparison)):
        return _compile_leaf(expr, resolve_field)

    if isinstance(expr, And):
        left = compile_sql_filter(expr.left, resolve_field)
        right = compile_sql_filter(expr.right, resolve_field)
        if left is None:
            return right
        if right is None:
            return left
        return and_(left, right)

    if isinstance(expr, Or):
        left = compile_sql_filter(expr.left, resolve_field)
        right = compile_sql_filter(expr.right, resolve_field)
        if left is None:
            return right
        if right is None:
            return left
        return or_(left, right)

    if isinstance(expr, Not):
        inner = compile_sql_filter(expr.expr, resolve_field)
        if inner is None:
            return None
        return ~inner

    raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
