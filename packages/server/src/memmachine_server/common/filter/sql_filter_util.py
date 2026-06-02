"""
Unified SQLAlchemy filter compiler for FilterExpr trees.

Supports different field encodings via `FieldEncoding`:
- `"column"`: direct column comparison, no casting.
- `"json"`: raw JSON value, cast based on the Python value type.
- `"properties_json"`: type-tagged JSON (`{"t": …, "v": …}`) supporting PropertyValue.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import ColumnElement, and_, false, or_

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.properties_json import (
    PROPERTY_TYPE_KEY,
    PROPERTY_VALUE_KEY,
)
from memmachine_server.common.utils import ensure_tz_aware

FieldEncoding = Literal["column", "json", "properties_json"]

FieldResolver = Callable[[str], tuple[ColumnElement, FieldEncoding]]
"""
Maps a filter field name to a ``(column, kind)`` pair.

Resolvers should call ``.expression`` on ORM ``InstrumentedAttribute``
values to obtain a ``ColumnElement``.

Raises `ValueError` for unrecognised fields.
"""


_COMPARISON_OPS: dict[str, Callable[[ColumnElement, object], ColumnElement[bool]]] = {
    "=": lambda col, val: col == val,
    "!=": lambda col, val: col != val,
    ">": lambda col, val: col > val,
    "<": lambda col, val: col < val,
    ">=": lambda col, val: col >= val,
    "<=": lambda col, val: col <= val,
}


def _get_op(op: str) -> Callable[[ColumnElement, object], ColumnElement[bool]]:
    op_fn = _COMPARISON_OPS.get(op)
    if op_fn is None:
        raise ValueError(f"Unsupported operator: {op!r}")
    return op_fn


def _compile_column_leaf(
    expr: IsNull | In | Comparison,
    column: ColumnElement,
) -> ColumnElement[bool]:
    if isinstance(expr, IsNull):
        return column.is_(None)
    if isinstance(expr, In):
        if not expr.values:
            return false()
        return column.in_(expr.values)
    return _get_op(expr.op)(column, expr.value)


def _cast_json_value(
    column: ColumnElement,
    value: bool | float | str,
) -> ColumnElement:
    """Cast a raw JSON path element based on the Python value type."""
    if isinstance(value, bool):
        return column.as_boolean()
    if isinstance(value, int):
        return column.as_integer()
    if isinstance(value, float):
        return column.as_float()
    return column.as_string()


def _check_json_value(value: PropertyValue) -> bool | int | float | str:
    """Validate that a filter value is usable with raw JSON fields."""
    if isinstance(value, datetime):
        raise TypeError(
            "datetime filtering requires 'properties_json' fields; "
            "raw 'json' fields do not support datetime"
        )
    return value


def _compile_json_leaf(
    expr: IsNull | In | Comparison,
    column: ColumnElement,
) -> ColumnElement[bool]:
    if isinstance(expr, IsNull):
        # .as_string() emits ->> instead of JSON_QUOTE(JSON_EXTRACT(...)),
        # which preserves SQL NULL for missing keys on SQLite.
        return column.as_string().is_(None)
    if isinstance(expr, In):
        if not expr.values:
            return false()
        return _cast_json_value(column, _check_json_value(expr.values[0])).in_(
            expr.values
        )
    return _get_op(expr.op)(
        _cast_json_value(column, _check_json_value(expr.value)), expr.value
    )


def _cast_properties_json_value(
    value_path: ColumnElement,
    value: PropertyValue,
) -> tuple[ColumnElement, object]:
    """Cast a typed-JSON value path and normalize the comparison value."""
    if isinstance(value, bool):
        return value_path.as_boolean(), value
    if isinstance(value, int):
        return value_path.as_integer(), value
    if isinstance(value, float):
        return value_path.as_float(), value
    if isinstance(value, datetime):
        return (
            value_path.as_string(),
            ensure_tz_aware(value).astimezone(UTC).isoformat(),
        )
    if isinstance(value, str):
        return value_path.as_string(), value
    raise TypeError(f"Unsupported property value type: {type(value)!r}")


def _compile_properties_json_leaf(
    expr: IsNull | In | Comparison,
    column: ColumnElement,
) -> ColumnElement[bool]:
    if isinstance(expr, IsNull):
        return column.as_string().is_(None)

    if isinstance(expr, In):
        if not expr.values:
            return false()
        first_value = expr.values[0]
        type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[type(first_value)]
        value_path = column[PROPERTY_VALUE_KEY]
        type_check = column[PROPERTY_TYPE_KEY].as_string() == type_name
        if isinstance(first_value, int):
            return and_(type_check, value_path.as_integer().in_(expr.values))
        return and_(type_check, value_path.as_string().in_(expr.values))

    # Comparison
    type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[type(expr.value)]
    value_path = column[PROPERTY_VALUE_KEY]
    type_check = column[PROPERTY_TYPE_KEY].as_string() == type_name
    casted_column, normalized_value = _cast_properties_json_value(
        value_path, expr.value
    )
    return and_(type_check, _get_op(expr.op)(casted_column, normalized_value))


def compile_sql_filter(
    expr: FilterExpr,
    resolve_field: FieldResolver,
) -> ColumnElement[bool]:
    """
    Compile a FilterExpr tree into a SQLAlchemy boolean expression.

    The `resolve_field` callback maps each field name to a
    `(column, FieldEncoding)` pair and raises `ValueError` for unknown fields.
    """
    if isinstance(expr, Comparison | In | IsNull):
        column, kind = resolve_field(expr.field)
        if kind == "column":
            return _compile_column_leaf(expr, column)
        if kind == "json":
            return _compile_json_leaf(expr, column)
        if kind == "properties_json":
            return _compile_properties_json_leaf(expr, column)
        raise ValueError(f"Unknown field kind: {kind!r}")

    if isinstance(expr, And):
        return and_(
            compile_sql_filter(expr.left, resolve_field),
            compile_sql_filter(expr.right, resolve_field),
        )

    if isinstance(expr, Or):
        return or_(
            compile_sql_filter(expr.left, resolve_field),
            compile_sql_filter(expr.right, resolve_field),
        )

    if isinstance(expr, Not):
        return ~compile_sql_filter(expr.expr, resolve_field)

    raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
