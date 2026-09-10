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

from sqlalchemy import ColumnElement, and_, or_

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)
from memmachine_server.common.properties_json import (
    PROPERTY_TYPE_KEY,
    PROPERTY_VALUE_KEY,
)
from memmachine_server.common.utils import ensure_tz_aware

from .filter_expression import (
    And,
    Equals,
    FilterExpr,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
    OrderingOp,
)

type _LeafExpr = Equals | NotEquals | Ordering | In | IsMissing

FieldEncoding = Literal["column", "json", "properties_json"]

FieldResolver = Callable[[str], tuple[ColumnElement, FieldEncoding]]
"""
Maps a filter field name to a ``(column, kind)`` pair.

Resolvers should call ``.expression`` on ORM ``InstrumentedAttribute``
values to obtain a ``ColumnElement``.

Raises `ValueError` for unrecognised fields.
"""


ORDERING_OPS: dict[
    OrderingOp, Callable[[ColumnElement, object], ColumnElement[bool]]
] = {
    ">": lambda col, val: col > val,
    "<": lambda col, val: col < val,
    ">=": lambda col, val: col >= val,
    "<=": lambda col, val: col <= val,
}


def _compile_column_leaf(
    expr: _LeafExpr,
    column: ColumnElement,
) -> ColumnElement[bool]:
    match expr:
        case IsMissing():
            return column.is_(None)
        case In(values=values):
            return column.in_(values)
        case Equals(value=value):
            return column == value
        case NotEquals(value=value):
            return column != value
        case Ordering(op=op, value=value):
            return ORDERING_OPS[op](column, value)


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
    expr: _LeafExpr,
    column: ColumnElement,
) -> ColumnElement[bool]:
    match expr:
        case IsMissing():
            # .as_string() emits ->> instead of JSON_QUOTE(JSON_EXTRACT(...)),
            # which preserves SQL NULL for missing keys on SQLite.
            return column.as_string().is_(None)
        case In(values=values):
            return _cast_json_value(column, values[0]).in_(values)
        case Equals(value=value):
            return _cast_json_value(column, _check_json_value(value)) == value
        case NotEquals(value=value):
            return _cast_json_value(column, _check_json_value(value)) != value
        case Ordering(op=op, value=value):
            return ORDERING_OPS[op](
                _cast_json_value(column, _check_json_value(value)), value
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
    return value_path.as_string(), value


def _properties_json_type_check(
    column: ColumnElement,
    property_type: type[PropertyValue],
) -> ColumnElement[bool]:
    """Restrict a typed-JSON field to values stored with the given type."""
    type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]
    return column[PROPERTY_TYPE_KEY].as_string() == type_name


def _compile_properties_json_leaf(
    expr: _LeafExpr,
    column: ColumnElement,
) -> ColumnElement[bool]:
    match expr:
        case IsMissing():
            return column.as_string().is_(None)
        case In(values=values):
            # Values are homogeneous, so the first one names the type for all.
            type_check = _properties_json_type_check(column, type(values[0]))
            value_path = column[PROPERTY_VALUE_KEY]
            if isinstance(values[0], int):
                return and_(type_check, value_path.as_integer().in_(values))
            return and_(type_check, value_path.as_string().in_(values))
        case Equals(value=value) | NotEquals(value=value) | Ordering(value=value):
            type_check = _properties_json_type_check(column, type(value))
            casted_column, normalized_value = _cast_properties_json_value(
                column[PROPERTY_VALUE_KEY], value
            )
            match expr:
                case Equals():
                    comparison = casted_column == normalized_value
                case NotEquals():
                    comparison = casted_column != normalized_value
                case Ordering(op=op):
                    comparison = ORDERING_OPS[op](casted_column, normalized_value)
            return and_(type_check, comparison)


def compile_sql_filter(
    expr: FilterExpr,
    resolve_field: FieldResolver,
) -> ColumnElement[bool]:
    """
    Compile a FilterExpr tree into a SQLAlchemy boolean expression.

    The `resolve_field` callback maps each field name to a
    `(column, FieldEncoding)` pair and raises `ValueError` for unknown fields.

    Datetime values arrive already normalized -- a node converts them to
    UTC-aware instants at construction -- so column-encoded leaves bind them
    as-is, and the JSON-text encodings only choose a representation.
    """
    match expr:
        case Equals() | NotEquals() | Ordering() | In() | IsMissing():
            column, kind = resolve_field(expr.field)
            match kind:
                case "column":
                    return _compile_column_leaf(expr, column)
                case "json":
                    return _compile_json_leaf(expr, column)
                case "properties_json":
                    return _compile_properties_json_leaf(expr, column)
        case And(operands):
            return and_(*(compile_sql_filter(o, resolve_field) for o in operands))
        case Or(operands):
            return or_(*(compile_sql_filter(o, resolve_field) for o in operands))
        case Not(operand):
            return ~compile_sql_filter(operand, resolve_field)
