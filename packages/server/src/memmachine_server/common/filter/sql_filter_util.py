"""
Unified SQLAlchemy filter compiler for FilterExpr trees.

Supports different field encodings via `FieldEncoding`:
- `"column"`: direct column comparison, no casting.
- `"json"`: raw JSON value, cast based on the Python value type.
- `"properties_json"`: type-tagged JSON (`{"t": …, "v": …}`) supporting PropertyValue.

Both JSON encodings hold dynamically typed values and compare numbers by
value: an int and a float are one numeric type, compared exactly as SQL
NUMERIC.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import (
    ColumnElement,
    Numeric,
    and_,
    case,
    cast,
    false,
    func,
    literal,
    or_,
    true,
)

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
    IsNull,
    Not,
    Or,
    Ordering,
    OrderingOp,
    Property,
    SystemField,
)

type _LeafExpr = Equals | Ordering | In | IsNull

FieldEncoding = Literal["column", "json", "properties_json", "uuid"]

FieldResolver = Callable[[SystemField | Property], tuple[ColumnElement, FieldEncoding]]
"""
Maps a filter field to a ``(column, kind)`` pair.

Resolvers should call ``.expression`` on ORM ``InstrumentedAttribute``
values to obtain a ``ColumnElement``.

Raises `ValueError` for unrecognized fields.
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
        case IsNull():
            return column.is_(None)
        case In(values=values):
            return column.in_(values) if values else false()
        case Equals(value=value):
            return column == value
        case Ordering(op=op, value=value):
            return ORDERING_OPS[op](column, value)


def _compile_uuid_leaf(
    expr: _LeafExpr,
    column: ColumnElement,
) -> ColumnElement[bool]:
    match expr:
        case IsNull():
            return column.is_(None)
        case In(values=values):
            return (
                column.in_([UUID(str(value)) for value in values])
                if values
                else false()
            )
        case Equals(value=value):
            return column == UUID(str(value))
        case Ordering(op=op, value=value):
            return ORDERING_OPS[op](column, UUID(str(value)))


def _cast_json_value(
    column: ColumnElement,
    value: bool | float | str,
) -> tuple[ColumnElement, object]:
    """Cast a raw JSON path element and the comparison value based on the Python value type."""
    if isinstance(value, bool):
        return column.as_boolean(), value
    if isinstance(value, int | float):
        return cast(column.as_string(), Numeric), _number_literal(value)
    return column.as_string(), value


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
        case IsNull():
            # .as_string() emits ->> instead of JSON_QUOTE(JSON_EXTRACT(...)),
            # which preserves SQL NULL for missing keys on SQLite.
            return column.as_string().is_(None)
        case In(values=values):
            if not values:
                return false()
            # Values are homogeneous: all of them are ints, or none is.
            numbers = [value for value in values if isinstance(value, int)]
            if numbers:
                return cast(column.as_string(), Numeric).in_(
                    [_number_literal(number) for number in numbers]
                )
            return column.as_string().in_(values)
        case Equals(value=value):
            casted_column, compared = _cast_json_value(column, _check_json_value(value))
            return casted_column == compared
        case Ordering(op=op, value=value):
            casted_column, compared = _cast_json_value(column, _check_json_value(value))
            return ORDERING_OPS[op](casted_column, compared)


def _cast_properties_json_value(
    value_path: ColumnElement,
    value: PropertyValue,
) -> tuple[ColumnElement, object]:
    """Cast a typed-JSON value path and normalize the comparison value."""
    if isinstance(value, bool):
        return value_path.as_boolean(), value
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
        case IsNull():
            return column.as_string().is_(None)
        case In(values=values):
            if not values:
                return false()
            # Values are homogeneous: all of them are ints, or none is.
            numbers = [value for value in values if isinstance(value, int)]
            if numbers:
                return _properties_json_number(column).in_(
                    [_number_literal(number) for number in numbers]
                )
            type_check = _properties_json_type_check(column, str)
            return and_(type_check, column[PROPERTY_VALUE_KEY].as_string().in_(values))
        case Equals(value=value) | Ordering(value=value):
            if isinstance(value, bool) or not isinstance(value, int | float):
                held, compared = _cast_properties_json_value(
                    column[PROPERTY_VALUE_KEY], value
                )
                type_check = _properties_json_type_check(column, type(value))
            else:
                held, compared = _properties_json_number(column), _number_literal(value)
                type_check = true()
            match expr:
                case Equals():
                    comparison = held == compared
                case Ordering(op=op):
                    comparison = ORDERING_OPS[op](held, compared)
            return and_(type_check, comparison)


def _properties_json_number(column: ColumnElement) -> ColumnElement:
    """A typed-JSON field's value as SQL NUMERIC where it holds an int or a float, else NULL.

    The type check guards the cast, which would fail on a value of another
    type, in a CASE, whose branches are evaluated only as taken.
    """
    is_number = (
        column[PROPERTY_TYPE_KEY]
        .as_string()
        .in_(
            (
                PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[int],
                PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[float],
            )
        )
    )
    return case((is_number, cast(column[PROPERTY_VALUE_KEY].as_string(), Numeric)))


def _number_literal(value: float) -> ColumnElement:
    """A numeric literal as SQL NUMERIC, written in the digits that name it exactly.

    A float's shortest round-trip digits, the ones its JSON holds, order
    like the float itself, so NUMERIC compares it with an int or another
    float exactly; a float bound as a double would round a held integer
    beyond 2**53 to compare it.
    """
    return cast(literal(repr(value)), Numeric)


def compile_sql_filter(
    expr: FilterExpr,
    resolve_field: FieldResolver,
) -> ColumnElement[bool]:
    """
    Compile a FilterExpr tree into a SQLAlchemy boolean expression.

    The `resolve_field` callback maps each field to a
    `(column, FieldEncoding)` pair and raises `ValueError` for unknown fields.

    Datetime values arrive already normalized -- a node converts them to
    UTC-aware instants at construction -- so column-encoded leaves bind them
    as-is, and the JSON-text encodings only choose a representation.

    A leaf whose field holds no value compares to SQL NULL, so `Not` makes
    its operand total with `COALESCE(..., FALSE)` before negating it and is
    the complement of a match, keeping rows that hold no value.
    """
    match expr:
        case Equals() | Ordering() | In() | IsNull():
            column, kind = resolve_field(expr.field)
            match kind:
                case "column":
                    return _compile_column_leaf(expr, column)
                case "uuid":
                    return _compile_uuid_leaf(expr, column)
                case "json":
                    return _compile_json_leaf(expr, column)
                case "properties_json":
                    return _compile_properties_json_leaf(expr, column)
        case And(operands):
            return and_(*(compile_sql_filter(o, resolve_field) for o in operands))
        case Or(operands):
            return or_(*(compile_sql_filter(o, resolve_field) for o in operands))
        case Not(operand):
            inner = compile_sql_filter(operand, resolve_field)
            return ~func.coalesce(inner, false())
