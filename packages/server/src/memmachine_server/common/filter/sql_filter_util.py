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
from uuid import UUID

from sqlalchemy import ColumnElement, Numeric, and_, case, false, or_

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

FieldEncoding = Literal["column", "json", "properties_json", "uuid"]

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


def _compile_uuid_leaf(
    expr: IsNull | In | Comparison,
    column: ColumnElement,
) -> ColumnElement[bool]:
    if isinstance(expr, IsNull):
        return column.is_(None)
    if isinstance(expr, In):
        if not expr.values:
            return false()
        return column.in_([UUID(str(value)) for value in expr.values])
    return _get_op(expr.op)(column, UUID(str(expr.value)))


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
    dialect_name: str | None = None,
) -> ColumnElement[bool]:
    if isinstance(expr, IsNull):
        # .as_string() emits ->> instead of JSON_QUOTE(JSON_EXTRACT(...)),
        # which preserves SQL NULL for missing keys on SQLite.
        return column.as_string().is_(None)
    if isinstance(expr, In):
        if not expr.values:
            return false()
        value = _check_json_value(expr.values[0])
        casted = _numeric_json_clause(column, value, dialect_name)
        if casted is not None:
            return casted.in_(expr.values)
        return _cast_json_value(column, value).in_(expr.values)
    value = _check_json_value(expr.value)
    casted = _numeric_json_clause(column, value, dialect_name)
    if casted is not None:
        return _get_op(expr.op)(casted, expr.value)
    return _get_op(expr.op)(_cast_json_value(column, value), expr.value)


# Numeric text a PostgreSQL numeric cast accepts, so every row the CASE
# guard passes to the cast parses and the query cannot raise (#1917).
_NUMERIC_TEXT_PATTERN = r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$"


def _numeric_json_clause(
    column: ColumnElement,
    value: bool | float | str,
    dialect_name: str | None,
) -> ColumnElement | None:
    """Cast a raw JSON path for a numeric literal, matching numeric strings.

    The REST API stores every metadata value as a string, so the documented
    filter `metadata.user_id=123` must still find a stored "123" (issue
    #1917). PostgreSQL raises for the whole query when a cast over a text
    value does not parse, so the cast is guarded by a CASE on a numeric
    pattern instead of the stored type; JSON numbers reach it as their own
    text and are admitted too. SQLite casts never raise, and non-numeric
    text casts to 0. Unknown dialects keep the unguarded cast.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    if dialect_name == "postgresql":
        text = column.as_string()
        return case((text.regexp_match(_NUMERIC_TEXT_PATTERN), text.cast(Numeric)))
    if dialect_name == "sqlite":
        return column.as_string().cast(Numeric)
    return None


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
        # One cast-and-normalize rule for both leaf shapes: every member
        # is cast and normalized the way a Comparison value would be, and
        # the column takes the first member's cast.
        casts = [
            _cast_properties_json_value(value_path, value) for value in expr.values
        ]
        casted_column = casts[0][0]
        normalized_values = [normalized for _, normalized in casts]
        return and_(type_check, casted_column.in_(normalized_values))

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
    dialect_name: str | None = None,
) -> ColumnElement[bool]:
    """
    Compile a FilterExpr tree into a SQLAlchemy boolean expression.

    The `resolve_field` callback maps each field name to a
    `(column, FieldEncoding)` pair and raises `ValueError` for unknown fields.

    `dialect_name` (e.g. "postgresql", "sqlite") enables dialect-specific
    guards for raw JSON ("json") fields; when it is None those fields keep
    the ungated behavior.

    Datetime values arrive already normalized -- `Comparison`/`In` nodes
    convert them to UTC-aware instants at construction -- so
    column-encoded leaves bind them as-is, and the JSON-text encodings
    only choose a representation.
    """
    if isinstance(expr, Comparison | In | IsNull):
        column, kind = resolve_field(expr.field)
        if kind == "column":
            return _compile_column_leaf(expr, column)
        if kind == "uuid":
            return _compile_uuid_leaf(expr, column)
        if kind == "json":
            return _compile_json_leaf(expr, column, dialect_name)
        if kind == "properties_json":
            return _compile_properties_json_leaf(expr, column)
        raise ValueError(f"Unknown field kind: {kind!r}")

    if isinstance(expr, And):
        return and_(
            compile_sql_filter(expr.left, resolve_field, dialect_name),
            compile_sql_filter(expr.right, resolve_field, dialect_name),
        )

    if isinstance(expr, Or):
        return or_(
            compile_sql_filter(expr.left, resolve_field, dialect_name),
            compile_sql_filter(expr.right, resolve_field, dialect_name),
        )

    if isinstance(expr, Not):
        return ~compile_sql_filter(expr.expr, resolve_field, dialect_name)

    raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
