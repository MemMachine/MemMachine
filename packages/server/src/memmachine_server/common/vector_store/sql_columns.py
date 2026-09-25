"""
Declared properties as typed SQL columns, for the SQLite-backed stores.

A store's declared schema gives each partition's records table one typed
column per key, each indexed, so a filter is an indexed comparison rather
than a scan over serialized properties. Column names are prefixed so a
declared key can never collide with the table's own columns.

SQLite has no datetime type, so a datetime property is stored as an integer
of microseconds since the epoch, the precision the SQL stores keep, and a
bound compares as the same integer.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    Index,
    Table,
    Text,
    and_,
    false,
    or_,
)
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import PropertyType, PropertyValue
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.utils import ensure_tz_aware

PROPERTY_COLUMN_PREFIX = "p_"

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND = timedelta(microseconds=1)


def property_column_name(key: str) -> str:
    """The column holding a declared property."""
    return f"{PROPERTY_COLUMN_PREFIX}{key}"


def epoch_microseconds(value: datetime) -> int:
    """A datetime as microseconds since the epoch, exactly."""
    return (ensure_tz_aware(value) - _EPOCH) // _MICROSECOND


def property_columns(indexed_properties: Mapping[str, PropertyType]) -> list[Column]:
    """One nullable column per declared key, typed by the key's declared type."""
    columns: list[Column] = []
    for key, property_type in indexed_properties.items():
        if property_type is bool:
            column_type = Boolean()
        elif property_type is int or property_type is datetime:
            column_type = BigInteger()
        elif property_type is float:
            column_type = Float()
        else:
            column_type = Text()
        columns.append(Column(property_column_name(key), column_type, nullable=True))
    return columns


def property_indexes(
    table: Table, indexed_properties: Mapping[str, PropertyType]
) -> list[Index]:
    """One index per declared key, named after the table and the column."""
    return [
        Index(
            f"{table.name}__{property_column_name(key)}",
            table.c[property_column_name(key)],
        )
        for key in indexed_properties
    ]


def property_column_values(
    properties: Mapping[str, PropertyValue],
    indexed_properties: Mapping[str, PropertyType],
) -> dict[str, PropertyValue | None]:
    """The column values of a record's properties; an absent key is NULL."""
    values: dict[str, PropertyValue | None] = {
        property_column_name(key): None for key in indexed_properties
    }
    for key, value in properties.items():
        values[property_column_name(key)] = (
            epoch_microseconds(value) if isinstance(value, datetime) else value
        )
    return values


def compile_property_filter(
    expr: FilterExpr,
    table: Table,
    indexed_properties: Mapping[str, PropertyType],
) -> ColumnElement[bool]:
    """
    Compile a filter over the declared columns of a records table.

    A predicate matches only a value of the compared type, so a leaf whose
    value is not of its key's declared type matches nothing rather than
    whatever the database's affinity would coerce.
    """
    if isinstance(expr, (Comparison, In, IsNull)):
        return _compile_leaf(expr, table, indexed_properties)
    if isinstance(expr, And):
        return and_(
            compile_property_filter(expr.left, table, indexed_properties),
            compile_property_filter(expr.right, table, indexed_properties),
        )
    if isinstance(expr, Or):
        return or_(
            compile_property_filter(expr.left, table, indexed_properties),
            compile_property_filter(expr.right, table, indexed_properties),
        )
    if isinstance(expr, Not):
        return ~compile_property_filter(expr.expr, table, indexed_properties)
    raise TypeError(f"Unsupported filter expression type: {type(expr)}")


def _compare(column: ColumnElement, op: str, bound: object) -> ColumnElement[bool]:
    """The SQL comparison of a column against a bound, by the filter's operator."""
    match op:
        case "=":
            return column == bound
        case "!=":
            return column != bound
        case ">":
            return column > bound
        case "<":
            return column < bound
        case ">=":
            return column >= bound
        case "<=":
            return column <= bound
    raise ValueError(f"Unsupported comparison operator {op!r}")


def _compile_leaf(
    expr: Comparison | In | IsNull,
    table: Table,
    indexed_properties: Mapping[str, PropertyType],
) -> ColumnElement[bool]:
    column = table.c[property_column_name(expr.field)]
    declared = indexed_properties[expr.field]
    if isinstance(expr, IsNull):
        return column.is_(None)
    if isinstance(expr, In):
        return column.in_(expr.values) if type(expr.values[0]) is declared else false()
    if type(expr.value) is not declared:
        return false()
    bound = (
        epoch_microseconds(expr.value)
        if isinstance(expr.value, datetime)
        else expr.value
    )
    return _compare(column, expr.op, bound)
