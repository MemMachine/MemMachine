"""Shared utilities for vector store implementations."""

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
    PropertyValue,
)
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
    Property,
    SystemField,
    filter_fields,
)

from .data_types import (
    MAX_INT_VALUE,
    MIN_INT_VALUE,
    PropertyTypeMismatchError,
    Record,
    UndeclaredPropertyKeyError,
)

_PARTITION_KEY_MAX_BYTES = 255


def validate_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the partition key is non-empty, at most `_PARTITION_KEY_MAX_BYTES` bytes, and free of NUL."""
    if (
        not partition_key
        or "\x00" in partition_key
        or len(partition_key.encode()) > _PARTITION_KEY_MAX_BYTES
    ):
        raise ValueError(
            f"Partition key {partition_key!r} must be non-empty, at most "
            f"{_PARTITION_KEY_MAX_BYTES} bytes, and free of NUL characters"
        )


def bind_record(
    record: Record, indexed_properties: Mapping[str, PropertyType]
) -> Record:
    """
    Bind a record to the declared schema: the record a store writes in its place.

    Raises unless every property is declared and holds a value of its type.
    Called before anything is sent, so an undeclared key never reaches a
    backend, and a declared key never holds a value its column or index
    would have to coerce: an int for a float key binds to the float it
    equals, a bool is never a number, a float is finite, and an int fits in
    64 signed bits.
    """
    undeclared = record.properties.keys() - indexed_properties.keys()
    if undeclared:
        raise UndeclaredPropertyKeyError(undeclared, indexed_properties.keys())
    properties = bind_properties(record.properties, indexed_properties)
    return record.model_copy(update={"properties": properties})


def bind_properties(
    properties: Mapping[str, PropertyValue],
    indexed_properties: Mapping[str, PropertyType],
) -> dict[str, PropertyValue]:
    """
    Bind properties to the declared schema: each declared key's value as its type holds it.

    Raises PropertyTypeMismatchError unless each declared key holds a value
    of its type, by the rules `bind_record` states; a key the schema does
    not declare keeps its value.
    """
    return {
        key: (
            _bound_value(key, value, indexed_properties[key])
            if key in indexed_properties
            else value
        )
        for key, value in properties.items()
    }


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


def bind_filter(
    property_filter: FilterExpr,
    indexed_properties: Mapping[str, PropertyType],
) -> FilterExpr:
    """
    Bind a filter to the declared schema: the filter a store compiles in its place.

    Raises unless the filter names declared keys only and compares each key
    with values of its declared type. An undeclared key never exists in the
    store, so a filter naming one has nothing to scan for. A membership test
    holds ints or strs, so one on a float or datetime key raises ValueError.

    In the bound filter every property carries its key's declared type and
    every value is of that type, so a backend compares a stored value only
    with a value of the same type. An int for a float key binds to the float
    it equals. A bool is never a number, a float key is compared with finite
    floats only, and any key with ints in 64 signed bits only.
    """
    keys = {property_key(field) for field in filter_fields(property_filter)}
    undeclared = keys - indexed_properties.keys()
    if undeclared:
        raise UndeclaredPropertyKeyError(undeclared, indexed_properties.keys())
    return _bound_filter(property_filter, indexed_properties)


def property_key(field: SystemField | Property) -> str:
    """The key a filter's field names; raises ValueError for a system field, since a vector store defines none."""
    match field:
        case Property(key=key):
            return key
        case SystemField(name=name):
            raise ValueError(
                f"A vector store filters on properties only, "
                f"not the system field {name!r}"
            )


def _bound_filter(
    expr: FilterExpr, indexed_properties: Mapping[str, PropertyType]
) -> FilterExpr:
    match expr:
        case And(operands):
            return And(tuple(_bound_filter(o, indexed_properties) for o in operands))
        case Or(operands):
            return Or(tuple(_bound_filter(o, indexed_properties) for o in operands))
        case Not(operand):
            return Not(_bound_filter(operand, indexed_properties))
        case IsNull(field):
            key = property_key(field)
            return IsNull(Property(key, indexed_properties[key]))
        case Equals(field, value):
            key = property_key(field)
            declared = indexed_properties[key]
            return Equals(Property(key, declared), _bound_value(key, value, declared))
        case Ordering(field, op, value):
            key = property_key(field)
            declared = indexed_properties[key]
            return Ordering(
                Property(key, declared), op, _bound_value(key, value, declared)
            )
        case In(field, values):
            key = property_key(field)
            declared = indexed_properties[key]
            if declared is float or declared is datetime:
                raise ValueError(
                    f"Property {key!r} is declared as "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared]}, "
                    "which a membership test does not compare"
                )
            for value in values:
                _validate_value(key, value, declared)
            return In(Property(key, declared), values)


def _bound_value[V: PropertyValue](
    key: str, value: V, declared: PropertyType
) -> V | float:
    """The value as its key's declared type holds it."""
    _validate_value(key, value, declared)
    if declared is float and isinstance(value, int):
        return float(value)
    return value


def _validate_value(key: str, value: PropertyValue, declared: PropertyType) -> None:
    """Raise unless the value is of its key's declared type, an int counting as a float."""
    # `bool` is an `int` at runtime and is its own property type here.
    if type(value) is int and not MIN_INT_VALUE <= value <= MAX_INT_VALUE:
        raise PropertyTypeMismatchError(
            key, declared, value, "outside the 64-bit signed range"
        )
    if declared is float and type(value) is int:
        return
    if type(value) is not declared:
        raise PropertyTypeMismatchError(key, declared, value)
    if isinstance(value, float) and not math.isfinite(value):
        raise PropertyTypeMismatchError(key, declared, value, "not finite")
