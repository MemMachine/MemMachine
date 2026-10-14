"""Enforcement of a store's declared schema at the write and query boundaries."""

import math
from collections.abc import Iterable, Mapping

from memmachine_server.common.data_types import PropertyType, PropertyValue
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
    filter_fields,
    filter_nodes,
)

from .data_types import (
    MAX_INT_VALUE,
    MIN_INT_VALUE,
    PropertyTypeMismatchError,
    Record,
    UndeclaredPropertyKeyError,
    UnsupportedFilterError,
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
    properties = {
        key: _bound_value(key, value, indexed_properties[key])
        for key, value in record.properties.items()
    }
    return record.model_copy(update={"properties": properties})


def bind_filter(
    property_filter: FilterExpr,
    indexed_properties: Mapping[str, PropertyType],
    supported_filter_nodes: Iterable[type],
) -> FilterExpr:
    """
    Bind a filter to the declared schema: the filter a store compiles in its place.

    Raises unless the filter names declared keys only, compares each key with
    values of its declared type, and, bound, uses supported nodes only. An
    undeclared key never exists in the store, so a filter naming one has
    nothing to scan for; a node outside `supported_filter_nodes` is one the
    backend cannot evaluate during its search.

    In the bound filter every value is of its key's declared type, so a
    backend compares a stored value only with a value of the same type. An
    int for a float key binds to the float it equals, and a membership test
    of ints on a float key to the disjunction of those floats' equalities,
    since a membership test holds ints or strs. A bool is never a number, a
    float key is compared with finite floats only, and any key with ints in
    64 signed bits only.
    """
    undeclared = filter_fields(property_filter) - indexed_properties.keys()
    if undeclared:
        raise UndeclaredPropertyKeyError(undeclared, indexed_properties.keys())
    bound = _bound_filter(property_filter, indexed_properties)
    supported = frozenset(supported_filter_nodes)
    unsupported = filter_nodes(bound) - supported
    if unsupported:
        raise UnsupportedFilterError(unsupported, supported)
    return bound


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
        case IsNull():
            return expr
        case Equals(field, value):
            return Equals(field, _bound_value(field, value, indexed_properties[field]))
        case Ordering(field, op, value):
            return Ordering(
                field, op, _bound_value(field, value, indexed_properties[field])
            )
        case In(field, values):
            declared = indexed_properties[field]
            for value in values:
                _require_value(field, value, declared)
            if declared is float and values:
                return Or(tuple(Equals(field, float(value)) for value in values))
            return expr


def _bound_value[V: PropertyValue](
    key: str, value: V, declared: PropertyType
) -> V | float:
    """The value as its key's declared type holds it."""
    _require_value(key, value, declared)
    if declared is float and isinstance(value, int):
        return float(value)
    return value


def _require_value(key: str, value: PropertyValue, declared: PropertyType) -> None:
    """Raise unless the value is of its key's declared type, an int counting as a float."""
    # `bool` is an `int` at runtime and is its own property type here.
    if type(value) is int and not MIN_INT_VALUE <= value <= MAX_INT_VALUE:
        raise PropertyTypeMismatchError(key, declared, value)
    if declared is float and type(value) is int:
        return
    if type(value) is not declared or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        raise PropertyTypeMismatchError(key, declared, value)
