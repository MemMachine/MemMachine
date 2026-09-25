"""Enforcement of a store's declared schema at the write and query boundaries."""

from collections.abc import Iterable, Mapping

from memmachine_server.common.data_types import PropertyType, PropertyValue
from memmachine_server.common.filter import FilterExpr, filter_fields, filter_nodes

from .data_types import (
    PropertyTypeMismatchError,
    UndeclaredPropertyKeyError,
    UnsupportedFilterError,
)


def require_declared_properties(
    properties: Mapping[str, PropertyValue],
    indexed_properties: Mapping[str, PropertyType],
) -> None:
    """
    Raise unless every property is declared and holds a value of its type.

    Called before anything is sent, so an undeclared key never reaches a
    backend, and a declared key never holds a value its column or index
    would have to coerce.
    """
    undeclared = properties.keys() - indexed_properties.keys()
    if undeclared:
        raise UndeclaredPropertyKeyError(undeclared, indexed_properties.keys())
    for key, value in properties.items():
        # `bool` is an `int` at runtime and is its own property type here.
        if type(value) is not indexed_properties[key]:
            raise PropertyTypeMismatchError(key, indexed_properties[key], value)


def require_supported_filter(
    property_filter: FilterExpr,
    indexed_properties: Mapping[str, PropertyType],
    supported_filter_nodes: Iterable[type],
) -> None:
    """
    Raise unless a filter names declared keys only and uses supported nodes.

    An undeclared key never exists in the store, so a filter naming one has
    nothing to scan for; a node outside `supported_filter_nodes` is one the
    backend cannot evaluate during its search.
    """
    undeclared = filter_fields(property_filter) - indexed_properties.keys()
    if undeclared:
        raise UndeclaredPropertyKeyError(undeclared, indexed_properties.keys())
    supported = frozenset(supported_filter_nodes)
    unsupported = filter_nodes(property_filter) - supported
    if unsupported:
        raise UnsupportedFilterError(unsupported, supported)
