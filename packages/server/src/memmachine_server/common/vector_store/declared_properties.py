"""Enforcement of a store's declared property types at the write boundary."""

from collections.abc import Mapping

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)


def require_declared_types(
    properties: Mapping[str, PropertyValue],
    indexed_properties: Mapping[str, type[PropertyValue]],
) -> None:
    """Raise unless every declared property holds a value of its declared type."""
    for key, value in properties.items():
        declared_type = indexed_properties.get(key)
        # `bool` is an `int` at runtime and is its own property type here.
        if declared_type is not None and type(value) is not declared_type:
            raise ValueError(
                f"Property {key!r} is declared as "
                f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared_type]}, "
                f"got {type(value).__name__} {value!r}."
            )
