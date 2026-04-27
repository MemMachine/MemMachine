"""
Type-tagged JSON property encoding and decoding.

Properties are stored in a JSON column as::

    {"field_name": {"t": type_name, "v": value}}

Datetimes additionally include a timezone offset::

    {"field_name": {"t": "datetime", "v": utc_iso_value, "tz": offset_seconds}}
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from typing import cast

from pydantic import JsonValue

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds

PROPERTY_TYPE_KEY = "t"
PROPERTY_VALUE_KEY = "v"
PROPERTY_TIMEZONE_OFFSET_KEY = "tz"


def encode_properties(
    properties: Mapping[str, PropertyValue] | None,
) -> dict[str, dict[str, bool | int | float | str]]:
    """
    Encode properties as type-tagged JSON.

    `None` is treated as `{}`.
    Each property becomes `{"v": value, "t": type_name}`.
    Datetimes are normalized to UTC and include `{"tz": offset_seconds}`
    for timezone reconstruction.
    """
    if not properties:
        return {}
    encoded: dict[str, dict[str, bool | int | float | str]] = {}
    for key, value in properties.items():
        type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME.get(type(value))
        if type_name is None:
            raise ValueError(f"Unsupported property value type: {type(value)!r}")
        if isinstance(value, datetime):
            aware_value = ensure_tz_aware(value)
            encoded[key] = {
                PROPERTY_VALUE_KEY: aware_value.astimezone(UTC).isoformat(),
                PROPERTY_TYPE_KEY: type_name,
                PROPERTY_TIMEZONE_OFFSET_KEY: utc_offset_seconds(value),
            }
        else:
            encoded[key] = {
                PROPERTY_VALUE_KEY: value,
                PROPERTY_TYPE_KEY: type_name,
            }
    return encoded


def decode_properties(
    encoded: Mapping[str, JsonValue] | None,
) -> dict[str, PropertyValue]:
    """
    Decode type-tagged JSON properties back to Python values.

    Returns {} if `encoded` is `None` or empty.
    Reconstructs the original timezone for datetime values.
    """
    if not encoded:
        return {}
    properties: dict[str, PropertyValue] = {}
    for key, entry in encoded.items():
        if not isinstance(entry, dict):
            raise TypeError(f"Expected dict for property entry, got {type(entry)!r}")
        type_name = str(entry[PROPERTY_TYPE_KEY])
        raw_value = entry[PROPERTY_VALUE_KEY]
        property_type = PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE.get(type_name)
        if property_type is None:
            raise ValueError(f"Unknown property type name: {type_name!r}")
        if property_type is datetime:
            utc_dt = datetime.fromisoformat(str(raw_value))
            tz_offset = entry.get(PROPERTY_TIMEZONE_OFFSET_KEY, 0)
            original_tz = timezone(timedelta(seconds=int(tz_offset)))
            properties[key] = ensure_tz_aware(utc_dt).astimezone(original_tz)
        else:
            properties[key] = cast(type[bool | int | float | str], property_type)(
                raw_value
            )
    return properties
