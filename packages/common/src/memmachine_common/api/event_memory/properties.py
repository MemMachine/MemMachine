"""
Type-tagged JSON property encoding and decoding.

Duplicates memmachine_server.common.properties_json so that
memmachine-client can deserialize properties without depending
on memmachine-server.

Properties are stored as::

    {"field_name": {"t": type_name, "v": value}}

Datetimes additionally include a timezone offset::

    {"field_name": {"t": "datetime", "v": utc_iso_value, "tz": offset_seconds}}
"""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

PropertyValue = bool | int | float | str | datetime
"""Type for stored property values."""

PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME: Final[dict[type[PropertyValue], str]] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    datetime: "datetime",
}

PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE: Final[dict[str, type[PropertyValue]]] = {
    v: k for k, v in PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME.items()
}

PROPERTY_TYPE_KEY = "t"
PROPERTY_VALUE_KEY = "v"
PROPERTY_TIMEZONE_OFFSET_KEY = "tz"

_EXPECTED_KEYS = frozenset({PROPERTY_TYPE_KEY, PROPERTY_VALUE_KEY})
_EXPECTED_DATETIME_KEYS = frozenset(
    {PROPERTY_TYPE_KEY, PROPERTY_VALUE_KEY, PROPERTY_TIMEZONE_OFFSET_KEY}
)


def _ensure_tz_aware(dt: datetime) -> datetime:
    """Return an aware datetime; treat naive datetimes as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _utc_offset_seconds(dt: datetime) -> int:
    """Return the UTC offset in seconds, treating naive datetimes as UTC."""
    offset = dt.utcoffset()
    return int(offset.total_seconds()) if offset is not None else 0


def encode_properties(
    properties: Mapping[str, PropertyValue] | None,
) -> dict[str, dict[str, bool | int | float | str]]:
    """Encode properties as type-tagged JSON."""
    if not properties:
        return {}
    encoded: dict[str, dict[str, bool | int | float | str]] = {}
    for key, value in properties.items():
        type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME.get(type(value))
        if type_name is None:
            raise ValueError(f"Unsupported property value type: {type(value)!r}")
        if isinstance(value, datetime):
            aware_value = _ensure_tz_aware(value)
            encoded[key] = {
                PROPERTY_VALUE_KEY: aware_value.astimezone(timezone.utc).isoformat(),
                PROPERTY_TYPE_KEY: type_name,
                PROPERTY_TIMEZONE_OFFSET_KEY: _utc_offset_seconds(value),
            }
        else:
            encoded[key] = {
                PROPERTY_VALUE_KEY: value,
                PROPERTY_TYPE_KEY: type_name,
            }
    return encoded


def decode_properties(
    encoded: Mapping | None,
) -> dict[str, PropertyValue]:
    """Decode type-tagged JSON properties back to Python values."""
    if not encoded:
        return {}

    properties: dict[str, PropertyValue] = {}
    for key, entry in encoded.items():
        if not isinstance(key, str):
            raise TypeError(f"Property key must be str, got {type(key).__name__}")
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"Property {key!r} value must be Mapping, got {type(entry).__name__}"
            )
        properties[key] = _decode_entry(key, entry)
    return properties


def _decode_entry(key: str, entry: Mapping) -> PropertyValue:
    """Decode a single type-tagged property entry."""
    type_name = entry.get(PROPERTY_TYPE_KEY)
    if not isinstance(type_name, str):
        raise TypeError(
            f"Property {key!r} entry {PROPERTY_TYPE_KEY!r} must be str, "
            f"got {type(type_name).__name__}"
        )

    raw_value = entry.get(PROPERTY_VALUE_KEY)
    if raw_value is None:
        raise ValueError(f"Property {key!r} entry missing {PROPERTY_VALUE_KEY!r}")

    prop_type = PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE.get(type_name)
    if prop_type is None:
        raise ValueError(f"Property {key!r} unknown type name: {type_name!r}")

    if prop_type is datetime:
        if set(entry.keys()) != _EXPECTED_DATETIME_KEYS:
            raise ValueError(
                f"Property {key!r} datetime entry must have exactly keys "
                f"{_EXPECTED_DATETIME_KEYS}"
            )
        tz_offset = entry[PROPERTY_TIMEZONE_OFFSET_KEY]
        if not isinstance(tz_offset, int):
            raise TypeError(
                f"Property {key!r} entry {PROPERTY_TIMEZONE_OFFSET_KEY!r} must be int, "
                f"got {type(tz_offset).__name__}"
            )
        utc_dt = datetime.fromisoformat(str(raw_value))
        original_tz = timezone(timedelta(seconds=tz_offset))
        return _ensure_tz_aware(utc_dt).astimezone(original_tz)

    if set(entry.keys()) != _EXPECTED_KEYS:
        raise ValueError(
            f"Property {key!r} entry must have exactly keys {_EXPECTED_KEYS}"
        )

    if not isinstance(raw_value, (bool, int, float, str)):
        raise TypeError(
            f"Property {key!r} entry {PROPERTY_VALUE_KEY!r} must be non-null scalar, "
            f"got {type(raw_value).__name__}"
        )
    if not isinstance(raw_value, prop_type):
        raise TypeError(
            f"Property {key!r} entry {PROPERTY_VALUE_KEY!r} must be {prop_type.__name__}, "
            f"got {type(raw_value).__name__}"
        )
    return raw_value
