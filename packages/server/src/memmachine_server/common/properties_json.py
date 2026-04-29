"""
Type-tagged JSON property encoding and decoding.

Properties are stored in a JSON column as::

    {"field_name": {"t": type_name, "v": value}}

Datetimes additionally include a timezone offset::

    {"field_name": {"t": "datetime", "v": utc_iso_value, "tz": offset_seconds}}
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds

PROPERTY_TYPE_KEY = "t"
PROPERTY_VALUE_KEY = "v"
PROPERTY_TIMEZONE_OFFSET_KEY = "tz"

_EXPECTED_KEYS = frozenset({PROPERTY_TYPE_KEY, PROPERTY_VALUE_KEY})
_EXPECTED_DATETIME_KEYS = frozenset(
    {PROPERTY_TYPE_KEY, PROPERTY_VALUE_KEY, PROPERTY_TIMEZONE_OFFSET_KEY}
)


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
    encoded: Mapping | None,
) -> dict[str, PropertyValue]:
    """
    Decode type-tagged JSON properties back to Python values.

    Returns {} if `encoded` is `None` or empty.
    Reconstructs the original timezone for datetime values.

    Each entry must be a dict with exactly keys {"t", "v"},
    or {"t", "v", "tz"} for datetimes.

    Raises TypeError for wrong types, ValueError for structural issues.
    """
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
            f"Property {key!r} entry {PROPERTY_TYPE_KEY!r} must be str, got {type(type_name).__name__}"
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
                f"Property {key!r} datetime entry must have exactly keys {_EXPECTED_DATETIME_KEYS}"
            )
        tz_offset = entry[PROPERTY_TIMEZONE_OFFSET_KEY]
        if not isinstance(tz_offset, int):
            raise TypeError(
                f"Property {key!r} entry {PROPERTY_TIMEZONE_OFFSET_KEY!r} must be int, got {type(tz_offset).__name__}"
            )
        utc_dt = datetime.fromisoformat(str(raw_value))
        original_tz = timezone(timedelta(seconds=tz_offset))
        return ensure_tz_aware(utc_dt).astimezone(original_tz)

    if set(entry.keys()) != _EXPECTED_KEYS:
        raise ValueError(
            f"Property {key!r} entry must have exactly keys {_EXPECTED_KEYS}"
        )

    if not isinstance(raw_value, (bool, int, float, str)):
        raise TypeError(
            f"Property {key!r} entry {PROPERTY_VALUE_KEY!r} must be non-null scalar, got {type(raw_value).__name__}"
        )
    if not isinstance(raw_value, prop_type):
        raise TypeError(
            f"Property {key!r} entry {PROPERTY_VALUE_KEY!r} must be {prop_type.__name__}, got {type(raw_value).__name__}"
        )
    return raw_value
