"""Prompt utility functions for Memmachine server."""

import zoneinfo
from datetime import datetime


def current_date_dow(tz: str = "UTC") -> str:
    """Get the current date and day of the week in the specified timezone."""
    try:
        zone = zoneinfo.ZoneInfo(tz)
    except zoneinfo.ZoneInfoNotFoundError:
        zone = zoneinfo.ZoneInfo("UTC")
    dt = datetime.now(zone)
    return f"{dt.strftime('%Y-%m-%d')}[{dt.strftime('%a')}]"


def enum_list(enum_values: list[str]) -> str:
    """Format a list of strings as an enumerated list with quotes."""
    return ", ".join(f'"{v}"' for v in enum_values)


class InvalidMetaTagError(Exception):
    """Exception raised for invalid meta tag formats."""


def parse_raw_meta_tags(raw: str) -> dict[str, str]:
    """Parse raw meta tags from a string into a dictionary."""
    tags = {}
    kv_splitter = ":"
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if kv_splitter in line:
            key, value = line.split(kv_splitter, 1)
            tags[key.strip()] = value.strip()
        else:
            raise InvalidMetaTagError(
                f"Invalid meta tag line: {line}, missing '{kv_splitter}'"
            )
    return tags
