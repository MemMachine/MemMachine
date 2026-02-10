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
