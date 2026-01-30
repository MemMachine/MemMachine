"""
Common utility functions for chatgpt2memmachine tool.

This module provides shared utility functions used across the tool.
"""

import datetime
from typing import Optional
from datetime import timezone


def parse_time(time_str: str) -> Optional[float]:
    """
    Parse time string to timestamp.

    Supports:
    - Unix timestamp (integer or float)
    - ISO format: YYYY-MM-DDTHH:MM:SS
    - Date only: YYYY-MM-DD (assumes 00:00:00)

    Args:
        time_str: Time string to parse

    Returns:
        Timestamp as float, or None if parsing fails
    """
    if not time_str or time_str == "0":
        return None

    # Try as integer timestamp
    try:
        ts = int(time_str)
        if ts > 0:
            return float(ts)
    except ValueError:
        pass

    # Try as float timestamp
    try:
        ts = float(time_str)
        if ts > 0:
            return ts
    except ValueError:
        pass

    # Try ISO format with time
    try:
        time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
        return time_obj.timestamp()
    except ValueError:
        pass

    # Try ISO format with date only
    try:
        time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d")
        return time_obj.timestamp()
    except ValueError:
        pass

    return None


def format_timestamp_iso8601(timestamp: float | int) -> str:
    """
    Convert Unix timestamp to ISO 8601 format (UTC).

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        ISO 8601 formatted string (e.g., "2024-01-15T10:00:00.000Z")
    """
    dt = datetime.datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def get_filename_safe_timestamp() -> str:
    """
    Get a filename-safe timestamp string.

    Returns:
        Timestamp string in format YYYYMMDDTHHMMSSffffff (Windows-safe, no colons)
    """
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
