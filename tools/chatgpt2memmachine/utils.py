"""
Common utility functions for chatgpt2memmachine tool.

This module provides shared utility functions used across the tool.
"""

import os
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


def parse_run_id_line(line: str) -> Optional[tuple[int, str]]:
    """
    Parse a line from success/error file into (conv_id, message_id) tuple.
    
    Expected format: conv_id:message_id
    Lines that don't match this format are skipped.
    
    Args:
        line: Line from success/error file
        
    Returns:
        Tuple of (conv_id, message_id) or None if line doesn't match expected format
    """
    line = line.strip()
    if not line or line.startswith("Error:"):
        return None
    
    # Parse format: conv_id:message_id
    if ":" not in line:
        return None
    
    parts = line.split(":", 1)
    if len(parts) != 2:
        return None
    
    conv_id_str, msg_id = parts
    if not msg_id:  # message_id cannot be empty
        return None
    
    try:
        conv_id = int(conv_id_str)
        return (conv_id, msg_id)
    except ValueError:
        return None


def load_run_id_file(filepath: str) -> set[tuple[int, str]]:
    """
    Load message IDs from a run ID file (success or error file).
    
    Args:
        filepath: Path to the file to load
        
    Returns:
        Set of (conv_id, message_id) tuples
    """
    result = set()
    if not os.path.exists(filepath):
        return result
    
    with open(filepath, "r") as f:
        for line in f:
            parsed = parse_run_id_line(line)
            if parsed:
                result.add(parsed)
    
    return result
