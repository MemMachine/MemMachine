"""Shared utilities for vector store implementations."""

import re

# Matched with fullmatch: `$` also matches before a trailing newline.
_IDENTIFIER_RE = re.compile(r"[a-z0-9_]+")
_IDENTIFIER_MAX_BYTES = 32


def validate_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.fullmatch(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )


def require_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the partition key is a valid identifier."""
    if not validate_identifier(partition_key):
        raise ValueError(
            f"Partition key {partition_key!r} must match [a-z0-9_]+ and be at most "
            "32 bytes"
        )
