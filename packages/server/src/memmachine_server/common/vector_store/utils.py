"""Shared utilities for vector store implementations."""

import re

_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]+$")
_IDENTIFIER_MAX_BYTES = 32


def validate_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.match(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )
