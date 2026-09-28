"""Shared utilities for segment store implementations."""

import re

# Matched with fullmatch: anchors would only suggest `$` does the work,
# and `$` matches before a trailing newline -- the bug fullmatch fixed.
_PARTITION_KEY_RE = re.compile(r"[a-z0-9_]+")

PARTITION_KEY_MAX_BYTES = 32
"""Maximum partition key length in bytes.

The charset is ASCII, so this equals the character count today; bytes is
the unit on purpose, because the budgets a key must fit (identifier and
column widths) are byte-denominated and stay honest if the charset ever
widens.
"""


def validate_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the key matches `[a-z0-9_]+` within PARTITION_KEY_MAX_BYTES."""
    if not partition_key:
        raise ValueError("Partition key must not be empty.")
    if not _PARTITION_KEY_RE.fullmatch(partition_key):
        raise ValueError(
            f"Partition key {partition_key!r} contains invalid characters. "
            "Only lowercase alphanumeric and underscores are allowed."
        )
    key_length_bytes = len(partition_key.encode())
    if key_length_bytes > PARTITION_KEY_MAX_BYTES:
        raise ValueError(
            f"Partition key {partition_key!r} is too long "
            f"({key_length_bytes} bytes). "
            f"Maximum is {PARTITION_KEY_MAX_BYTES}."
        )
