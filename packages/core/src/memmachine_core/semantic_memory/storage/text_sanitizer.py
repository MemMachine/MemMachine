"""Sanitize text values before PostgreSQL insertion."""

import logging

logger = logging.getLogger(__name__)


def sanitize_pg_text(value: str, *, context: str = "") -> str:
    """Strip characters that PostgreSQL TEXT columns cannot store.

    PostgreSQL rejects null bytes (0x00) in TEXT/VARCHAR columns. This
    function removes them and logs a warning so the data-quality issue
    is visible for monitoring.
    """
    if "\x00" in value:
        logger.warning(
            "Stripped null byte(s) from text value%s: %.200r",
            f" ({context})" if context else "",
            value,
        )
        return value.replace("\x00", "")
    return value
