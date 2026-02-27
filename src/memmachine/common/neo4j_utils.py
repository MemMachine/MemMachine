"""Common helpers for working with Neo4j values and comparisons."""

from __future__ import annotations

import datetime as _dt
import re as _re
from typing import TypeVar

from neo4j.time import DateTime as _Neo4jDateTime

from memmachine.common.data_types import FilterValue, PropertyValue
from memmachine.common.vector_graph_store.data_types import (
    PropertyValue as VGSPropertyValue,
)

TScalar = TypeVar("TScalar", bound=object)
Neo4jSanitizedValue = TScalar | list[TScalar]


def sanitize_value_for_neo4j(value: Neo4jSanitizedValue) -> Neo4jSanitizedValue:
    """Normalize Python values before sending them to Neo4j."""
    if isinstance(value, _dt.datetime):
        tzinfo = value.tzinfo
        if tzinfo is None:
            tz = _dt.UTC
            return value.replace(tzinfo=tz)
        utc_offset = value.utcoffset()
        tz = _dt.timezone(utc_offset) if utc_offset is not None else tzinfo
        return value.astimezone(tz)
    if isinstance(value, list):
        return [sanitize_value_for_neo4j(item) for item in value]
    return value


def value_from_neo4j(value: VGSPropertyValue) -> VGSPropertyValue:
    """Convert Neo4j driver values into native Python equivalents."""
    if isinstance(value, _Neo4jDateTime):
        return value.to_native()
    return value


def render_comparison(
    left: str,
    op: str,
    right: str,
    value: PropertyValue,
) -> str:
    """Render a Cypher comparison clause that is safe for temporal values."""
    if op == "!=":
        op = "<>"
    if isinstance(value, list):
        raise TypeError(f"'{op}' comparison cannot accept list values")
    if isinstance(value, _dt.datetime):
        if op == "=":
            return (
                "("
                f"{left} = {right}"
                " OR "
                "("
                f"{left}.epochSeconds = {right}.epochSeconds"
                " AND "
                f"{left}.nanosecond = {right}.nanosecond"
                ")"
                ")"
            )

        if op == "<>":
            return (
                "("
                f"{left} <> {right}"
                " AND "
                "("
                f"{left}.epochSeconds <> {right}.epochSeconds"
                " OR "
                f"{left}.nanosecond <> {right}.nanosecond"
                ")"
                ")"
            )

        return (
            "("
            f"{left} {op} {right}"
            " AND "
            "("
            f"{left}.epochSeconds {op} {right}.epochSeconds"
            " OR "
            "("
            f"{left}.epochSeconds = {right}.epochSeconds"
            " AND "
            f"{left}.nanosecond {op} {right}.nanosecond"
            ")"
            ")"
            ")"
        )

    return f"{left} {op} {right}"


def coerce_datetime_to_timestamp(
    value: FilterValue,
) -> FilterValue:
    """Convert filter values into epoch timestamps when appropriate."""
    if isinstance(value, _dt.datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            parsed = _dt.datetime.fromisoformat(value)
        except ValueError:
            return value
        return parsed.timestamp()
    return value


# ---------------------------------------------------------------------------
# Entity type label helpers
# ---------------------------------------------------------------------------

ENTITY_TYPE_PREFIX = "ENTITY_TYPE_"


def sanitize_entity_type(name: str) -> str:
    """Sanitize an entity type name into a safe Neo4j label.

    Uses the ``ENTITY_TYPE_`` prefix (distinct from ``SANITIZED_`` used for
    collection/property names) so that entity type labels can be reliably
    distinguished from other labels when reading nodes back from Neo4j.
    """
    return ENTITY_TYPE_PREFIX + "".join(
        c if c.isalnum() else f"_u{ord(c):x}_" for c in name
    )


def desanitize_entity_type(sanitized_name: str) -> str:
    """Restore an entity type label to its original name."""
    return _re.sub(
        r"_u([0-9a-fA-F]+)_",
        lambda match: chr(int(match[1], 16)),
        sanitized_name.removeprefix(ENTITY_TYPE_PREFIX),
    )


__all__ = [
    "ENTITY_TYPE_PREFIX",
    "coerce_datetime_to_timestamp",
    "desanitize_entity_type",
    "render_comparison",
    "sanitize_entity_type",
    "sanitize_value_for_neo4j",
    "value_from_neo4j",
]
