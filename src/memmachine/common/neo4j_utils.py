"""Common helpers for working with Neo4j values and comparisons."""

from __future__ import annotations

import datetime as _dt
from typing import TypeVar

from neo4j.time import DateTime as _Neo4jDateTime

from memmachine.common.data_types import FilterablePropertyValue
from memmachine.common.vector_graph_store.data_types import PropertyValue

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


def value_from_neo4j(value: PropertyValue) -> PropertyValue:
    """Convert Neo4j driver values into native Python equivalents."""
    if isinstance(value, _Neo4jDateTime):
        return value.to_native()
    return value


def render_temporal_comparison(
    left: str,
    op: str,
    right: str,
    value: FilterablePropertyValue | list[FilterablePropertyValue],
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
    value: FilterablePropertyValue,
) -> FilterablePropertyValue:
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


__all__ = [
    "coerce_datetime_to_timestamp",
    "render_temporal_comparison",
    "sanitize_value_for_neo4j",
    "value_from_neo4j",
]
