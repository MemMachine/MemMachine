"""
System fields of EventMemory as filters.

EventMemory writes its system fields into every vector record under reserved
property keys, and a search bounds them with typed parameters (`since`,
`until`, `session_ids`, `source_ids`, `block_kinds`) beside the caller's
property filter. This module owns the translation between the two forms:
`system_predicates` builds the tree a vector store gets from the typed values,
and `split_system` recovers the typed values from a tree that names the
reserved keys. Nothing else in EventMemory knows the keys' spelling.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Final

from pydantic import BaseModel

from memmachine_server.common.filter import (
    Equals,
    FilterExpr,
    In,
    Ordering,
    conjoin,
    conjuncts,
    filter_fields,
)
from memmachine_server.common.property_keys import (
    is_reserved_property_key,
    reserved_property_key,
)

# Built rather than written out so the alphabet and length budget of the
# vector store's naming contract are checked at import time.
EVENT_TIMESTAMP_KEY: Final[str] = reserved_property_key("event", "timestamp")
EVENT_SESSION_KEY: Final[str] = reserved_property_key("event", "session")
EVENT_SOURCE_KEY: Final[str] = reserved_property_key("event", "source")
BLOCK_KIND_KEY: Final[str] = reserved_property_key("block", "kind")


class SystemFilters(BaseModel):
    """The typed system filters of a search or an expansion."""

    since: datetime | None = None
    """Inclusive lower bound on the event timestamp."""
    until: datetime | None = None
    """Exclusive upper bound on the event timestamp."""
    session_ids: list[str] | None = None
    source_ids: list[str] | None = None
    block_kinds: list[str] | None = None


def system_predicates(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    session_ids: Iterable[str] | None = None,
    source_ids: Iterable[str] | None = None,
    block_kinds: Iterable[str] | None = None,
) -> FilterExpr | None:
    """The predicates on reserved keys that a vector store evaluates.

    `since` is inclusive and `until` exclusive, so ranges meet without
    overlap. None admits everything. An empty id or kind list admits
    nothing, which is not a predicate: the caller answers that without a
    query, and passing one here raises ValueError.
    """
    clauses: list[FilterExpr | None] = [
        Ordering(EVENT_TIMESTAMP_KEY, ">=", since) if since is not None else None,
        Ordering(EVENT_TIMESTAMP_KEY, "<", until) if until is not None else None,
        _in_ids(EVENT_SESSION_KEY, session_ids),
        _in_ids(EVENT_SOURCE_KEY, source_ids),
        _in_ids(BLOCK_KIND_KEY, block_kinds),
    ]
    return conjoin(clauses)


def _in_ids(key: str, ids: Iterable[str] | None) -> In | None:
    if ids is None:
        return None
    values = tuple(ids)
    if not values:
        raise ValueError(f"An empty {key} list admits nothing and is not a predicate")
    return In(key, values)


def split_system(expr: FilterExpr | None) -> tuple[SystemFilters, FilterExpr | None]:
    """Pull the conjuncts naming reserved keys out of a tree into typed values.

    The inverse of `system_predicates`, for an API boundary that lets a
    caller write a system field inside a filter: the returned tree names no
    reserved key and is what the caller's `property_filter` becomes. A
    reserved key is accepted only as a top-level conjunct in the shape
    `system_predicates` builds (a `>=` or `<` on the timestamp, an `Equals`
    or an `In` on the session, source or block kind); repeated conjuncts on
    one field intersect. Anywhere else, or on any other reserved key, it
    raises ValueError, since no typed value could carry it.
    """
    filters = SystemFilters()
    rest: list[FilterExpr] = []
    for conjunct in conjuncts(expr):
        if not any(is_reserved_property_key(f) for f in filter_fields(conjunct)):
            rest.append(conjunct)
            continue
        _absorb(filters, conjunct)
    return filters, conjoin(rest)


def _absorb(filters: SystemFilters, conjunct: FilterExpr) -> None:
    match conjunct:
        case Ordering(field, op, value) if field == EVENT_TIMESTAMP_KEY:
            if not isinstance(value, datetime):
                raise TypeError(f"{EVENT_TIMESTAMP_KEY} compares with a datetime")
            if op == ">=":
                filters.since = _later(filters.since, value)
            elif op == "<":
                filters.until = _earlier(filters.until, value)
            else:
                raise ValueError(
                    f"{EVENT_TIMESTAMP_KEY} takes only >= (since) and < (until), "
                    f"got {op!r}"
                )
        case Equals(field, value) if isinstance(value, str):
            _absorb_ids(filters, field, [value])
        case In(field, values) if all(isinstance(value, str) for value in values):
            _absorb_ids(filters, field, [str(value) for value in values])
        case _:
            raise ValueError(
                f"A reserved key is accepted only as a top-level Equals or In "
                f"conjunct over strings, or >= and < on {EVENT_TIMESTAMP_KEY}; "
                f"got {conjunct!r}"
            )


def _absorb_ids(filters: SystemFilters, field: str, ids: list[str]) -> None:
    if field == EVENT_SESSION_KEY:
        filters.session_ids = _intersect(filters.session_ids, ids)
    elif field == EVENT_SOURCE_KEY:
        filters.source_ids = _intersect(filters.source_ids, ids)
    elif field == BLOCK_KIND_KEY:
        filters.block_kinds = _intersect(filters.block_kinds, ids)
    else:
        raise ValueError(f"{field!r} is not a filterable system field")


def _later(current: datetime | None, bound: datetime) -> datetime:
    return bound if current is None else max(current, bound)


def _earlier(current: datetime | None, bound: datetime) -> datetime:
    return bound if current is None else min(current, bound)


def _intersect(current: list[str] | None, ids: list[str]) -> list[str]:
    if current is None:
        return ids
    return [value for value in current if value in ids]
