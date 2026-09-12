"""
EventMemory helpers: its system fields as filters.

The reserved keys of the system fields, and the translation between the
typed filters over them (`since`, `until`, `session_ids`, `source_ids`,
`block_kinds`) and the filter tree `system_predicates` builds from them.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Final

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
)
from memmachine_server.common.property_keys import (
    reserved_property_key,
)

# Built rather than written out so the alphabet and length budget of the
# vector store's naming contract are checked at import time.
EVENT_TIMESTAMP_KEY: Final[str] = reserved_property_key("event", "timestamp")
EVENT_SESSION_KEY: Final[str] = reserved_property_key("event", "session")
EVENT_SOURCE_KEY: Final[str] = reserved_property_key("event", "source")
BLOCK_KIND_KEY: Final[str] = reserved_property_key("block", "kind")


def conjoin(clauses: Iterable[FilterExpr | None]) -> FilterExpr | None:
    """The conjunction of the given clauses; None when there are none."""
    combined: FilterExpr | None = None
    for clause in clauses:
        if clause is None:
            continue
        combined = clause if combined is None else And(left=combined, right=clause)
    return combined


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
    overlap. A list admits its members and nothing else, so an empty list
    admits nothing; a list left `None` admits everything.
    """
    clauses: list[FilterExpr | None] = [
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=since)
        if since is not None
        else None,
        Comparison(field=EVENT_TIMESTAMP_KEY, op="<", value=until)
        if until is not None
        else None,
        In(field=EVENT_SESSION_KEY, values=list(session_ids))
        if session_ids is not None
        else None,
        In(field=EVENT_SOURCE_KEY, values=list(source_ids))
        if source_ids is not None
        else None,
        In(field=BLOCK_KIND_KEY, values=list(block_kinds))
        if block_kinds is not None
        else None,
    ]
    return conjoin(clauses)
