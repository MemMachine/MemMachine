"""Tests for the translation between typed system filters and filter trees."""

from datetime import UTC, datetime

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
)
from memmachine_server.episodic_memory.event_memory.utils import (
    BLOCK_KIND_KEY,
    EVENT_SESSION_KEY,
    EVENT_SOURCE_KEY,
    EVENT_TIMESTAMP_KEY,
    system_predicates,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 1, 2, tzinfo=UTC)


def _conjuncts(expr):
    if isinstance(expr, And):
        return [*_conjuncts(expr.left), *_conjuncts(expr.right)]
    return [expr]


def test_no_system_filter_is_no_tree():
    assert system_predicates() is None


def test_predicates_name_the_reserved_keys():
    tree = system_predicates(
        since=_T0,
        until=_T1,
        session_ids=["s1"],
        source_ids=["alice", "bob"],
        block_kinds=["text"],
    )
    assert _conjuncts(tree) == [
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T0),
        Comparison(field=EVENT_TIMESTAMP_KEY, op="<", value=_T1),
        In(field=EVENT_SESSION_KEY, values=["s1"]),
        In(field=EVENT_SOURCE_KEY, values=["alice", "bob"]),
        In(field=BLOCK_KIND_KEY, values=["text"]),
    ]


def test_empty_ids_admit_nothing_and_none_admits_everything():
    assert system_predicates(session_ids=None) is None
    assert system_predicates(session_ids=[]) == In(field=EVENT_SESSION_KEY, values=[])
