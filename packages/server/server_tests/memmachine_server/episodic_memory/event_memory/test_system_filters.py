"""Tests for the translation between typed system filters and filter trees."""

from datetime import UTC, datetime

import pytest

from memmachine_server.common.filter import (
    And,
    Equals,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
    conjoin,
    conjuncts,
)
from memmachine_server.common.property_keys import reserved_property_key
from memmachine_server.episodic_memory.event_memory.system_filters import (
    BLOCK_KIND_KEY,
    EVENT_SESSION_KEY,
    EVENT_SOURCE_KEY,
    EVENT_TIMESTAMP_KEY,
    SystemFilters,
    split_system,
    system_predicates,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 1, 2, tzinfo=UTC)


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
    assert conjuncts(tree) == [
        Ordering(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T0),
        Ordering(field=EVENT_TIMESTAMP_KEY, op="<", value=_T1),
        In(field=EVENT_SESSION_KEY, values=("s1",)),
        In(field=EVENT_SOURCE_KEY, values=("alice", "bob")),
        In(field=BLOCK_KIND_KEY, values=("text",)),
    ]


def test_none_admits_everything_and_an_empty_list_is_not_a_predicate():
    assert system_predicates(session_ids=None) is None
    # Admitting nothing needs no query, so it is the caller's answer, not a tree.
    with pytest.raises(ValueError, match="admits nothing"):
        system_predicates(session_ids=[])


def test_split_recovers_the_typed_values_and_leaves_the_rest():
    caller = Equals(field="color", value="red")
    tree = conjoin(
        [
            system_predicates(since=_T0, until=_T1, session_ids=["s1"]),
            caller,
            Equals(field=EVENT_SOURCE_KEY, value="alice"),
        ]
    )

    filters, rest = split_system(tree)

    assert filters == SystemFilters(
        since=_T0, until=_T1, session_ids=["s1"], source_ids=["alice"]
    )
    assert rest == caller


def test_split_of_nothing_is_nothing():
    assert split_system(None) == (SystemFilters(), None)


def test_split_intersects_repeated_conjuncts():
    tree = And(
        (
            In(field=EVENT_SESSION_KEY, values=("a", "b")),
            And(
                (
                    In(field=EVENT_SESSION_KEY, values=("b", "c")),
                    And(
                        (
                            Ordering(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T0),
                            Ordering(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T1),
                        )
                    ),
                )
            ),
        )
    )
    filters, rest = split_system(tree)
    assert filters.session_ids == ["b"]
    assert filters.since == _T1
    assert rest is None


@pytest.mark.parametrize(
    "tree",
    [
        Or(
            (
                Equals(field=EVENT_SESSION_KEY, value="s1"),
                Equals(field="color", value="red"),
            )
        ),
        Not(Equals(field=EVENT_SESSION_KEY, value="s1")),
        IsMissing(field=EVENT_SESSION_KEY),
        NotEquals(field=EVENT_SESSION_KEY, value="s1"),
        Ordering(field=EVENT_TIMESTAMP_KEY, op=">", value=_T0),
        Equals(field=reserved_property_key("other", "field"), value="x"),
    ],
)
def test_split_rejects_a_reserved_key_no_typed_value_carries(tree):
    with pytest.raises(ValueError, match=r"reserved key|system field|takes only"):
        split_system(tree)
