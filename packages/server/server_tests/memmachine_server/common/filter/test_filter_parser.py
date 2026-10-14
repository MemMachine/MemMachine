import datetime
import itertools
import random
from collections.abc import Iterator

import pytest

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
    extract_filter,
    filter_fields,
    map_filter_fields,
)
from memmachine_server.common.filter.filter_parser import (
    FilterParseError,
    normalize_filter_field,
    parse_filter,
    to_property_filter,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_partition import (
    evaluate_filter,
)

from .random_filters import VALUES, random_filter, random_properties


def _flatten_and(expr: And) -> list[Equals | Ordering]:
    result: list[Equals | Ordering] = []

    def _walk(node):
        if isinstance(node, And):
            for operand in node.operands:
                _walk(operand)
        else:
            assert isinstance(node, Equals | Ordering)
            result.append(node)

    _walk(expr)
    return result


def test_parse_filter_empty_string() -> None:
    assert parse_filter("") is None
    assert parse_filter(None) is None


def test_parse_filter_simple_equality() -> None:
    expr = parse_filter("owner = 'alice'")
    assert expr == Equals(field="owner", value="alice")


def test_parse_filter_in_clause() -> None:
    expr = parse_filter("priority in (HIGH,LOW)")
    assert expr == In(field="priority", values=("HIGH", "LOW"))


def test_parse_filter_boolean_and_numeric_values() -> None:
    expr = parse_filter("count = 10 AND pi = 3.14 AND done = true AND flag = FALSE")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Equals(field="count", value=10)
    assert children[1] == Equals(field="pi", value=3.14)
    assert children[2] == Equals(field="done", value=True)
    assert children[3] == Equals(field="flag", value=False)


def test_parse_filter_greater_and_less_than() -> None:
    expr = parse_filter("count > 10 AND pi < 3.14")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Ordering(field="count", op=">", value=10)
    assert children[1] == Ordering(field="pi", op="<", value=3.14)


def test_parse_filter_greater_equal_and_less_equal() -> None:
    expr = parse_filter("count >= 10 AND pi <= 3.14")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Ordering(field="count", op=">=", value=10)
    assert children[1] == Ordering(field="pi", op="<=", value=3.14)


def test_parse_filter_and_or_precedence() -> None:
    expr = parse_filter("owner = alice OR priority = HIGH AND status = OPEN")
    assert isinstance(expr, Or)
    left = expr.operands[0]
    right = expr.operands[1]
    assert left == Equals(field="owner", value="alice")
    assert isinstance(right, And)
    assert _flatten_and(right) == [
        Equals(field="priority", value="HIGH"),
        Equals(field="status", value="OPEN"),
    ]


def test_parse_filter_grouping_changes_precedence() -> None:
    expr = parse_filter("(owner = alice OR priority = HIGH) AND status = OPEN")
    assert isinstance(expr, And)
    left = expr.operands[0]
    right = expr.operands[1]
    assert isinstance(left, Or)
    assert left.operands[0] == Equals(field="owner", value="alice")
    assert left.operands[1] == Equals(field="priority", value="HIGH")
    assert right == Equals(field="status", value="OPEN")


def test_parse_filter_complex_parentheses_precedence() -> None:
    expr = parse_filter(
        "status = OPEN AND (project = memmachine OR project = memguard) OR owner = bob"
    )
    assert isinstance(expr, Or)
    assert isinstance(expr.operands[0], And)
    assert isinstance(expr.operands[0].operands[1], Or)
    assert expr.operands[0].operands[0] == Equals(field="status", value="OPEN")
    assert expr.operands[0].operands[1].operands[0] == Equals(
        field="project", value="memmachine"
    )
    assert expr.operands[0].operands[1].operands[1] == Equals(
        field="project", value="memguard"
    )
    assert expr.operands[1] == Equals(field="owner", value="bob")


def test_parse_filter_deeply_nested_groups() -> None:
    expr = parse_filter(
        "((project = 'memmachine' AND owner = 'alice') OR (priority = 'HIGH' AND (status = 'OPEN' OR status = 'NEW'))) AND flag = TRUE"
    )
    assert isinstance(expr, And)
    assert expr.operands[1] == Equals(field="flag", value=True)

    left = expr.operands[0]
    assert isinstance(left, Or)

    assert isinstance(left.operands[0], And)
    assert left.operands[0].operands[0] == Equals(field="project", value="memmachine")
    assert left.operands[0].operands[1] == Equals(field="owner", value="alice")

    assert isinstance(left.operands[1], And)
    assert left.operands[1].operands[0] == Equals(field="priority", value="HIGH")
    assert isinstance(left.operands[1].operands[1], Or)
    assert left.operands[1].operands[1].operands[0] == Equals(
        field="status", value="OPEN"
    )
    assert left.operands[1].operands[1].operands[1] == Equals(
        field="status", value="NEW"
    )


def test_parse_filter_is_null_operator() -> None:
    expr = parse_filter("metadata.note IS NULL")
    assert expr == IsNull(field="metadata.note")


def test_parse_filter_is_not_null_and_or_combination() -> None:
    expr = parse_filter(
        "(metadata.note IS NOT NULL AND status = 'OPEN') OR owner IS NULL"
    )
    assert isinstance(expr, Or)
    assert isinstance(expr.operands[0], And)
    assert expr.operands[0].operands[0] == Not(IsNull(field="metadata.note"))
    assert expr.operands[0].operands[1] == Equals(field="status", value="OPEN")
    assert expr.operands[1] == IsNull(field="owner")


def test_keywords_case_insensitive() -> None:
    expr = parse_filter("Owner In ('Alice', 'Bob') or PRIORITY = high")
    assert isinstance(expr, Or)
    assert expr.operands[0] == In(field="Owner", values=("Alice", "Bob"))
    assert expr.operands[1] == Equals(field="PRIORITY", value="high")


def test_legacy_mapping_generation() -> None:
    expr = parse_filter("owner = alice AND project = memmachine")
    assert to_property_filter(expr) == {
        "owner": "alice",
        "project": "memmachine",
    }


def test_legacy_mapping_rejects_or_and_in() -> None:
    error_msg = "Legacy property filters"
    with pytest.raises(TypeError, match=error_msg):
        to_property_filter(parse_filter("owner = alice OR owner = bob"))
    with pytest.raises(TypeError, match=error_msg):
        to_property_filter(parse_filter("owner IN ('alice', 'bob')"))


def test_to_property_filter_returns_none_for_empty_expr() -> None:
    assert to_property_filter(None) is None


def test_parse_filter_raises_custom_error() -> None:
    with pytest.raises(FilterParseError):
        parse_filter("owner =")


@pytest.fixture(
    params=[
        "set_id in ('user-88') AND tag in ('writing_style')",
        "set_id in ('user-88')",
        "created_at<date('2026-01-19T01:56:41.513342Z')",
        "created_at < date('2026-01-19T01:56:41.513342Z')",
    ]
)
def valid_filters(request) -> str:
    return request.param


def test_datetime_parsing() -> None:
    expr = parse_filter("created_at < date('2026-01-19T01:56:41.513342Z')")
    assert expr is not None
    assert expr == Ordering(
        field="created_at",
        op="<",
        value=datetime.datetime.fromisoformat("2026-01-19T01:56:41.513342Z"),
    )


def test_datetime_parsing_with_and_expression() -> None:
    expr = parse_filter("name='test' AND created_at >= date('2025-01-01T00:00:00')")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Equals(field="name", value="test")
    assert children[1] == Ordering(
        field="created_at",
        op=">=",
        value=datetime.datetime.fromisoformat("2025-01-01T00:00:00"),
    )


def test_datetime_parsing_with_equality() -> None:
    expr = parse_filter("created_at = date('2026-01-19T01:56:41Z')")
    assert expr == Equals(
        field="created_at",
        value=datetime.datetime.fromisoformat("2026-01-19T01:56:41Z"),
    )


def test_comparison_normalizes_datetime_values_at_construction() -> None:
    """The language owns datetime semantics: instants, naive means UTC."""
    offset_aware = Ordering(
        field="created_at",
        op=">=",
        value=datetime.datetime.fromisoformat("2024-01-01T13:30:45-08:00"),
    )
    assert isinstance(offset_aware.value, datetime.datetime)
    assert offset_aware.value == datetime.datetime(
        2024, 1, 1, 21, 30, 45, tzinfo=datetime.UTC
    )
    assert offset_aware.value.tzinfo == datetime.UTC

    naive = Ordering(
        field="created_at",
        op=">=",
        value=datetime.datetime.fromisoformat("2024-01-01T13:30:45"),
    )
    assert isinstance(naive.value, datetime.datetime)
    assert naive.value == datetime.datetime(2024, 1, 1, 13, 30, 45, tzinfo=datetime.UTC)


def test_parsed_date_literals_carry_utc_instants() -> None:
    """A date() literal's offset is consumed into a UTC instant."""
    expr = parse_filter("created_at >= date('2024-01-01T13:30:45-08:00')")
    assert isinstance(expr, Ordering)
    assert isinstance(expr.value, datetime.datetime)
    assert expr.value == datetime.datetime(2024, 1, 1, 21, 30, 45, tzinfo=datetime.UTC)
    assert expr.value.tzinfo == datetime.UTC


def test_datetime_parsing_invalid_format() -> None:
    with pytest.raises(FilterParseError, match="Invalid ISO format date string"):
        parse_filter("created_at<date('invalid-date')")


def test_valid_fixtures_return(valid_filters) -> None:
    expr = parse_filter(valid_filters)
    assert expr is not None


# --- != / <> (Not(Equals)) ---


def test_parse_filter_ne_bang_equal() -> None:
    expr = parse_filter("status != 'CLOSED'")
    assert expr == Not(Equals(field="status", value="CLOSED"))


def test_parse_filter_ne_diamond() -> None:
    expr = parse_filter("status <> 'CLOSED'")
    assert expr == Not(Equals(field="status", value="CLOSED"))


def test_parse_filter_ne_numeric() -> None:
    expr = parse_filter("count != 0")
    assert expr == Not(Equals(field="count", value=0))


def test_parse_filter_ne_boolean() -> None:
    expr = parse_filter("active <> false")
    assert expr == Not(Equals(field="active", value=False))


def test_parse_filter_ne_in_conjunction() -> None:
    expr = parse_filter("owner = 'alice' AND status != 'CLOSED'")
    assert isinstance(expr, And)
    assert expr.operands[0] == Equals(field="owner", value="alice")
    assert expr.operands[1] == Not(Equals(field="status", value="CLOSED"))


def test_parse_filter_ne_in_disjunction() -> None:
    expr = parse_filter("status <> 'CLOSED' OR priority != 'LOW'")
    assert isinstance(expr, Or)
    assert expr.operands[0] == Not(Equals(field="status", value="CLOSED"))
    assert expr.operands[1] == Not(Equals(field="priority", value="LOW"))


def test_legacy_mapping_rejects_ne() -> None:
    with pytest.raises(TypeError, match="Legacy property filters"):
        to_property_filter(parse_filter("status != 'CLOSED'"))


# --- NOT (unary logical negation) tests ---


def test_parse_filter_not_simple() -> None:
    expr = parse_filter("NOT status = 'CLOSED'")
    assert isinstance(expr, Not)
    assert expr.operand == Equals(field="status", value="CLOSED")


def test_parse_filter_not_with_parenthesized_or() -> None:
    expr = parse_filter("NOT (status = 'CLOSED' OR status = 'ARCHIVED')")
    assert isinstance(expr, Not)
    inner = expr.operand
    assert isinstance(inner, Or)
    assert inner.operands[0] == Equals(field="status", value="CLOSED")
    assert inner.operands[1] == Equals(field="status", value="ARCHIVED")


def test_parse_filter_not_binds_tighter_than_and() -> None:
    # NOT x = 1 AND y = 2  =>  (NOT (x = 1)) AND (y = 2)
    expr = parse_filter("NOT x = 1 AND y = 2")
    assert isinstance(expr, And)
    assert isinstance(expr.operands[0], Not)
    assert expr.operands[0].operand == Equals(field="x", value=1)
    assert expr.operands[1] == Equals(field="y", value=2)


def test_parse_filter_not_binds_tighter_than_or() -> None:
    # NOT x = 1 OR y = 2  =>  (NOT (x = 1)) OR (y = 2)
    expr = parse_filter("NOT x = 1 OR y = 2")
    assert isinstance(expr, Or)
    assert isinstance(expr.operands[0], Not)
    assert expr.operands[0].operand == Equals(field="x", value=1)
    assert expr.operands[1] == Equals(field="y", value=2)


def test_parse_filter_double_not() -> None:
    expr = parse_filter("NOT NOT status = 'OPEN'")
    assert isinstance(expr, Not)
    assert isinstance(expr.operand, Not)
    assert expr.operand.operand == Equals(field="status", value="OPEN")


def test_parse_filter_not_with_in() -> None:
    expr = parse_filter("NOT priority IN ('LOW', 'MEDIUM')")
    assert isinstance(expr, Not)
    assert expr.operand == In(field="priority", values=("LOW", "MEDIUM"))


def test_parse_filter_not_with_is_null() -> None:
    expr = parse_filter("NOT owner IS NULL")
    assert isinstance(expr, Not)
    assert expr.operand == IsNull(field="owner")


def test_parse_filter_not_with_ne() -> None:
    # NOT status != 'OPEN'  =>  NOT(NOT(status = 'OPEN'))
    expr = parse_filter("NOT status != 'OPEN'")
    assert isinstance(expr, Not)
    assert expr.operand == Not(Equals(field="status", value="OPEN"))


def test_parse_filter_not_case_insensitive() -> None:
    expr = parse_filter("not status = 'CLOSED'")
    assert isinstance(expr, Not)
    assert expr.operand == Equals(field="status", value="CLOSED")


def test_parse_filter_not_and_or_full_precedence() -> None:
    # NOT a = 1 OR b = 2 AND c = 3  =>  (NOT (a = 1)) OR ((b = 2) AND (c = 3))
    expr = parse_filter("NOT a = 1 OR b = 2 AND c = 3")
    assert isinstance(expr, Or)
    assert isinstance(expr.operands[0], Not)
    assert expr.operands[0].operand == Equals(field="a", value=1)
    assert isinstance(expr.operands[1], And)
    assert expr.operands[1].operands[0] == Equals(field="b", value=2)
    assert expr.operands[1].operands[1] == Equals(field="c", value=3)


def test_parse_filter_not_inside_and_or_chain() -> None:
    # a = 1 AND NOT b = 2 OR c = 3  =>  ((a = 1) AND (NOT (b = 2))) OR (c = 3)
    expr = parse_filter("a = 1 AND NOT b = 2 OR c = 3")
    assert isinstance(expr, Or)
    assert isinstance(expr.operands[0], And)
    assert expr.operands[0].operands[0] == Equals(field="a", value=1)
    assert isinstance(expr.operands[0].operands[1], Not)
    assert expr.operands[0].operands[1].operand == Equals(field="b", value=2)
    assert expr.operands[1] == Equals(field="c", value=3)


def test_parse_filter_multiple_nots_in_expression() -> None:
    # NOT a = 1 AND NOT b = 2  =>  (NOT (a = 1)) AND (NOT (b = 2))
    expr = parse_filter("NOT a = 1 AND NOT b = 2")
    assert isinstance(expr, And)
    assert isinstance(expr.operands[0], Not)
    assert expr.operands[0].operand == Equals(field="a", value=1)
    assert isinstance(expr.operands[1], Not)
    assert expr.operands[1].operand == Equals(field="b", value=2)


def test_legacy_mapping_rejects_not() -> None:
    with pytest.raises(TypeError, match="Legacy property filters"):
        to_property_filter(parse_filter("NOT status = 'CLOSED'"))


# --- NOT IN (field NOT IN (...)) tests ---


def test_parse_filter_not_in_simple() -> None:
    expr = parse_filter("priority NOT IN ('LOW', 'MEDIUM')")
    assert expr == Not(In(field="priority", values=("LOW", "MEDIUM")))


def test_parse_filter_not_in_single_value() -> None:
    expr = parse_filter("status NOT IN ('CLOSED')")
    assert expr == Not(In(field="status", values=("CLOSED",)))


def test_parse_filter_not_in_numeric_values() -> None:
    expr = parse_filter("code NOT IN (1, 2, 3)")
    assert expr == Not(In(field="code", values=(1, 2, 3)))


def test_parse_filter_not_in_with_and() -> None:
    expr = parse_filter("owner = 'alice' AND status NOT IN ('CLOSED', 'ARCHIVED')")
    assert isinstance(expr, And)
    assert expr.operands[0] == Equals(field="owner", value="alice")
    assert expr.operands[1] == Not(In(field="status", values=("CLOSED", "ARCHIVED")))


def test_parse_filter_not_in_with_or() -> None:
    expr = parse_filter("priority NOT IN ('LOW') OR owner = 'bob'")
    assert isinstance(expr, Or)
    assert expr.operands[0] == Not(In(field="priority", values=("LOW",)))
    assert expr.operands[1] == Equals(field="owner", value="bob")


def test_parse_filter_not_in_case_insensitive() -> None:
    expr = parse_filter("status not in ('CLOSED', 'ARCHIVED')")
    assert expr == Not(In(field="status", values=("CLOSED", "ARCHIVED")))


def test_parse_filter_not_without_in_raises() -> None:
    with pytest.raises(FilterParseError, match="Expected IN after NOT"):
        parse_filter("status NOT 'CLOSED'")


# --- normalize_filter_field tests ---


def test_normalize_filter_field_user_property_m_prefix() -> None:
    internal_name, is_user_property = normalize_filter_field("m.foo")
    assert internal_name == "metadata.foo"
    assert is_user_property is True


def test_normalize_filter_field_user_property_metadata_prefix() -> None:
    internal_name, is_user_property = normalize_filter_field("metadata.bar")
    assert internal_name == "metadata.bar"
    assert is_user_property is True


def test_normalize_filter_field_system_field() -> None:
    internal_name, is_user_property = normalize_filter_field("producer_id")
    assert internal_name == "producer_id"
    assert is_user_property is False


def test_normalize_filter_field_system_field_with_underscore() -> None:
    internal_name, is_user_property = normalize_filter_field("producer_role")
    assert internal_name == "producer_role"
    assert is_user_property is False


def test_normalize_filter_field_preserves_case() -> None:
    # User property keys should preserve their original case
    internal_name, is_user_property = normalize_filter_field("m.MyKey")
    assert internal_name == "metadata.MyKey"
    assert is_user_property is True


# --- map_filter_fields tests ---


def test_map_filter_fields_comparison() -> None:
    expr = Equals(field="m.foo", value="bar")
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == Equals(field="M.FOO", value="bar")


def test_map_filter_fields_in() -> None:
    expr = In(field="m.tag", values=("a", "b"))
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == In(field="M.TAG", values=("a", "b"))


def test_map_filter_fields_is_null() -> None:
    expr = IsNull(field="m.note")
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == IsNull(field="M.NOTE")


def test_map_filter_fields_and() -> None:
    expr = And((Equals(field="a", value=1), Equals(field="b", value=2)))
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, And)
    assert result.operands[0] == Equals(field="A", value=1)
    assert result.operands[1] == Equals(field="B", value=2)


def test_map_filter_fields_or() -> None:
    expr = Or((Equals(field="x", value=1), Equals(field="y", value=2)))
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, Or)
    assert result.operands[0] == Equals(field="X", value=1)
    assert result.operands[1] == Equals(field="Y", value=2)


def test_map_filter_fields_not() -> None:
    expr = Not(Equals(field="status", value="CLOSED"))
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, Not)
    assert result.operand == Equals(field="STATUS", value="CLOSED")


def test_map_filter_fields_nested() -> None:
    # NOT (a = 1 AND b = 2)
    expr = Not(And((Equals(field="a", value=1), Equals(field="b", value=2))))
    result = map_filter_fields(expr, lambda f: f"prefix_{f}")
    assert isinstance(result, Not)
    assert isinstance(result.operand, And)
    assert result.operand.operands[0] == Equals(field="prefix_a", value=1)
    assert result.operand.operands[1] == Equals(field="prefix_b", value=2)


def test_parse_filter_in_rejects_bool() -> None:
    with pytest.raises(FilterParseError, match="IN lists only support int and str"):
        parse_filter("flag IN (true, false)")


def test_parse_filter_in_rejects_float() -> None:
    with pytest.raises(FilterParseError, match="IN lists only support int and str"):
        parse_filter("x IN (1.5, 2.5)")


def test_parse_filter_in_rejects_mixed_types() -> None:
    with pytest.raises(FilterParseError, match="Mixed types in IN list"):
        parse_filter("x IN (1, 'two')")


def test_map_filter_fields_with_normalize() -> None:
    """Test map_filter_fields combined with normalize_filter_field."""
    expr = And(
        (Equals(field="m.foo", value="bar"), Equals(field="producer_id", value="alice"))
    )
    result = map_filter_fields(expr, lambda f: normalize_filter_field(f)[0])
    assert isinstance(result, And)
    assert result.operands[0] == Equals(field="metadata.foo", value="bar")
    assert result.operands[1] == Equals(field="producer_id", value="alice")


def test_filter_fields_names_every_field_under_every_node() -> None:
    expr = parse_filter(
        "m.color = 'red' AND (age > 3 OR NOT tag IN ('a')) AND name IS NULL"
    )
    assert expr is not None
    assert filter_fields(expr) == frozenset({"m.color", "age", "tag", "name"})


def test_parse_filter_ordering_takes_a_number_or_a_date() -> None:
    with pytest.raises(FilterParseError, match="take a number or a date"):
        parse_filter("age > 'x'")
    with pytest.raises(FilterParseError, match="take a number or a date"):
        parse_filter("age >= true")


# --- extract_filter tests ---

_ALLOWED = frozenset({"d1", "d2", "d3"})
_OTHER = ("u1", "u2")
_FIELDS = (*sorted(_ALLOWED), *_OTHER)

_D1 = Equals(field="d1", value=1)
_D2 = Ordering(field="d2", op="<", value=2)
_D3 = In(field="d3", values=(1, 2))
_U1 = Equals(field="u1", value=1)
_U2 = IsNull(field="u2")


@pytest.mark.parametrize(
    ("expr", "part"),
    [
        pytest.param(_D1, _D1, id="an allowed leaf"),
        pytest.param(_U1, None, id="another leaf"),
        pytest.param(And((_D1, _D2)), And((_D1, _D2)), id="and, all allowed"),
        pytest.param(And((_D1, _U1)), _D1, id="and, last other"),
        pytest.param(And((_U1, _D1)), _D1, id="and, first other"),
        pytest.param(And((_D1, _U1, _D2)), And((_D1, _D2)), id="and, middle other"),
        pytest.param(And((_U1, _U2)), None, id="and, all other"),
        pytest.param(Or((_D1, _D2)), Or((_D1, _D2)), id="or, all allowed"),
        pytest.param(Or((_D1, _U1)), None, id="or, last other"),
        pytest.param(Or((_U1, _D1)), None, id="or, first other"),
        pytest.param(
            Or((And((_D1, _U1)), And((_U2, _D2)))),
            Or((_D1, _D2)),
            id="or of ands",
        ),
        pytest.param(And((_D1,)), And((_D1,)), id="and of one, allowed"),
        pytest.param(Or((_U1,)), None, id="or of one, other"),
        pytest.param(Not(_D1), Not(_D1), id="not, allowed"),
        pytest.param(Not(_U1), None, id="not, other"),
        pytest.param(
            Not(And((_D1, _D2))), Not(And((_D1, _D2))), id="negated and, all allowed"
        ),
        pytest.param(Not(And((_D1, _U1))), None, id="negated and, last other"),
        pytest.param(Not(And((_U1, _D1))), None, id="negated and, first other"),
        pytest.param(Not(Or((_D1, _U1))), Not(_D1), id="negated or, last other"),
        pytest.param(Not(Or((_U1, _D1))), Not(_D1), id="negated or, first other"),
        pytest.param(
            Not(Or((_D1, _U1, _D2))), Not(Or((_D1, _D2))), id="negated or, middle other"
        ),
        pytest.param(Not(Or((_U1, _U2))), None, id="negated or, all other"),
        pytest.param(Not(Not(And((_D1, _U1)))), Not(Not(_D1)), id="double negation"),
        pytest.param(
            Not(Or((_U1, Not(_D1)))), Not(Not(_D1)), id="negation under a negated or"
        ),
        pytest.param(
            And((_D1, Or((And((_D2, _U1)), Not(Or((_U2, _D3))))))),
            And((_D1, Or((_D2, Not(_D3))))),
            id="deep, with every node",
        ),
        pytest.param(
            Or((_D1, And((_U1, Or((_U2, _D2)))))),
            None,
            id="deep, an or arm with no allowed part",
        ),
        pytest.param(
            Not(And((_D1, Or((_U1, _D2))))),
            Not(And((_D1, _D2))),
            id="deep, negated",
        ),
    ],
)
def test_extract_filter(expr: FilterExpr, part: FilterExpr | None) -> None:
    """Each node keeps what it can of its operands, by its kind and its parity of negations."""
    assert extract_filter(expr, allowed_fields=_ALLOWED) == part


def test_extract_filter_keeps_every_record_the_filter_matches() -> None:
    """The part names only allowed fields, and only ever widens the filter."""
    rng = random.Random(1702)
    for _ in range(2_000):
        expr = random_filter(rng, _FIELDS, depth=4)
        part = extract_filter(expr, allowed_fields=_ALLOWED)
        assert part is None or filter_fields(part) <= _ALLOWED, (expr, part)
        for _ in range(32):
            properties = random_properties(rng, _FIELDS)
            if evaluate_filter(expr, properties):
                assert _matches(part, properties), (expr, part, properties)


def test_extract_filter_returns_a_filter_on_allowed_fields_alone_unchanged() -> None:
    rng = random.Random(1703)
    for _ in range(500):
        expr = random_filter(rng, sorted(_ALLOWED), depth=4)
        assert extract_filter(expr, allowed_fields=_ALLOWED) == expr


def test_extract_filter_is_the_tightest_when_each_other_field_appears_once() -> None:
    """A record matches the part exactly when some values of the other fields make it match the filter."""
    rng = random.Random(1704)
    # Missing, and values on both sides of every compared value, so each leaf
    # on another field can go either way.
    other_values = (None, -1, *VALUES, 4)
    checked = 0
    while checked < 500:
        expr = random_filter(rng, _FIELDS, depth=4)
        leaf_fields = list(_leaf_fields(expr))
        if any(leaf_fields.count(field) > 1 for field in _OTHER):
            continue
        part = extract_filter(expr, allowed_fields=_ALLOWED)
        for _ in range(16):
            allowed = random_properties(rng, sorted(_ALLOWED))
            completions = (
                allowed
                | {
                    field: value
                    for field, value in zip(_OTHER, values, strict=True)
                    if value is not None
                }
                for values in itertools.product(other_values, repeat=len(_OTHER))
            )
            assert _matches(part, allowed) == any(
                evaluate_filter(expr, completion) for completion in completions
            ), (expr, part, allowed)
        checked += 1


def _leaf_fields(expr: FilterExpr) -> Iterator[str]:
    """The field of every leaf, once per leaf."""
    match expr:
        case And(operands) | Or(operands):
            for operand in operands:
                yield from _leaf_fields(operand)
        case Not(operand):
            yield from _leaf_fields(operand)
        case _:
            yield from filter_fields(expr)


def _matches(part: FilterExpr | None, properties: dict[str, PropertyValue]) -> bool:
    """Whether a record matches a part; no part places no condition."""
    return part is None or evaluate_filter(part, properties)
