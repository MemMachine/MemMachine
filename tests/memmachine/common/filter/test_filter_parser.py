import datetime

import pytest

from memmachine.common.filter.filter_parser import (
    And,
    Comparison,
    FilterParseError,
    In,
    IsNull,
    Not,
    Or,
    map_filter_fields,
    normalize_filter_field,
    parse_filter,
    to_property_filter,
)


def _flatten_and(expr: And) -> list[Comparison]:
    result: list[Comparison] = []

    def _walk(node):
        if isinstance(node, And):
            _walk(node.left)
            _walk(node.right)
        else:
            assert isinstance(node, Comparison)
            result.append(node)

    _walk(expr)
    return result


def test_parse_filter_empty_string() -> None:
    assert parse_filter("") is None
    assert parse_filter(None) is None


def test_parse_filter_simple_equality() -> None:
    expr = parse_filter("owner = 'alice'")
    assert expr == Comparison(field="owner", op="=", value="alice")


def test_parse_filter_in_clause() -> None:
    expr = parse_filter("priority in (HIGH,LOW)")
    assert expr == In(field="priority", values=["HIGH", "LOW"])


def test_parse_filter_boolean_and_numeric_values() -> None:
    expr = parse_filter("count = 10 AND pi = 3.14 AND done = true AND flag = FALSE")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Comparison(field="count", op="=", value=10)
    assert children[1] == Comparison(field="pi", op="=", value=3.14)
    assert children[2] == Comparison(field="done", op="=", value=True)
    assert children[3] == Comparison(field="flag", op="=", value=False)


def test_parse_filter_greater_and_less_than() -> None:
    expr = parse_filter("count > 10 AND pi < 3.14")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Comparison(field="count", op=">", value=10)
    assert children[1] == Comparison(field="pi", op="<", value=3.14)


def test_parse_filter_greater_equal_and_less_equal() -> None:
    expr = parse_filter("count >= 10 AND pi <= 3.14")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Comparison(field="count", op=">=", value=10)
    assert children[1] == Comparison(field="pi", op="<=", value=3.14)


def test_parse_filter_and_or_precedence() -> None:
    expr = parse_filter("owner = alice OR priority = HIGH AND status = OPEN")
    assert isinstance(expr, Or)
    left = expr.left
    right = expr.right
    assert left == Comparison(field="owner", op="=", value="alice")
    assert isinstance(right, And)
    assert _flatten_and(right) == [
        Comparison(field="priority", op="=", value="HIGH"),
        Comparison(field="status", op="=", value="OPEN"),
    ]


def test_parse_filter_grouping_changes_precedence() -> None:
    expr = parse_filter("(owner = alice OR priority = HIGH) AND status = OPEN")
    assert isinstance(expr, And)
    left = expr.left
    right = expr.right
    assert isinstance(left, Or)
    assert left.left == Comparison(field="owner", op="=", value="alice")
    assert left.right == Comparison(field="priority", op="=", value="HIGH")
    assert right == Comparison(field="status", op="=", value="OPEN")


def test_parse_filter_complex_parentheses_precedence() -> None:
    expr = parse_filter(
        "status = OPEN AND (project = memmachine OR project = memguard) OR owner = bob"
    )
    assert isinstance(expr, Or)
    assert isinstance(expr.left, And)
    assert isinstance(expr.left.right, Or)
    assert expr.left.left == Comparison(field="status", op="=", value="OPEN")
    assert expr.left.right.left == Comparison(
        field="project", op="=", value="memmachine"
    )
    assert expr.left.right.right == Comparison(
        field="project", op="=", value="memguard"
    )
    assert expr.right == Comparison(field="owner", op="=", value="bob")


def test_parse_filter_deeply_nested_groups() -> None:
    expr = parse_filter(
        "((project = 'memmachine' AND owner = 'alice') OR (priority = 'HIGH' AND (status = 'OPEN' OR status = 'NEW'))) AND flag = TRUE"
    )
    assert isinstance(expr, And)
    assert expr.right == Comparison(field="flag", op="=", value=True)

    left = expr.left
    assert isinstance(left, Or)

    assert isinstance(left.left, And)
    assert left.left.left == Comparison(field="project", op="=", value="memmachine")
    assert left.left.right == Comparison(field="owner", op="=", value="alice")

    assert isinstance(left.right, And)
    assert left.right.left == Comparison(field="priority", op="=", value="HIGH")
    assert isinstance(left.right.right, Or)
    assert left.right.right.left == Comparison(field="status", op="=", value="OPEN")
    assert left.right.right.right == Comparison(field="status", op="=", value="NEW")


def test_parse_filter_is_null_operator() -> None:
    expr = parse_filter("properties.note IS NULL")
    assert expr == IsNull(field="properties.note")


def test_parse_filter_is_not_null_and_or_combination() -> None:
    expr = parse_filter(
        "(properties.note IS NOT NULL AND status = 'OPEN') OR owner IS NULL"
    )
    assert isinstance(expr, Or)
    assert isinstance(expr.left, And)
    assert expr.left.left == Not(expr=IsNull(field="properties.note"))
    assert expr.left.right == Comparison(field="status", op="=", value="OPEN")
    assert expr.right == IsNull(field="owner")


def test_keywords_case_insensitive() -> None:
    expr = parse_filter("Owner In ('Alice', 'Bob') or PRIORITY = high")
    assert isinstance(expr, Or)
    assert expr.left == In(field="Owner", values=["Alice", "Bob"])
    assert expr.right == Comparison(field="PRIORITY", op="=", value="high")


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
    assert expr == Comparison(
        field="created_at",
        op="<",
        value=datetime.datetime.fromisoformat("2026-01-19T01:56:41.513342Z"),
    )


def test_datetime_parsing_with_and_expression() -> None:
    expr = parse_filter("name='test' AND created_at >= date('2025-01-01T00:00:00')")
    assert isinstance(expr, And)
    children = _flatten_and(expr)
    assert children[0] == Comparison(field="name", op="=", value="test")
    assert children[1] == Comparison(
        field="created_at",
        op=">=",
        value=datetime.datetime.fromisoformat("2025-01-01T00:00:00"),
    )


def test_datetime_parsing_with_equality() -> None:
    expr = parse_filter("created_at = date('2026-01-19T01:56:41Z')")
    assert expr == Comparison(
        field="created_at",
        op="=",
        value=datetime.datetime.fromisoformat("2026-01-19T01:56:41Z"),
    )


def test_datetime_parsing_invalid_format() -> None:
    with pytest.raises(FilterParseError, match="Invalid ISO format date string"):
        parse_filter("created_at<date('invalid-date')")


def test_valid_fixtures_return(valid_filters) -> None:
    expr = parse_filter(valid_filters)
    assert expr is not None


# --- != / <> (Comparison with op="!=") ---


def test_parse_filter_ne_bang_equal() -> None:
    expr = parse_filter("status != 'CLOSED'")
    assert expr == Comparison(field="status", op="!=", value="CLOSED")


def test_parse_filter_ne_diamond() -> None:
    expr = parse_filter("status <> 'CLOSED'")
    assert expr == Comparison(field="status", op="!=", value="CLOSED")


def test_parse_filter_ne_numeric() -> None:
    expr = parse_filter("count != 0")
    assert expr == Comparison(field="count", op="!=", value=0)


def test_parse_filter_ne_boolean() -> None:
    expr = parse_filter("active <> false")
    assert expr == Comparison(field="active", op="!=", value=False)


def test_parse_filter_ne_in_conjunction() -> None:
    expr = parse_filter("owner = 'alice' AND status != 'CLOSED'")
    assert isinstance(expr, And)
    assert expr.left == Comparison(field="owner", op="=", value="alice")
    assert expr.right == Comparison(field="status", op="!=", value="CLOSED")


def test_parse_filter_ne_in_disjunction() -> None:
    expr = parse_filter("status <> 'CLOSED' OR priority != 'LOW'")
    assert isinstance(expr, Or)
    assert expr.left == Comparison(field="status", op="!=", value="CLOSED")
    assert expr.right == Comparison(field="priority", op="!=", value="LOW")


def test_legacy_mapping_rejects_ne() -> None:
    with pytest.raises(TypeError, match="Legacy property filters"):
        to_property_filter(parse_filter("status != 'CLOSED'"))


# --- NOT (unary logical negation) tests ---


def test_parse_filter_not_simple() -> None:
    expr = parse_filter("NOT status = 'CLOSED'")
    assert isinstance(expr, Not)
    assert expr.expr == Comparison(field="status", op="=", value="CLOSED")


def test_parse_filter_not_with_parenthesized_or() -> None:
    expr = parse_filter("NOT (status = 'CLOSED' OR status = 'ARCHIVED')")
    assert isinstance(expr, Not)
    inner = expr.expr
    assert isinstance(inner, Or)
    assert inner.left == Comparison(field="status", op="=", value="CLOSED")
    assert inner.right == Comparison(field="status", op="=", value="ARCHIVED")


def test_parse_filter_not_binds_tighter_than_and() -> None:
    # NOT x = 1 AND y = 2  =>  (NOT (x = 1)) AND (y = 2)
    expr = parse_filter("NOT x = 1 AND y = 2")
    assert isinstance(expr, And)
    assert isinstance(expr.left, Not)
    assert expr.left.expr == Comparison(field="x", op="=", value=1)
    assert expr.right == Comparison(field="y", op="=", value=2)


def test_parse_filter_not_binds_tighter_than_or() -> None:
    # NOT x = 1 OR y = 2  =>  (NOT (x = 1)) OR (y = 2)
    expr = parse_filter("NOT x = 1 OR y = 2")
    assert isinstance(expr, Or)
    assert isinstance(expr.left, Not)
    assert expr.left.expr == Comparison(field="x", op="=", value=1)
    assert expr.right == Comparison(field="y", op="=", value=2)


def test_parse_filter_double_not() -> None:
    expr = parse_filter("NOT NOT status = 'OPEN'")
    assert isinstance(expr, Not)
    assert isinstance(expr.expr, Not)
    assert expr.expr.expr == Comparison(field="status", op="=", value="OPEN")


def test_parse_filter_not_with_in() -> None:
    expr = parse_filter("NOT priority IN ('LOW', 'MEDIUM')")
    assert isinstance(expr, Not)
    assert expr.expr == In(field="priority", values=["LOW", "MEDIUM"])


def test_parse_filter_not_with_is_null() -> None:
    expr = parse_filter("NOT owner IS NULL")
    assert isinstance(expr, Not)
    assert expr.expr == IsNull(field="owner")


def test_parse_filter_not_with_ne() -> None:
    # NOT status != 'OPEN'  =>  NOT(status != 'OPEN')
    expr = parse_filter("NOT status != 'OPEN'")
    assert isinstance(expr, Not)
    assert expr.expr == Comparison(field="status", op="!=", value="OPEN")


def test_parse_filter_not_case_insensitive() -> None:
    expr = parse_filter("not status = 'CLOSED'")
    assert isinstance(expr, Not)
    assert expr.expr == Comparison(field="status", op="=", value="CLOSED")


def test_parse_filter_not_and_or_full_precedence() -> None:
    # NOT a = 1 OR b = 2 AND c = 3  =>  (NOT (a = 1)) OR ((b = 2) AND (c = 3))
    expr = parse_filter("NOT a = 1 OR b = 2 AND c = 3")
    assert isinstance(expr, Or)
    assert isinstance(expr.left, Not)
    assert expr.left.expr == Comparison(field="a", op="=", value=1)
    assert isinstance(expr.right, And)
    assert expr.right.left == Comparison(field="b", op="=", value=2)
    assert expr.right.right == Comparison(field="c", op="=", value=3)


def test_parse_filter_not_inside_and_or_chain() -> None:
    # a = 1 AND NOT b = 2 OR c = 3  =>  ((a = 1) AND (NOT (b = 2))) OR (c = 3)
    expr = parse_filter("a = 1 AND NOT b = 2 OR c = 3")
    assert isinstance(expr, Or)
    assert isinstance(expr.left, And)
    assert expr.left.left == Comparison(field="a", op="=", value=1)
    assert isinstance(expr.left.right, Not)
    assert expr.left.right.expr == Comparison(field="b", op="=", value=2)
    assert expr.right == Comparison(field="c", op="=", value=3)


def test_parse_filter_multiple_nots_in_expression() -> None:
    # NOT a = 1 AND NOT b = 2  =>  (NOT (a = 1)) AND (NOT (b = 2))
    expr = parse_filter("NOT a = 1 AND NOT b = 2")
    assert isinstance(expr, And)
    assert isinstance(expr.left, Not)
    assert expr.left.expr == Comparison(field="a", op="=", value=1)
    assert isinstance(expr.right, Not)
    assert expr.right.expr == Comparison(field="b", op="=", value=2)


def test_legacy_mapping_rejects_not() -> None:
    with pytest.raises(TypeError, match="Legacy property filters"):
        to_property_filter(parse_filter("NOT status = 'CLOSED'"))


# --- NOT IN (field NOT IN (...)) tests ---


def test_parse_filter_not_in_simple() -> None:
    expr = parse_filter("priority NOT IN ('LOW', 'MEDIUM')")
    assert expr == Not(expr=In(field="priority", values=["LOW", "MEDIUM"]))


def test_parse_filter_not_in_single_value() -> None:
    expr = parse_filter("status NOT IN ('CLOSED')")
    assert expr == Not(expr=In(field="status", values=["CLOSED"]))


def test_parse_filter_not_in_numeric_values() -> None:
    expr = parse_filter("code NOT IN (1, 2, 3)")
    assert expr == Not(expr=In(field="code", values=[1, 2, 3]))


def test_parse_filter_not_in_with_and() -> None:
    expr = parse_filter("owner = 'alice' AND status NOT IN ('CLOSED', 'ARCHIVED')")
    assert isinstance(expr, And)
    assert expr.left == Comparison(field="owner", op="=", value="alice")
    assert expr.right == Not(expr=In(field="status", values=["CLOSED", "ARCHIVED"]))


def test_parse_filter_not_in_with_or() -> None:
    expr = parse_filter("priority NOT IN ('LOW') OR owner = 'bob'")
    assert isinstance(expr, Or)
    assert expr.left == Not(expr=In(field="priority", values=["LOW"]))
    assert expr.right == Comparison(field="owner", op="=", value="bob")


def test_parse_filter_not_in_case_insensitive() -> None:
    expr = parse_filter("status not in ('CLOSED', 'ARCHIVED')")
    assert expr == Not(expr=In(field="status", values=["CLOSED", "ARCHIVED"]))


def test_parse_filter_not_without_in_raises() -> None:
    with pytest.raises(FilterParseError, match="Expected IN after NOT"):
        parse_filter("status NOT 'CLOSED'")


# --- normalize_filter_field tests ---


def test_normalize_filter_field_user_property_p_prefix() -> None:
    internal_name, is_user_property = normalize_filter_field("p.foo")
    assert internal_name == "p_foo"
    assert is_user_property is True


def test_normalize_filter_field_user_property_properties_prefix() -> None:
    internal_name, is_user_property = normalize_filter_field("properties.bar")
    assert internal_name == "p_bar"
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
    internal_name, is_user_property = normalize_filter_field("p.MyKey")
    assert internal_name == "p_MyKey"
    assert is_user_property is True


# --- map_filter_fields tests ---


def test_map_filter_fields_comparison() -> None:
    expr = Comparison(field="p.foo", op="=", value="bar")
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == Comparison(field="P.FOO", op="=", value="bar")


def test_map_filter_fields_in() -> None:
    expr = In(field="p.tag", values=["a", "b"])
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == In(field="P.TAG", values=["a", "b"])


def test_map_filter_fields_is_null() -> None:
    expr = IsNull(field="p.note")
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == IsNull(field="P.NOTE")


def test_map_filter_fields_and() -> None:
    expr = And(
        left=Comparison(field="a", op="=", value=1),
        right=Comparison(field="b", op="=", value=2),
    )
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, And)
    assert result.left == Comparison(field="A", op="=", value=1)
    assert result.right == Comparison(field="B", op="=", value=2)


def test_map_filter_fields_or() -> None:
    expr = Or(
        left=Comparison(field="x", op="=", value=1),
        right=Comparison(field="y", op="=", value=2),
    )
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, Or)
    assert result.left == Comparison(field="X", op="=", value=1)
    assert result.right == Comparison(field="Y", op="=", value=2)


def test_map_filter_fields_not() -> None:
    expr = Not(expr=Comparison(field="status", op="=", value="CLOSED"))
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, Not)
    assert result.expr == Comparison(field="STATUS", op="=", value="CLOSED")


def test_map_filter_fields_nested() -> None:
    # NOT (a = 1 AND b = 2)
    expr = Not(
        expr=And(
            left=Comparison(field="a", op="=", value=1),
            right=Comparison(field="b", op="=", value=2),
        )
    )
    result = map_filter_fields(expr, lambda f: f"prefix_{f}")
    assert isinstance(result, Not)
    assert isinstance(result.expr, And)
    assert result.expr.left == Comparison(field="prefix_a", op="=", value=1)
    assert result.expr.right == Comparison(field="prefix_b", op="=", value=2)


def test_map_filter_fields_with_normalize() -> None:
    """Test map_filter_fields combined with normalize_filter_field."""
    expr = And(
        left=Comparison(field="p.foo", op="=", value="bar"),
        right=Comparison(field="producer_id", op="=", value="alice"),
    )
    result = map_filter_fields(expr, lambda f: normalize_filter_field(f)[0])
    assert isinstance(result, And)
    assert result.left == Comparison(field="p_foo", op="=", value="bar")
    assert result.right == Comparison(field="producer_id", op="=", value="alice")
