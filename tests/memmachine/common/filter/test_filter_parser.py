import pytest

from memmachine.common.filter.filter_parser import (
    And,
    Comparison,
    FilterParseError,
    Or,
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
    assert expr == Comparison(
        field="priority",
        op="in",
        value=["HIGH", "LOW"],
    )


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
    expr = parse_filter("metadata.note IS NULL")
    assert expr == Comparison(field="metadata.note", op="is_null", value=None)


def test_parse_filter_is_not_null_and_or_combination() -> None:
    expr = parse_filter(
        "(metadata.note IS NOT NULL AND status = 'OPEN') OR owner IS NULL"
    )
    assert isinstance(expr, Or)
    assert isinstance(expr.left, And)
    assert expr.left.left == Comparison(
        field="metadata.note",
        op="is_not_null",
        value=None,
    )
    assert expr.left.right == Comparison(
        field="status",
        op="=",
        value="OPEN",
    )
    assert expr.right == Comparison(field="owner", op="is_null", value=None)


def test_keywords_case_insensitive() -> None:
    expr = parse_filter("Owner In ('Alice', 'Bob') or PRIORITY = high")
    assert isinstance(expr, Or)
    assert expr.left == Comparison(field="Owner", op="in", value=["Alice", "Bob"])
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
    with pytest.raises(TypeError, match="IN"):
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
    ]
)
def valid_filters(request) -> str:
    return request.param


def test_valid_fixtures_return(valid_filters) -> None:
    expr = parse_filter(valid_filters)
    assert expr is not None
