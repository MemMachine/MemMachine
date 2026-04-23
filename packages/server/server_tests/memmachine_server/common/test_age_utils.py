"""Unit tests for age_utils.

These tests run without Docker or any database. They exercise only the pure
helpers (sanitization, agtype parsing, parameter encoding, Cypher wrapping).
"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest

from memmachine_server.common.age_utils import (
    AgeEdge,
    AgePath,
    AgeVertex,
    AgtypeParseError,
    age_value_from_python,
    age_value_to_python,
    build_cypher_call,
    desanitize_identifier,
    encode_agtype_params,
    parse_agtype,
    render_comparison,
    sanitize_identifier,
    validate_graph_name,
)

# ---------------------------------------------------------------------------
# Identifier sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "normal_name",
        "123",
        ")(*&^%$#@!",
        "😀",
        "𰻝",
        " \t\n",
        "",
        "with spaces",
        "snake_case_name",
    ],
)
def test_sanitize_identifier_round_trip(name):
    sanitized = sanitize_identifier(name)
    assert sanitized.startswith("SANITIZED_")
    assert sanitized[0].isalpha()
    assert all(c.isalnum() or c == "_" for c in sanitized)
    assert desanitize_identifier(sanitized) == name


def test_sanitize_identifier_distinct_for_distinct_inputs():
    seen = set()
    for value in ["alpha", "ALPHA", "alpha ", " alpha", "alpha!", "alpha?"]:
        sanitized = sanitize_identifier(value)
        assert sanitized not in seen
        seen.add(sanitized)


# ---------------------------------------------------------------------------
# Graph-name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    ["mem_graph", "g1", "_underscore", "ABCxyz123", "a" * 63],
)
def test_validate_graph_name_accepts_valid(good):
    assert validate_graph_name(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "1leading_digit",
        "has space",
        "has-dash",
        "ends_with_quote'",
        "a" * 64,
        "DROP TABLE x; --",
        "has;semicolon",
    ],
)
def test_validate_graph_name_rejects_invalid(bad):
    with pytest.raises(ValueError, match="Invalid AGE graph name"):
        validate_graph_name(bad)


# ---------------------------------------------------------------------------
# Cypher call wrapping
# ---------------------------------------------------------------------------


def test_build_cypher_call_no_params():
    sql = build_cypher_call("mem_graph", "MATCH (n) RETURN n", returns=("n",))
    assert "cypher('mem_graph', $cypher_body$MATCH (n) RETURN n$cypher_body$)" in sql
    assert sql.endswith("AS (n agtype)")


def test_build_cypher_call_with_params():
    sql = build_cypher_call(
        "mem_graph",
        "MATCH (n) WHERE n.uid = $uid RETURN n",
        returns=("n",),
        has_params=True,
    )
    assert "$1::agtype" in sql
    assert "$cypher_body$" in sql
    assert "MATCH (n) WHERE n.uid = $uid RETURN n" in sql


def test_build_cypher_call_multiple_returns():
    sql = build_cypher_call(
        "mem_graph",
        "MATCH (n) RETURN n, count(n)",
        returns=("n", "c"),
    )
    assert "AS (n agtype, c agtype)" in sql


def test_build_cypher_call_rejects_invalid_graph_name():
    with pytest.raises(ValueError, match="Invalid AGE graph name"):
        build_cypher_call("bad name", "RETURN 1", returns=("v",))


def test_build_cypher_call_rejects_returns_empty():
    with pytest.raises(ValueError, match="at least one return column"):
        build_cypher_call("mem_graph", "RETURN 1", returns=())


def test_build_cypher_call_rejects_collision_with_delimiter():
    with pytest.raises(ValueError, match="reserved dollar-quoted delimiter"):
        build_cypher_call(
            "mem_graph",
            "MATCH (n) RETURN $cypher_body$ as v",
            returns=("v",),
        )


# ---------------------------------------------------------------------------
# Parameter encoding
# ---------------------------------------------------------------------------


def test_encode_agtype_params_none():
    assert encode_agtype_params(None) == "{}"


def test_encode_agtype_params_primitives():
    out = encode_agtype_params({"name": "abc", "n": 5, "ok": True})
    # JSON keys remain quoted; ordering follows dict insertion.
    assert '"name":"abc"' in out
    assert '"n":5' in out
    assert '"ok":true' in out


def test_encode_agtype_params_datetime_to_iso_utc():
    dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)
    out = encode_agtype_params({"when": dt})
    assert '"when":"2024-06-15T12:30:00+00:00"' in out


def test_encode_agtype_params_datetime_naive_treated_as_utc():
    dt = datetime(2024, 6, 15, 12, 30, 0)  # noqa: DTZ001  (intentional)
    out = encode_agtype_params({"when": dt})
    # Naive becomes UTC-aware; rendered as +00:00 offset.
    assert "2024-06-15T12:30:00+00:00" in out


def test_encode_agtype_params_datetime_normalized_to_utc():
    pst = timezone(timedelta(hours=-8))
    dt = datetime(2024, 6, 15, 4, 30, 0, tzinfo=pst)
    out = encode_agtype_params({"when": dt})
    assert "2024-06-15T12:30:00+00:00" in out


def test_encode_agtype_params_nested():
    payload = {"props": {"a": 1, "b": [1, 2, 3]}}
    out = encode_agtype_params(payload)
    assert '"props":{"a":1,"b":[1,2,3]}' in out


# ---------------------------------------------------------------------------
# agtype parsing
# ---------------------------------------------------------------------------


def test_parse_agtype_primitives():
    assert parse_agtype("null") is None
    assert parse_agtype("true") is True
    assert parse_agtype("false") is False
    assert parse_agtype("42") == 42
    assert parse_agtype("-7") == -7
    assert parse_agtype("3.14") == 3.14
    assert parse_agtype("-1.5e2") == -150.0


def test_parse_agtype_strings_with_escapes():
    assert parse_agtype('"hello"') == "hello"
    assert parse_agtype(r'"with \"quotes\""') == 'with "quotes"'
    assert parse_agtype(r'"\n\t\\"') == "\n\t\\"
    assert parse_agtype(r'"\u0041"') == "A"


def test_parse_agtype_arrays_and_objects():
    assert parse_agtype("[]") == []
    assert parse_agtype("[1, 2, 3]") == [1, 2, 3]
    assert parse_agtype('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_agtype_vertex():
    raw = '{"id": 844424930131969, "label": "Entity", "properties": {"uid": "abc"}}::vertex'
    vertex = parse_agtype(raw)
    assert isinstance(vertex, AgeVertex)
    assert vertex.id == 844424930131969
    assert vertex.label == "Entity"
    assert vertex.properties == {"uid": "abc"}


def test_parse_agtype_edge():
    raw = (
        '{"id": 1, "label": "REL", "start_id": 2, "end_id": 3, '
        '"properties": {"weight": 0.5}}::edge'
    )
    edge = parse_agtype(raw)
    assert isinstance(edge, AgeEdge)
    assert edge.id == 1
    assert edge.label == "REL"
    assert edge.start_id == 2
    assert edge.end_id == 3
    assert edge.properties == {"weight": 0.5}


def test_parse_agtype_path():
    raw = "[1, 2, 3]::path"
    path = parse_agtype(raw)
    assert isinstance(path, AgePath)
    assert path.elements == (1, 2, 3)


def test_parse_agtype_handles_whitespace_and_none():
    assert parse_agtype(None) is None
    assert parse_agtype("  null  ") is None
    assert parse_agtype('  "padded"  ') == "padded"


def test_parse_agtype_bytes_input():
    assert parse_agtype(b'"hello"') == "hello"


def test_parse_agtype_rejects_trailing_data():
    with pytest.raises(AgtypeParseError):
        parse_agtype("123 garbage")


def test_parse_agtype_rejects_unterminated_string():
    with pytest.raises(AgtypeParseError):
        parse_agtype('"open')


def test_parse_agtype_rejects_unknown_token():
    with pytest.raises(AgtypeParseError):
        parse_agtype("undefined")


# ---------------------------------------------------------------------------
# Value round-trip
# ---------------------------------------------------------------------------


def test_age_value_round_trip_for_datetime_utc():
    original = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    encoded = age_value_from_python(original)
    assert isinstance(encoded, str)
    decoded = age_value_to_python(encoded)
    assert decoded == original


def test_age_value_round_trip_for_naive_datetime():
    original = datetime(2024, 1, 2, 3, 4, 5)  # noqa: DTZ001  (intentional)
    encoded = age_value_from_python(original)
    decoded = age_value_to_python(encoded)
    assert decoded.tzinfo is not None
    # Naive datetimes are interpreted as UTC.
    assert decoded == original.replace(tzinfo=UTC)


def test_age_value_round_trip_for_lists():
    original = [datetime(2024, 1, 1, tzinfo=UTC), "x", 5]
    encoded = age_value_from_python(original)
    decoded = age_value_to_python(encoded)
    assert decoded[0] == original[0]
    assert decoded[1] == "x"
    assert decoded[2] == 5


def test_age_value_to_python_returns_string_when_not_iso():
    assert age_value_to_python("hello") == "hello"
    assert age_value_to_python("2024") == "2024"


# ---------------------------------------------------------------------------
# render_comparison
# ---------------------------------------------------------------------------


def test_render_comparison_basic():
    assert render_comparison("n.x", "=", "$p1", value=5) == "n.x = $p1"
    assert render_comparison("n.x", "!=", "$p1", value=5) == "n.x <> $p1"
    assert render_comparison("n.x", ">=", "$p1", value=5) == "n.x >= $p1"


def test_render_comparison_datetime_string_compare():
    dt = datetime(2024, 6, 15, tzinfo=UTC)
    # AGE stores datetimes as ISO-8601 UTC strings, so comparison is plain.
    rendered = render_comparison("n.t", ">", "$p1", value=dt)
    assert rendered == "n.t > $p1"


def test_render_comparison_rejects_list_value():
    with pytest.raises(TypeError, match="comparison cannot accept list"):
        # Intentionally passing a list here to exercise the defensive
        # guard in render_comparison, which normally takes a scalar
        # PropertyValue. Cast to Any so ty doesn't complain.
        render_comparison("n.x", "=", "$p1", value=cast(Any, [1, 2, 3]))
