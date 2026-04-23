# ruff: noqa: ANN401
"""Helpers for working with Apache AGE values, Cypher wrapping, and connections.

Apache AGE is a PostgreSQL extension that adds openCypher support. It uses its
own custom type (``agtype``) for vertices, edges, paths, maps, lists, and
primitive values. Cypher statements must be wrapped in a SQL ``cypher()`` call,
so this module provides the small set of primitives needed by
``AgeVectorGraphStore`` to build and execute queries.

The utilities here are intentionally narrow in scope: they cover the subset of
``agtype`` that MemMachine persists (primitives, lists, maps, vertices, edges),
the exact Cypher-wrapping shape that AGE requires, and the per-connection
extension setup that every pooled connection must run.

``ANN401`` is disabled module-wide: the agtype parser, parameter encoder, and
SQLAlchemy DB-API adapter all operate on genuinely-polymorphic values, and
replacing ``Any`` with wide unions would obscure the point of each helper.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from memmachine_server.common.data_types import PropertyValue

__all__ = [
    "AgeEdge",
    "AgePath",
    "AgeVertex",
    "AgtypeParseError",
    "age_value_from_python",
    "age_value_to_python",
    "build_cypher_call",
    "encode_agtype_params",
    "parse_agtype",
    "render_comparison",
    "sanitize_identifier",
    "setup_age_sync_connection",
    "validate_graph_name",
]


# AGE graph names become PostgreSQL schema names, so they must be valid
# identifiers. We apply a strict safelist to avoid any need for quoting.
_GRAPH_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

# Sanitized identifiers for labels and properties use a fixed prefix followed by
# only ASCII letters, digits, and underscores. This mirrors the Neo4j backend's
# approach so that encoded names remain deterministic across implementations.
_SANITIZE_PREFIX = "SANITIZED_"


class AgtypeParseError(ValueError):
    """Raised when an ``agtype`` text value cannot be decoded."""


@dataclass(frozen=True)
class AgeVertex:
    """Decoded representation of an AGE vertex (``{...}::vertex``)."""

    id: int
    label: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class AgeEdge:
    """Decoded representation of an AGE edge (``{...}::edge``)."""

    id: int
    label: str
    start_id: int
    end_id: int
    properties: dict[str, Any]


@dataclass(frozen=True)
class AgePath:
    """Decoded representation of an AGE path (alternating vertices and edges)."""

    elements: tuple[AgeVertex | AgeEdge, ...]


def validate_graph_name(name: str) -> str:
    """Validate that ``name`` is safe to interpolate as an AGE graph name.

    AGE graph names cannot be parameterized in the ``cypher()`` SQL call, so
    they are interpolated directly. This check guarantees the value is a
    conservative identifier (letters, digits, underscores) and rejects anything
    that could break out of the surrounding single-quoted string.
    """
    if not _GRAPH_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid AGE graph name '{name}': must match {_GRAPH_NAME_RE.pattern}"
        )
    return name


def sanitize_identifier(name: str) -> str:
    """Return a deterministic identifier safe for use as a Cypher label or key.

    Non-alphanumeric characters are escaped as ``_u<codepoint_hex>_`` so the
    resulting identifier always starts with a letter, contains only ASCII
    letters, digits, and underscores, and can be losslessly reversed by
    :func:`desanitize_identifier`.
    """
    return _SANITIZE_PREFIX + "".join(
        c if c.isalnum() else f"_u{ord(c):x}_" for c in name
    )


def desanitize_identifier(sanitized_name: str) -> str:
    """Reverse :func:`sanitize_identifier`."""
    return re.sub(
        r"_u([0-9a-fA-F]+)_",
        lambda match: chr(int(match[1], 16)),
        sanitized_name.removeprefix(_SANITIZE_PREFIX),
    )


def age_value_from_python(value: Any) -> Any:
    """Convert a Python property value into something JSON/agtype-encodable.

    ``datetime`` values are normalized to UTC and emitted as ISO-8601 strings
    so that lexical and chronological ordering coincide. Lists recurse. All
    other primitives pass through unchanged. Accepts ``Any`` because callers
    pass through free-form values (embedding lists, filter literals, mixed
    map values) and the function is intentionally polymorphic.
    """
    if isinstance(value, _dt.datetime):
        tzinfo = value.tzinfo
        if tzinfo is None:
            value = value.replace(tzinfo=_dt.UTC)
        return value.astimezone(_dt.UTC).isoformat()
    if isinstance(value, list):
        return [age_value_from_python(item) for item in value]
    return value


def age_value_to_python(value: Any) -> Any:
    """Inverse of :func:`age_value_from_python` for values read from AGE.

    Strings that look like ISO-8601 datetimes are parsed back into aware
    ``datetime`` objects; other values are returned as-is.
    """
    if isinstance(value, str) and _looks_like_iso_datetime(value):
        try:
            parsed = _dt.datetime.fromisoformat(value)
        except ValueError:
            return value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.UTC)
        return parsed
    if isinstance(value, list):
        return [age_value_to_python(item) for item in value]
    return value


_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:[Zz]|[+-]\d{2}:?\d{2})?$"
)


def _looks_like_iso_datetime(value: str) -> bool:
    return bool(_ISO_DATETIME_RE.match(value))


def encode_agtype_params(params: Mapping[str, Any] | None) -> str:
    """JSON-encode a parameter map so it can be bound as an ``agtype`` argument.

    Cypher parameters are passed to ``cypher()`` as a single ``agtype`` value.
    ``agtype`` accepts JSON object literals for map parameters, so we emit
    standard JSON. Datetime values are normalized to ISO-8601 UTC strings via
    :func:`age_value_from_python`.

    ``allow_nan=True`` is deliberate: Python's ``json`` emits ``NaN``,
    ``Infinity``, and ``-Infinity`` for non-finite floats, and :func:`parse_agtype`
    accepts the same spellings, so the round trip stays symmetric.
    """
    if params is None:
        return "{}"
    prepared = {key: _prepare_for_json(value) for key, value in params.items()}
    return json.dumps(prepared, allow_nan=True, separators=(",", ":"))


def _prepare_for_json(value: Any) -> Any:
    if isinstance(value, _dt.datetime):
        return age_value_from_python(value)
    if isinstance(value, list):
        return [_prepare_for_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _prepare_for_json(item) for key, item in value.items()}
    return value


# Dollar-quoted delimiter used to wrap the Cypher body. Distinct from the empty
# ``$$`` delimiter so that doubled dollar signs cannot accidentally terminate
# the string literal from inside the Cypher text.
_CYPHER_DOLLAR_TAG = "$cypher_body$"


def build_cypher_call(
    graph_name: str,
    cypher_body: str,
    *,
    returns: Iterable[str] = ("v",),
    has_params: bool = False,
) -> str:
    """Build the ``SELECT ... FROM cypher(...)`` wrapper that AGE requires.

    Args:
        graph_name: The AGE graph name. Must pass :func:`validate_graph_name`.
        cypher_body: The Cypher statement to execute.
        returns: Names of the returned columns; each is typed as ``agtype``.
        has_params: Whether the caller binds a parameter map as ``$1``.

    Returns:
        A SQL statement that can be executed via SQLAlchemy or asyncpg.
    """
    validate_graph_name(graph_name)
    if _CYPHER_DOLLAR_TAG in cypher_body:
        raise ValueError(
            "Cypher body contains the reserved dollar-quoted delimiter; "
            "either escape or rename the delimiter in the source statement."
        )

    return_columns = ", ".join(f"{name} agtype" for name in returns)
    if not return_columns:
        raise ValueError("at least one return column is required")

    params_slot = ", $1::agtype" if has_params else ""
    return (
        f"SELECT * FROM cypher('{graph_name}', "
        f"{_CYPHER_DOLLAR_TAG}{cypher_body}{_CYPHER_DOLLAR_TAG}"
        f"{params_slot}"
        f") AS ({return_columns})"
    )


def setup_age_sync_connection(dbapi_connection: Any) -> None:
    """Run the per-session setup required for AGE on a DB-API connection.

    Designed to be registered on a SQLAlchemy engine's ``connect`` event so the
    extension is loaded and the search path is set before any Cypher query
    runs. The caller supplies the raw DB-API connection; this function opens
    and closes its own cursor.

    The AGE extension must be installed (via ``CREATE EXTENSION age``) before
    ``LOAD 'age'`` can succeed. On the very first connection to a fresh
    database, the extension has not yet been installed, so we probe
    ``pg_extension`` first and skip the LOAD in that case — the later
    ``_ensure_graph_initialized`` step runs ``CREATE EXTENSION`` explicitly,
    and subsequent pooled connections see the extension and load it normally.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'age'")
        has_age = cursor.fetchone() is not None
        if has_age:
            cursor.execute("LOAD 'age'")
        cursor.execute('SET search_path = ag_catalog, "$user", public')
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# agtype parsing
# ---------------------------------------------------------------------------

_AGTYPE_SUFFIXES = ("::vertex", "::edge", "::path", "::numeric")


def parse_agtype(text: str | bytes | None) -> Any:
    """Parse an ``agtype`` text value into native Python data.

    The parser accepts the JSON grammar extended with the ``::vertex``,
    ``::edge``, ``::path``, and ``::numeric`` type annotations that AGE emits
    on its custom types. Vertex and edge objects are returned as
    :class:`AgeVertex` / :class:`AgeEdge` instances; paths as :class:`AgePath`.
    """
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    parser = _AgtypeParser(text)
    value = parser.parse_value()
    parser.skip_ws()
    if not parser.at_end():
        raise AgtypeParseError(
            f"unexpected trailing data at position {parser.pos}: {text!r}"
        )
    return value


class _AgtypeParser:
    """Minimal recursive-descent parser for the ``agtype`` text grammar."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        if self.pos >= len(self.text):
            raise AgtypeParseError("unexpected end of input")
        return self.text[self.pos]

    def _consume(self, literal: str) -> bool:
        if self.text.startswith(literal, self.pos):
            self.pos += len(literal)
            return True
        return False

    def _require(self, literal: str) -> None:
        if not self._consume(literal):
            raise AgtypeParseError(f"expected {literal!r} at position {self.pos}")

    def parse_value(self) -> Any:
        self.skip_ws()
        value = self._parse_raw_value()
        self.skip_ws()
        value = self._maybe_apply_suffix(value)
        return value

    def _maybe_apply_suffix(self, value: Any) -> Any:
        for suffix in _AGTYPE_SUFFIXES:
            if self._consume(suffix):
                return _coerce_typed_value(suffix, value)
        return value

    def _parse_raw_value(self) -> Any:  # noqa: C901
        if self.at_end():
            raise AgtypeParseError("unexpected end of input")
        ch = self._peek()
        if ch == "{":
            return self._parse_object()
        if ch == "[":
            return self._parse_array()
        if ch == '"':
            return self._parse_string()
        if ch == "-" or ch.isdigit():
            return self._parse_number()
        if self._consume("true"):
            return True
        if self._consume("false"):
            return False
        if self._consume("null"):
            return None
        # Non-JSON numeric literals that AGE may emit.
        for token, replacement in (
            ("NaN", float("nan")),
            ("Infinity", float("inf")),
            ("-Infinity", float("-inf")),
        ):
            if self._consume(token):
                return replacement
        raise AgtypeParseError(f"unexpected character {ch!r} at position {self.pos}")

    def _parse_object(self) -> dict[str, Any]:
        self._require("{")
        self.skip_ws()
        result: dict[str, Any] = {}
        if self._consume("}"):
            return result
        while True:
            self.skip_ws()
            key = self._parse_string()
            self.skip_ws()
            self._require(":")
            value = self.parse_value()
            result[key] = value
            self.skip_ws()
            if self._consume(","):
                continue
            self._require("}")
            return result

    def _parse_array(self) -> list[Any]:
        self._require("[")
        self.skip_ws()
        result: list[Any] = []
        if self._consume("]"):
            return result
        while True:
            value = self.parse_value()
            result.append(value)
            self.skip_ws()
            if self._consume(","):
                continue
            self._require("]")
            return result

    def _parse_string(self) -> str:
        self._require('"')
        chars: list[str] = []
        while True:
            if self.at_end():
                raise AgtypeParseError("unterminated string literal")
            ch = self.text[self.pos]
            self.pos += 1
            if ch == '"':
                return "".join(chars)
            if ch == "\\":
                chars.append(self._parse_string_escape())
                continue
            chars.append(ch)

    def _parse_string_escape(self) -> str:  # noqa: C901
        if self.at_end():
            raise AgtypeParseError("unterminated escape sequence")
        ch = self.text[self.pos]
        self.pos += 1
        if ch == '"':
            return '"'
        if ch == "\\":
            return "\\"
        if ch == "/":
            return "/"
        if ch == "b":
            return "\b"
        if ch == "f":
            return "\f"
        if ch == "n":
            return "\n"
        if ch == "r":
            return "\r"
        if ch == "t":
            return "\t"
        if ch == "u":
            if self.pos + 4 > len(self.text):
                raise AgtypeParseError("truncated unicode escape")
            hex_value = self.text[self.pos : self.pos + 4]
            self.pos += 4
            try:
                return chr(int(hex_value, 16))
            except ValueError as err:
                raise AgtypeParseError(
                    f"invalid unicode escape \\u{hex_value}"
                ) from err
        raise AgtypeParseError(f"invalid escape character \\{ch}")

    def _parse_number(self) -> int | float:
        start = self.pos
        if self._consume("-"):
            pass
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        is_float = False
        if self.pos < len(self.text) and self.text[self.pos] == ".":
            is_float = True
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        if self.pos < len(self.text) and self.text[self.pos] in "eE":
            is_float = True
            self.pos += 1
            if self.pos < len(self.text) and self.text[self.pos] in "+-":
                self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        literal = self.text[start : self.pos]
        try:
            return float(literal) if is_float else int(literal)
        except ValueError as err:
            raise AgtypeParseError(f"invalid number literal {literal!r}") from err


def _coerce_typed_value(suffix: str, value: Any) -> Any:
    if suffix == "::vertex":
        if not isinstance(value, dict):
            raise AgtypeParseError("expected object before ::vertex annotation")
        return AgeVertex(
            id=int(value.get("id", 0)),
            label=str(value.get("label", "")),
            properties=dict(value.get("properties", {})),
        )
    if suffix == "::edge":
        if not isinstance(value, dict):
            raise AgtypeParseError("expected object before ::edge annotation")
        return AgeEdge(
            id=int(value.get("id", 0)),
            label=str(value.get("label", "")),
            start_id=int(value.get("start_id", 0)),
            end_id=int(value.get("end_id", 0)),
            properties=dict(value.get("properties", {})),
        )
    if suffix == "::path":
        if not isinstance(value, list):
            raise AgtypeParseError("expected array before ::path annotation")
        return AgePath(elements=tuple(value))
    if suffix == "::numeric":
        # ``::numeric`` keeps arbitrary-precision decimals as their string form;
        # we defer to the caller to interpret it rather than losing precision.
        return value
    raise AgtypeParseError(f"unknown agtype suffix {suffix!r}")


# ---------------------------------------------------------------------------
# Comparison rendering
# ---------------------------------------------------------------------------


def render_comparison(
    left: str,
    op: str,
    right: str,
    value: PropertyValue,
) -> str:
    """Render a Cypher comparison clause for a property value.

    AGE stores datetimes as ISO-8601 strings, so lexical and chronological
    ordering coincide and a plain operator is sufficient. The ``!=`` form is
    normalized to ``<>`` to match the Cypher grammar used elsewhere.
    """
    if op == "!=":
        op = "<>"
    if isinstance(value, list):
        raise TypeError(f"'{op}' comparison cannot accept list values")
    return f"{left} {op} {right}"
