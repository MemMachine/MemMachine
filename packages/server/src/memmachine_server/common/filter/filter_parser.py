"""
The server's textual filter language, parsed into filter expression trees.

The HTTP API still speaks this language, so the parser stays as the server's
translation into the tree in `filter_expression`; the tree is what every
store compiles, and nothing below the API boundary parses text.
"""

import re
from datetime import datetime
from typing import NamedTuple, cast

from memmachine_server.common.data_types import OrderedValue, PropertyValue

from .filter_expression import (
    And,
    Equals,
    FilterExpr,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
    OrderingOp,
)


class FilterParseError(ValueError):
    """Raised when the textual filter specification is invalid."""


class Token(NamedTuple):
    """Token emitted by the lexer while parsing filter strings."""

    type: str
    value: str


_OP_PRECEDENCE = {
    "OR": 1,
    "AND": 2,
}


_TOKEN_SPEC = [
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("NE", r"!=|<>"),
    ("GE", r">="),
    ("LE", r"<="),
    ("EQ", r"="),
    ("GT", r">"),
    ("LT", r"<"),
    ("STRING", r"'[^']*'"),
    ("IDENT", r"[A-Za-z0-9_\.]+"),
    ("WS", r"\s+"),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC)
)


def _tokenize(s: str) -> list[Token]:
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        if kind is None:
            continue
        value = m.group()
        if kind == "WS":
            continue
        if kind == "STRING":
            # Strip quotes from string literals
            tokens.append(Token("STRING", value[1:-1]))
        elif kind == "IDENT":
            upper = value.upper()
            if upper in ("AND", "OR", "IN", "IS", "NOT"):
                tokens.append(Token(upper, upper))
            else:
                tokens.append(Token("IDENT", value))
        else:
            tokens.append(Token(kind, value))
    return tokens


_ORDERING_OPS: dict[str, OrderingOp] = {
    "GE": ">=",
    "LE": "<=",
    "GT": ">",
    "LT": "<",
}


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _accept(self, *types: str) -> Token | None:
        tok = self._peek()
        if tok and tok.type in types:
            self.pos += 1
            return tok
        return None

    def _expect(self, *types: str) -> Token:
        tok = self._peek()
        if not tok or tok.type not in types:
            expected = " or ".join(types)
            actual = tok.type if tok else "EOF"
            raise FilterParseError(f"Expected {expected}, got {actual}")
        self.pos += 1
        return tok

    def parse(self) -> FilterExpr | None:
        if not self.tokens:
            return None
        expr = self._parse_expression()
        if self._peek() is not None:
            raise FilterParseError(f"Unexpected token: {self._peek()}")
        return expr

    def _parse_expression(self, min_prec: int = 1) -> FilterExpr:
        expr = self._parse_primary()

        while True:
            tok = self._peek()
            if not tok or tok.type not in _OP_PRECEDENCE:
                break

            prec = _OP_PRECEDENCE[tok.type]
            if prec < min_prec:
                break

            self.pos += 1
            rhs = self._parse_expression(prec + 1)
            # A chain of one operator is one n-ary node, so the tree's shape
            # does not depend on how the text happened to group it.
            if tok.type == "AND":
                left = expr.operands if isinstance(expr, And) else (expr,)
                expr = And((*left, rhs))
            else:
                left = expr.operands if isinstance(expr, Or) else (expr,)
                expr = Or((*left, rhs))

        return expr

    def _parse_primary(self) -> FilterExpr:
        if self._accept("LPAREN"):
            expr = self._parse_expression()
            self._expect("RPAREN")
            return expr
        if self._accept("NOT"):
            return Not(self._parse_primary())
        return self._parse_predicate()

    def _parse_predicate(self) -> FilterExpr:
        # Note: NOT IN has two grammar entry points:
        # - "NOT field IN (...)" is handled in _parse_primary via Not(parse_primary())
        # - "field NOT IN (...)" is handled below as a special case
        # Both produce Not(In(...)).
        field_tok = self._expect("IDENT")
        field = field_tok.value

        if self._accept("EQ"):
            return Equals(field, self._parse_value())
        if self._accept("NE"):
            return NotEquals(field, self._parse_value())
        op_tok = self._accept("GE", "LE", "GT", "LT")
        if op_tok:
            return Ordering(field, _ORDERING_OPS[op_tok.type], self._parse_ordered())

        # IN / NOT IN
        negate = self._accept("NOT") is not None
        if self._accept("IN"):
            values = self._parse_value_list()
            expr: FilterExpr = In(field, values)
            return Not(expr) if negate else expr
        if negate:
            raise FilterParseError(f"Expected IN after NOT for field {field}")

        # IS NULL / IS NOT NULL
        if self._accept("IS"):
            negate = self._accept("NOT") is not None
            null_tok = self._expect("IDENT")
            if null_tok.value.upper() != "NULL":
                raise FilterParseError(
                    "Expected NULL after IS/IS NOT",
                )
            expr = IsMissing(field)
            return Not(expr) if negate else expr

        raise FilterParseError(
            f"Expected operator after field {field}: "
            "comparison (=, !=, <>, >, <, >=, <=), "
            "membership (IN, NOT IN), or nullity (IS NULL, IS NOT NULL)"
        )

    def _parse_value_list(self) -> tuple[int, ...] | tuple[str, ...]:
        self._expect("LPAREN")
        raw: list[int | str] = [self._parse_in_value()]
        while self._accept("COMMA"):
            raw.append(self._parse_in_value())
        self._expect("RPAREN")
        if all(isinstance(v, int) for v in raw):
            return tuple(cast(list[int], raw))
        if all(isinstance(v, str) for v in raw):
            return tuple(cast(list[str], raw))
        raise FilterParseError(
            "Mixed types in IN list: all values must be int or all str"
        )

    def _parse_in_value(self) -> int | str:
        """Parse a single value inside an IN list (only int and str allowed)."""
        value = self._parse_value()
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise FilterParseError(
                f"IN lists only support int and str values, got {type(value).__name__}"
            )
        return value

    def _parse_ordered(self) -> OrderedValue:
        """Parse the value of an ordering comparison: a number or a date."""
        value = self._parse_value()
        if isinstance(value, bool) or not isinstance(value, int | float | datetime):
            raise FilterParseError(
                "Ordering comparisons (>, <, >=, <=) take a number or a date(), "
                f"got {type(value).__name__}"
            )
        return value

    def _parse_value(self) -> PropertyValue:
        tok = self._expect("IDENT", "STRING")
        raw = tok.value
        # If it's a string literal, return it as-is
        if tok.type == "STRING":
            return raw
        # Check for date() function call
        if tok.type == "IDENT" and raw.lower() == "date":
            self._expect("LPAREN")
            date_str_tok = self._expect("STRING")
            self._expect("RPAREN")
            try:
                return datetime.fromisoformat(date_str_tok.value)
            except ValueError as e:
                raise FilterParseError(f"Invalid ISO format date string: {e}") from e
        # Otherwise, parse IDENT for boolean/numeric values
        upper = raw.upper()
        if upper == "TRUE":
            return True
        if upper == "FALSE":
            return False
        if raw.isdigit():
            return int(raw)
        if _looks_like_float(raw):
            return float(raw)
        return raw


def _looks_like_float(value: str) -> bool:
    if value.count(".") != 1:
        return False
    left, right = value.split(".")
    return bool(left) and bool(right) and left.isdigit() and right.isdigit()


# ---------------------------------------------------------------------------
# Centralized field name normalization utilities
# ---------------------------------------------------------------------------

# Query language prefixes for user-defined metadata
USER_METADATA_QUERY_PREFIXES = ("m.", "metadata.")

# Internal storage prefix for user metadata
USER_METADATA_STORAGE_PREFIX = "metadata."


def normalize_filter_field(field: str) -> tuple[str, bool]:
    """Normalize a query field name to internal storage name.

    Returns (internal_name, is_user_metadata).
    - User metadata (m.foo, metadata.foo): returns ("metadata.foo", True)
    - System field (producer_id): returns ("producer_id", False)
    """
    for prefix in USER_METADATA_QUERY_PREFIXES:
        if field.startswith(prefix):
            key = field.removeprefix(prefix)
            return f"{USER_METADATA_STORAGE_PREFIX}{key}", True
    return field, False


def mangle_user_metadata_key(key: str) -> str:
    """Add user metadata prefix to a key for storage."""
    return USER_METADATA_STORAGE_PREFIX + key


def demangle_user_metadata_key(mangled_key: str) -> str:
    """Remove user metadata prefix from a storage key."""
    return mangled_key.removeprefix(USER_METADATA_STORAGE_PREFIX)


def is_user_metadata_key(candidate_key: str) -> bool:
    """Check if a key has the user metadata prefix."""
    return candidate_key.startswith(USER_METADATA_STORAGE_PREFIX)


def parse_filter(spec: str | None) -> FilterExpr | None:
    """Parse the given textual filter specification."""
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    tokens = _tokenize(spec)
    return _Parser(tokens).parse()


def to_property_filter(
    expr: FilterExpr | None,
) -> dict[str, PropertyValue | None] | None:
    """Convert a filter expression into a legacy equality mapping."""
    if expr is None:
        return None

    equalities = _flatten_conjunction(expr)
    if not equalities:
        return None

    return {equality.field: equality.value for equality in equalities}


def _flatten_conjunction(expr: FilterExpr) -> list[Equals]:
    if isinstance(expr, Equals):
        return [expr]
    if isinstance(expr, And):
        return [
            equality
            for operand in expr.operands
            for equality in _flatten_conjunction(operand)
        ]
    raise TypeError(
        "Legacy property filters only support AND expressions made of simple "
        "equality comparisons",
    )
