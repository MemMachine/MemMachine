"""Module for parsing filter strings into dictionaries."""

import re
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from memmachine.common.data_types import FilterablePropertyValue


class FilterParseError(ValueError):
    """Raised when the textual filter specification is invalid."""


class FilterExpr(Protocol):
    """Marker protocol for filter expression nodes."""


@dataclass(frozen=True)
class Comparison(FilterExpr):
    """Filter comparison of a field against a value or list of values."""

    field: str
    op: str  # "=", "in", "is_null", "is_not_null"
    value: FilterablePropertyValue | list[FilterablePropertyValue]


@dataclass(frozen=True)
class And(FilterExpr):
    """Logical conjunction of two filter expressions."""

    left: FilterExpr
    right: FilterExpr


@dataclass(frozen=True)
class Or(FilterExpr):
    """Logical disjunction of two filter expressions."""

    left: FilterExpr
    right: FilterExpr


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
    ("EQ", r"="),
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
            if tok.type == "AND":
                expr = And(left=expr, right=rhs)
            else:
                expr = Or(left=expr, right=rhs)

        return expr

    def _parse_primary(self) -> FilterExpr:
        if self._accept("LPAREN"):
            expr = self._parse_expression()
            self._expect("RPAREN")
            return expr
        return self._parse_comparison()

    def _parse_comparison(self) -> FilterExpr:
        field_tok = self._expect("IDENT")
        field = field_tok.value

        if self._accept("EQ"):
            # field = value
            value = self._parse_value()
            return Comparison(field=field, op="=", value=value)

        if self._accept("IN"):
            self._expect("LPAREN")
            values: list[FilterablePropertyValue] = []
            values.append(self._parse_value())
            while self._accept("COMMA"):
                values.append(self._parse_value())
            self._expect("RPAREN")
            return Comparison(field=field, op="in", value=values)

        if self._accept("IS"):
            negate = self._accept("NOT") is not None
            null_tok = self._expect("IDENT")
            if null_tok.value.upper() != "NULL":
                raise FilterParseError(
                    "Expected NULL after IS/IS NOT. NOT doesn't support other operations as of now",
                )
            op = "is_not_null" if negate else "is_null"
            return Comparison(field=field, op=op, value=None)

        raise FilterParseError(f"Expected '=' or IN after field {field}")

    def _parse_value(self) -> FilterablePropertyValue:
        tok = self._expect("IDENT", "STRING")
        raw = tok.value
        # If it's a string literal, return it as-is
        if tok.type == "STRING":
            return raw
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
) -> dict[str, FilterablePropertyValue] | None:
    """Convert a filter expression into a legacy equality mapping."""
    if expr is None:
        return None

    comparisons = _flatten_conjunction(expr)
    if not comparisons:
        return None

    property_filter: dict[str, FilterablePropertyValue] = {}
    for comp in comparisons:
        if comp.op != "=":
            op_name = "IN" if comp.op == "in" else comp.op
            raise TypeError(
                f"Legacy property filters only support '=' comparisons, not {op_name}",
            )
        value = comp.value
        if isinstance(value, list):
            raise TypeError(
                "Legacy property filters do not support 'IN' values",
            )
        property_filter[comp.field] = value
    return property_filter


def _flatten_conjunction(expr: FilterExpr) -> list[Comparison]:
    if isinstance(expr, Comparison):
        return [expr]
    if isinstance(expr, And):
        flattened: list[Comparison] = []
        flattened.extend(_flatten_conjunction(expr.left))
        flattened.extend(_flatten_conjunction(expr.right))
        return flattened
    raise TypeError(
        "Legacy property filters only support AND expressions made of simple comparisons",
    )
