"""
The server's textual filter language, parsed into filter expression trees.

The HTTP API still speaks this language, so the parser stays as the server's
translation into the tree in `filter_expression`; the tree is what every
store compiles, and nothing below the API boundary parses text. The parser
resolves each field name once: `m.<key>` and `metadata.<key>` name the
property `<key>`, and any other name is a system field. It types each
literal compared with a system field of a static type by that type; a
literal compared with a property keeps its own type.
"""

import re
from collections.abc import Mapping
from datetime import datetime
from typing import NamedTuple, cast

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    OrderedValue,
    PropertyType,
    PropertyValue,
)

from .filter_expression import (
    And,
    Equals,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
    OrderingOp,
    Property,
    SystemField,
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
    def __init__(
        self, tokens: list[Token], system_fields: Mapping[str, PropertyType]
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.system_fields = system_fields

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
        name = self._expect("IDENT").value
        field = _field_reference(name)
        declared = None if isinstance(field, Property) else self.system_fields.get(name)

        if self._accept("EQ"):
            return Equals(field, _typed(name, declared, self._parse_value()))
        if self._accept("NE"):
            return Not(Equals(field, _typed(name, declared, self._parse_value())))
        op_tok = self._accept("GE", "LE", "GT", "LT")
        if op_tok:
            return Ordering(
                field,
                _ORDERING_OPS[op_tok.type],
                _typed(name, declared, self._parse_ordered()),
            )

        # IN / NOT IN
        negate = self._accept("NOT") is not None
        if self._accept("IN"):
            if declared is float or declared is datetime:
                raise FilterParseError(
                    f"Field {name} is of type "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared]}, "
                    "which IN does not compare"
                )
            values = self._parse_value_list()
            for value in values:
                _typed(name, declared, value)
            expr: FilterExpr = In(field, values)
            return Not(expr) if negate else expr
        if negate:
            raise FilterParseError(f"Expected IN after NOT for field {name}")

        # IS NULL / IS NOT NULL
        if self._accept("IS"):
            negate = self._accept("NOT") is not None
            null_tok = self._expect("IDENT")
            if null_tok.value.upper() != "NULL":
                raise FilterParseError(
                    "Expected NULL after IS/IS NOT",
                )
            expr = IsNull(field)
            return Not(expr) if negate else expr

        raise FilterParseError(
            f"Expected operator after field {name}: "
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


def _typed[V: PropertyValue](
    name: str, declared: PropertyType | None, value: V
) -> V | float:
    """The literal as a field of the declared type holds it; a field of no static type takes any literal."""
    if declared is None:
        return value
    if declared is float and type(value) is int:
        return float(value)
    # `bool` is an `int` at runtime and is its own property type here.
    if type(value) is not declared:
        written = " (written date('...'))" if declared is datetime else ""
        raise FilterParseError(
            f"Field {name} is of type {PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared]}"
            f"{written}, got {type(value).__name__} {value!r}"
        )
    return value


_PROPERTY_PREFIXES = ("m.", "metadata.")


def _field_reference(name: str) -> SystemField | Property:
    """The field a name denotes: a property after a property prefix, else a system field."""
    for prefix in _PROPERTY_PREFIXES:
        if name.startswith(prefix):
            return Property(name.removeprefix(prefix))
    return SystemField(name)


def _looks_like_float(value: str) -> bool:
    if value.count(".") != 1:
        return False
    left, right = value.split(".")
    return bool(left) and bool(right) and left.isdigit() and right.isdigit()


def parse_filter(
    spec: str | None, *, system_fields: Mapping[str, PropertyType]
) -> FilterExpr | None:
    """
    Parse a textual filter specification for a system of the given fields.

    `system_fields` maps the name of each system field of a static type to
    that type. A literal compared with one is of its type: an int for a
    float field is the float it equals, and any other literal of another
    type, or a membership test on a float or datetime field, raises
    FilterParseError.
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    tokens = _tokenize(spec)
    return _Parser(tokens, system_fields).parse()


def to_property_filter(
    expr: FilterExpr | None,
) -> dict[str, PropertyValue | None] | None:
    """Convert a filter expression into a legacy equality mapping, keyed by each field's name or key."""
    if expr is None:
        return None

    equalities = _flatten_conjunction(expr)
    if not equalities:
        return None

    mapping: dict[str, PropertyValue | None] = {}
    for equality in equalities:
        match equality.field:
            case SystemField(name=key) | Property(key=key):
                mapping[key] = equality.value
    return mapping


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
