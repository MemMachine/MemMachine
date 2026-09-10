"""Helpers for building filter nodes in parametrized tests."""

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter import (
    Equals,
    FilterExpr,
    NotEquals,
    Ordering,
)


def comparison(field: str, op: str, value: PropertyValue) -> FilterExpr:
    """The node a textual comparison operator denotes."""
    match op:
        case "=":
            return Equals(field=field, value=value)
        case "!=":
            return NotEquals(field=field, value=value)
        case ">" | "<" | ">=" | "<=":
            if isinstance(value, bool | str):
                raise TypeError(f"{op} does not order {type(value).__name__}")
            return Ordering(field=field, op=op, value=value)
        case _:
            raise ValueError(f"Unknown comparison operator {op!r}")
