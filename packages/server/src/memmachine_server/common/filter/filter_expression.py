"""
Filter expression trees.

A filter expression is built from the nodes below and compiled by each store
into its own query language. `FilterExpr` is a closed union, so a compiler
written as a `match` with no default arm is checked for exhaustiveness:
adding a node here fails type checking in every store rather than raising on
whichever query first reaches the new node.

The grammar is two-valued. A predicate matches a record only when the field
holds a value of the compared type; a record that does not carry the field,
or carries it with a different type, is not a match, and `IsNull` on such a
field is true. Nothing is ever unknown, so `And`, `Or` and `Not` are the
ordinary boolean connectives over the records: `Not` is the complement of a
match and keeps the records holding no comparable value, and `!=` is
`Not(Equals(...))` rather than a node of its own. A caller who wants only
records holding some other value conjoins `Not(IsNull(...))`.

A datetime value denotes an instant, and a naive one means UTC. A node
normalizes its datetime to a UTC-aware instant at construction, so every
compiler receives instants and only chooses a representation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from memmachine_server.common.data_types import OrderedValue, PropertyValue
from memmachine_server.common.utils import ensure_tz_aware

type FilterExpr = Equals | Ordering | In | IsNull | And | Or | Not
"""Any node of a filter expression tree."""

OrderingOp = Literal[">", "<", ">=", "<="]


def _utc_instant(value: datetime) -> datetime:
    return ensure_tz_aware(value).astimezone(UTC)


@dataclass(frozen=True)
class Equals:
    """Field holds a value equal to `value`."""

    field: str
    value: PropertyValue

    def __post_init__(self) -> None:
        """Normalize a datetime value to a UTC-aware instant."""
        if isinstance(self.value, datetime):
            object.__setattr__(self, "value", _utc_instant(self.value))


@dataclass(frozen=True)
class Ordering:
    """
    Field holds a value ordered against `value`.

    Only values with a total order are comparable, so `bool` and `str` are not
    accepted: ordering booleans is meaningless, and ordering strings is a
    lexicographic comparison whose result depends on how a store happens to
    encode the value. Equality on those types is expressed with `Equals`.
    """

    field: str
    op: OrderingOp
    value: OrderedValue

    def __post_init__(self) -> None:
        """Reject a boolean, which `int` admits at runtime, and normalize a datetime."""
        if isinstance(self.value, bool):
            raise TypeError(f"Ordering({self.field!r}) value must not be a bool")
        if isinstance(self.value, datetime):
            object.__setattr__(self, "value", _utc_instant(self.value))


@dataclass(frozen=True)
class In:
    """
    Field holds a value among `values`.

    Values are homogeneous because a store indexes and compares a property by
    its type; a mixed list has no single type to compare against. Booleans are
    excluded rather than treated as integers. No values admits nothing: every
    compiler renders an empty `In` as false, so a caller whose set of admitted
    values came out empty can still say so.
    """

    field: str
    values: tuple[int, ...] | tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject value lists a store cannot compare as a single type."""
        if any(isinstance(value, bool) for value in self.values):
            raise TypeError(f"In({self.field!r}) values must be int or str, not bool")
        if len({type(value) for value in self.values}) > 1:
            raise TypeError(
                f"In({self.field!r}) values must all be int or all be str, got "
                f"{sorted({type(value).__name__ for value in self.values})}"
            )


@dataclass(frozen=True)
class IsNull:
    """
    Field holds no value.

    True for a record that does not carry the field. Property values are
    never null, so a missing key is the only no-value state.
    """

    field: str


@dataclass(frozen=True)
class And:
    """
    All operands match.

    At least one operand is required: an empty conjunction would oblige every
    store to render an identity element, and callers already spell "no filter"
    as `None`.
    """

    operands: tuple[FilterExpr, ...]

    def __post_init__(self) -> None:
        """Reject an empty conjunction."""
        if not self.operands:
            raise ValueError("And requires at least one operand")


@dataclass(frozen=True)
class Or:
    """
    At least one operand matches.

    At least one operand is required, for the reason given on `And`.
    """

    operands: tuple[FilterExpr, ...]

    def __post_init__(self) -> None:
        """Reject an empty disjunction."""
        if not self.operands:
            raise ValueError("Or requires at least one operand")


@dataclass(frozen=True)
class Not:
    """The operand does not match."""

    operand: FilterExpr


def map_filter_fields(
    expr: FilterExpr,
    transform: Callable[[str], str],
) -> FilterExpr:
    """Apply a field name transformation to every field in a filter tree."""
    match expr:
        case Equals(field, value):
            return Equals(transform(field), value)
        case Ordering(field, op, value):
            return Ordering(transform(field), op, value)
        case In(field, values):
            return In(transform(field), values)
        case IsNull(field):
            return IsNull(transform(field))
        case And(operands):
            return And(tuple(map_filter_fields(o, transform) for o in operands))
        case Or(operands):
            return Or(tuple(map_filter_fields(o, transform) for o in operands))
        case Not(operand):
            return Not(map_filter_fields(operand, transform))


def filter_fields(expr: FilterExpr) -> frozenset[str]:
    """Every field name a filter tree addresses."""
    match expr:
        case Equals(field) | Ordering(field) | In(field) | IsNull(field):
            return frozenset((field,))
        case Not(operand):
            return filter_fields(operand)
        case And(operands) | Or(operands):
            return frozenset(field for o in operands for field in filter_fields(o))


def filter_nodes(expr: FilterExpr) -> frozenset[type]:
    """The node classes a filter tree is built from."""
    match expr:
        case Not(operand):
            return frozenset((Not,)) | filter_nodes(operand)
        case And(operands) | Or(operands):
            return frozenset((type(expr),)).union(*(filter_nodes(o) for o in operands))
        case Equals() | Ordering() | In() | IsNull():
            return frozenset((type(expr),))
