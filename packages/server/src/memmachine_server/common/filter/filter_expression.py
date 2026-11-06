"""
Filter expression trees.

A filter expression is built from the nodes below and compiled by each store
into its own query language. `FilterExpr` is a closed union, so a compiler
written as a `match` with no default arm is checked for exhaustiveness:
adding a node here fails type checking in every store rather than raising on
whichever query first reaches the new node.

A leaf compares a field, a reference rather than a name: a `SystemField`,
which the consuming system defines, or a `Property`, by key.

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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from memmachine_server.common.data_types import (
    OrderedValue,
    PropertyType,
    PropertyValue,
)
from memmachine_server.common.utils import ensure_tz_aware

type FilterExpr = Equals | Ordering | In | IsNull | And | Or | Not
"""Any node of a filter expression tree."""

OrderingOp = Literal[">", "<", ">=", "<="]


def _utc_instant(value: datetime) -> datetime:
    return ensure_tz_aware(value).astimezone(UTC)


@dataclass(frozen=True)
class SystemField:
    """A field the consuming system defines, by name."""

    name: str


@dataclass(frozen=True)
class Property:
    """
    A property, by key.

    `type` is the key's declared type where a schema declares one, else None.
    """

    key: str
    type: PropertyType | None = None


@dataclass(frozen=True)
class Equals:
    """Field holds a value equal to `value`."""

    field: SystemField | Property
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

    field: SystemField | Property
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

    Values are ints or strs, which every store matches exactly; booleans are
    excluded rather than treated as integers, and a float or datetime field
    takes no membership test. Values are homogeneous because a store indexes
    and compares a property by its type; a mixed list has no single type to
    compare against. No values admits nothing: every compiler renders an
    empty `In` as false, so a caller whose set of admitted values came out
    empty can still say so.
    """

    field: SystemField | Property
    values: tuple[int, ...] | tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject value lists a store cannot compare as a single type, and a float or datetime field."""
        if any(type(value) not in (int, str) for value in self.values):
            raise TypeError(
                f"In({self.field!r}) values must be int or str, got "
                f"{sorted({type(value).__name__ for value in self.values})}"
            )
        if len({type(value) for value in self.values}) > 1:
            raise TypeError(
                f"In({self.field!r}) values must all be int or all be str, got "
                f"{sorted({type(value).__name__ for value in self.values})}"
            )
        if isinstance(self.field, Property) and self.field.type in (float, datetime):
            raise TypeError(f"In({self.field!r}) takes no float or datetime field")


@dataclass(frozen=True)
class IsNull:
    """
    Field holds no value.

    True for a record that does not carry the field. Property values are
    never null, so a missing key is the only no-value state.
    """

    field: SystemField | Property


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
    transform: Callable[[SystemField | Property], SystemField | Property],
) -> FilterExpr:
    """Apply a field transformation to every field in a filter tree."""
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


def filter_fields(expr: FilterExpr) -> frozenset[SystemField | Property]:
    """Every field a filter tree addresses."""
    match expr:
        case Equals(field) | Ordering(field) | In(field) | IsNull(field):
            return frozenset((field,))
        case Not(operand):
            return filter_fields(operand)
        case And(operands) | Or(operands):
            return frozenset(field for o in operands for field in filter_fields(o))


def extract_filter(
    expr: FilterExpr, *, allowed_fields: AbstractSet[SystemField | Property]
) -> FilterExpr | None:
    """The part of a filter that names only the allowed fields; None when no part does.

    Every record that matches the filter matches its part.
    """
    return _extract_filter(expr, allowed_fields, negated=False)


def _extract_filter(
    expr: FilterExpr,
    allowed_fields: AbstractSet[SystemField | Property],
    *,
    negated: bool,
) -> FilterExpr | None:
    """The part of a filter under an odd number of negations when `negated`.

    The roles then swap: every record that matches the part matches the
    filter, so the negation around the part is again the wider one.
    """
    match expr:
        case Equals(field) | Ordering(field) | In(field) | IsNull(field):
            return expr if field in allowed_fields else None
        case Not(operand):
            part = _extract_filter(operand, allowed_fields, negated=not negated)
            return None if part is None else Not(part)
        case And(operands) | Or(operands):
            parts = [
                _extract_filter(operand, allowed_fields, negated=negated)
                for operand in operands
            ]
            kept = tuple(part for part in parts if part is not None)
            if len(kept) == len(parts):
                return type(expr)(kept)
            # Leaving out an operand widens an And and narrows an Or; under a
            # negation, the reverse.
            if isinstance(expr, And) == negated or not kept:
                return None
            return kept[0] if len(kept) == 1 else type(expr)(kept)
