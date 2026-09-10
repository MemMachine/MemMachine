"""
Filter expression trees.

A filter expression is built from the nodes below and compiled by each store
into its own query language. `FilterExpr` is a closed union, so a compiler
written as a `match` with no default arm is checked for exhaustiveness:
adding a node here fails type checking in every store rather than raising on
whichever query first reaches the new node.

A predicate matches a record only when the field holds a value of the
compared type. A record that does not carry the field, or carries it with a
different type, is not a match -- which is what `NotEquals` and
`Not(Equals(...))` distinguish: the former keeps only records that hold a
comparable, differing value, the latter also keeps records that hold no
comparable value at all.

A datetime value denotes an instant, and a naive one means UTC. A node
normalizes its datetime to a UTC-aware instant at construction, so every
compiler receives instants and only chooses a representation.
"""

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from memmachine_server.common.data_types import OrderedValue, PropertyValue
from memmachine_server.common.utils import ensure_tz_aware

type FilterExpr = Equals | NotEquals | Ordering | In | IsMissing | And | Or | Not
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
class NotEquals:
    """Field holds a comparable value that differs from `value`."""

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
    excluded rather than treated as integers. At least one value is required:
    a caller with nothing to admit has no query to make.
    """

    field: str
    values: tuple[int, ...] | tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject value lists a store cannot compare as a single type."""
        if not self.values:
            raise ValueError(f"In({self.field!r}) requires at least one value")
        if any(isinstance(value, bool) for value in self.values):
            raise TypeError(f"In({self.field!r}) values must be int or str, not bool")
        if len({type(value) for value in self.values}) > 1:
            raise TypeError(
                f"In({self.field!r}) values must all be int or all be str, got "
                f"{sorted({type(value).__name__ for value in self.values})}"
            )


@dataclass(frozen=True)
class IsMissing:
    """
    Field holds no comparable value.

    True for a record that does not carry the field at all. Property values
    are never null, so absence is the only way a field can hold nothing.
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
        case NotEquals(field, value):
            return NotEquals(transform(field), value)
        case Ordering(field, op, value):
            return Ordering(transform(field), op, value)
        case In(field, values):
            return In(transform(field), values)
        case IsMissing(field):
            return IsMissing(transform(field))
        case And(operands):
            return And(tuple(map_filter_fields(o, transform) for o in operands))
        case Or(operands):
            return Or(tuple(map_filter_fields(o, transform) for o in operands))
        case Not(operand):
            return Not(map_filter_fields(operand, transform))


def filter_fields(expr: FilterExpr) -> frozenset[str]:
    """Every field name a filter tree addresses."""
    match expr:
        case (
            Equals(field)
            | NotEquals(field)
            | Ordering(field)
            | In(field)
            | IsMissing(field)
        ):
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
        case Equals() | NotEquals() | Ordering() | In() | IsMissing():
            return frozenset((type(expr),))


def conjoin(clauses: Iterable[FilterExpr | None]) -> FilterExpr | None:
    """The conjunction of the given clauses; None when there are none."""
    operands = tuple(clause for clause in clauses if clause is not None)
    if not operands:
        return None
    if len(operands) == 1:
        return operands[0]
    return And(operands)


def conjuncts(expr: FilterExpr | None) -> list[FilterExpr]:
    """The top-level conjuncts of a tree, with nested conjunctions flattened."""
    if expr is None:
        return []
    if isinstance(expr, And):
        return [conjunct for o in expr.operands for conjunct in conjuncts(o)]
    return [expr]


def split_declared(
    expr: FilterExpr | None, declared: Collection[str]
) -> tuple[FilterExpr | None, FilterExpr | None]:
    """
    Split a tree into the conjuncts naming declared fields only, and the rest.

    A conjunct is declared when every field it names is in `declared`; a
    disjunction or negation that mixes declared and undeclared fields is
    undeclared as a whole, since no part of it can be evaluated alone.
    """
    declared_part: list[FilterExpr] = []
    undeclared_part: list[FilterExpr] = []
    for conjunct in conjuncts(expr):
        if filter_fields(conjunct) <= frozenset(declared):
            declared_part.append(conjunct)
        else:
            undeclared_part.append(conjunct)
    return conjoin(declared_part), conjoin(undeclared_part)
