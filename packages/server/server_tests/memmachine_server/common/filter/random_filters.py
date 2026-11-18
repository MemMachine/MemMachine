"""Random filters and records, for checking filter rewrites against an evaluator."""

import random
from collections.abc import Sequence

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)

VALUES = (0, 1, 2, 3)
_OPS = ("=", "!=", ">", "<", ">=", "<=")


def random_filter(rng: random.Random, fields: Sequence[str], depth: int) -> FilterExpr:
    """A filter tree up to `depth` levels deep over `fields`, with every kind of node."""
    if depth == 0 or rng.random() < 0.25:
        field = rng.choice(fields)
        match rng.randrange(3):
            case 0:
                return Comparison(
                    field=field, op=rng.choice(_OPS), value=rng.choice(VALUES)
                )
            case 1:
                return In(field=field, values=rng.sample(VALUES, rng.randint(1, 3)))
            case _:
                return IsNull(field=field)
    match rng.randrange(3):
        case 0:
            return And(
                left=random_filter(rng, fields, depth - 1),
                right=random_filter(rng, fields, depth - 1),
            )
        case 1:
            return Or(
                left=random_filter(rng, fields, depth - 1),
                right=random_filter(rng, fields, depth - 1),
            )
        case _:
            return Not(expr=random_filter(rng, fields, depth - 1))


def random_properties(
    rng: random.Random, fields: Sequence[str]
) -> dict[str, PropertyValue]:
    """Values from `VALUES` for `fields`, each missing one time in five."""
    return {field: rng.choice(VALUES) for field in fields if rng.random() < 0.8}
