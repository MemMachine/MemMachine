"""Random filters and records, for checking filter rewrites against an evaluator."""

import random
from collections.abc import Sequence

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter import (
    And,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Property,
)

from .nodes import comparison

VALUES = (0, 1, 2, 3)
_OPS = ("=", "!=", ">", "<", ">=", "<=")


def random_filter(rng: random.Random, keys: Sequence[str], depth: int) -> FilterExpr:
    """A filter tree up to `depth` levels deep over the properties of `keys`, with every kind of node."""
    if depth == 0 or rng.random() < 0.25:
        field = Property(rng.choice(keys))
        match rng.randrange(3):
            case 0:
                return comparison(field, rng.choice(_OPS), rng.choice(VALUES))
            case 1:
                return In(
                    field=field, values=tuple(rng.sample(VALUES, rng.randint(1, 3)))
                )
            case _:
                return IsNull(field=field)
    match rng.randrange(3):
        case 0:
            return And(
                tuple(
                    random_filter(rng, keys, depth - 1)
                    for _ in range(rng.randint(2, 3))
                )
            )
        case 1:
            return Or(
                tuple(
                    random_filter(rng, keys, depth - 1)
                    for _ in range(rng.randint(2, 3))
                )
            )
        case _:
            return Not(random_filter(rng, keys, depth - 1))


def random_properties(
    rng: random.Random, keys: Sequence[str]
) -> dict[str, PropertyValue]:
    """Values from `VALUES` for `keys`, each missing one time in five."""
    return {key: rng.choice(VALUES) for key in keys if rng.random() < 0.8}
