"""Unit tests for NebulaGraph's cosine-as-inner-product vector encoding.

These need no server: the encoding is pure, and it is what makes an IP
index answer cosine. The rest of the NebulaGraph suite is integration-only,
so without these the invariant would go unexercised everywhere.
"""

import math
import re
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("nebulagraph_python")

from nebulagraph_python.client import NebulaAsyncClient

from memmachine_server.common.vector_graph_store.nebula_graph_vector_graph_store import (
    NebulaGraphVectorGraphStore,
    NebulaGraphVectorGraphStoreParams,
)


@pytest.fixture
def store() -> NebulaGraphVectorGraphStore:
    """A store that never talks to a server; only the encoding is exercised."""
    return NebulaGraphVectorGraphStore(
        NebulaGraphVectorGraphStoreParams(
            # Never dereferenced: nothing here reaches the wire.
            client=cast(NebulaAsyncClient, object()),
            schema_name="/test_schema",
            graph_type_name="test_graph_type",
            graph_name="test_graph",
        )
    )


def _components(literal: str) -> list[float]:
    inside = literal[literal.index("([") + 2 : literal.rindex("])")]
    return [float(part) for part in inside.split(",")]


class TestUnit:
    def test_scales_to_unit_length(self):
        assert NebulaGraphVectorGraphStore._unit([3.0, 4.0]) == [0.6, 0.8]

    def test_preserves_direction(self):
        scaled = NebulaGraphVectorGraphStore._unit([2.0, 4.0, 4.0])
        assert scaled[1] == pytest.approx(scaled[2])
        assert scaled[1] == pytest.approx(2 * scaled[0])

    def test_already_unit_vector_is_unchanged(self):
        assert NebulaGraphVectorGraphStore._unit([0.0, 1.0]) == [0.0, 1.0]

    def test_zero_vector_is_left_alone(self):
        """It has no direction to preserve, and dividing would raise."""
        assert NebulaGraphVectorGraphStore._unit([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]

    def test_negative_components_survive(self):
        assert NebulaGraphVectorGraphStore._unit([-3.0, 4.0]) == [-0.6, 0.8]


class TestVectorLiteral:
    def test_normalizes_the_components(self, store: NebulaGraphVectorGraphStore):
        components = _components(store._vector_to_gql_literal([3.0, 4.0]))
        assert components == [0.6, 0.8]

    def test_declares_the_original_dimension(self, store: NebulaGraphVectorGraphStore):
        literal = store._vector_to_gql_literal([3.0, 4.0, 0.0])
        assert literal.startswith("VECTOR<3, FLOAT>(")
        assert len(_components(literal)) == 3

    def test_a_zero_vector_still_renders(self, store: NebulaGraphVectorGraphStore):
        assert _components(store._vector_to_gql_literal([0.0, 0.0])) == [0.0, 0.0]

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ([1.0, 0.0], [1.0, 0.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 0.0], [-1.0, 0.0]),
            ([3.0, 4.0], [4.0, 3.0]),
            ([2.0, 1.0, 0.5], [0.25, 8.0, 1.0]),
            # Magnitude must not leak into the score: a scaled-up vector has
            # the same direction, so it must score identically.
            ([300.0, 400.0], [4.0, 3.0]),
        ],
    )
    def test_inner_product_of_literals_is_cosine_of_the_originals(
        self, store: NebulaGraphVectorGraphStore, left: list[float], right: list[float]
    ):
        """This is the whole reason an IP index can answer a cosine query."""
        inner_product = sum(
            a * b
            for a, b in zip(
                _components(store._vector_to_gql_literal(left)),
                _components(store._vector_to_gql_literal(right)),
                strict=True,
            )
        )

        dot = sum(a * b for a, b in zip(left, right, strict=True))
        magnitudes = math.sqrt(sum(a * a for a in left)) * math.sqrt(
            sum(b * b for b in right)
        )
        assert inner_product == pytest.approx(dot / magnitudes)


def test_only_one_site_turns_a_vector_into_a_literal():
    """Stored and query sides cannot disagree while they share one encoder.

    Both write paths and both read paths call `_vector_to_gql_literal`; a
    second place building a `VECTOR<...>(...)` value would be free to skip
    normalization and silently corrupt IP ranking.
    """
    source = Path(NebulaGraphVectorGraphStore.__module__.replace(".", "/") + ".py")
    module_text = (Path("packages/server/src") / source).read_text()

    # `VECTOR<n, FLOAT>` alone is a type name (used in schema declarations);
    # only a following `(` makes it a value.
    value_literals = re.findall(r"VECTOR<[^>]*>\(", module_text)
    assert len(value_literals) == 1
