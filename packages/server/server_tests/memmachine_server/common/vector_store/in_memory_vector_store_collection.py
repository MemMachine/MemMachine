"""In-memory VectorStoreCollection implementation for testing."""

import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import override
from uuid import UUID

from memmachine_server.common.data_types import PropertyType, PropertyValue
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
)
from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.common.vector_store import VectorStoreCollection
from memmachine_server.common.vector_store.data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.declared_properties import (
    require_declared_properties,
    require_supported_filter,
)

# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------

_ORDERING_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def _comparable(value: PropertyValue) -> PropertyValue:
    return ensure_tz_aware(value) if isinstance(value, datetime) else value


def _same_type(held: PropertyValue, value: PropertyValue) -> bool:
    # A predicate matches only a value of the compared type; `bool` is an
    # `int` at runtime and its own type here.
    return type(held) is type(value)


def evaluate_filter(expr: FilterExpr, properties: Mapping[str, PropertyValue]) -> bool:
    """Evaluate a FilterExpr against a properties mapping, by the language's semantics."""
    match expr:
        case Equals(field, value):
            held = properties.get(field)
            return (
                held is not None
                and _same_type(held, value)
                and _comparable(held) == _comparable(value)
            )
        case NotEquals(field, value):
            held = properties.get(field)
            return (
                held is not None
                and _same_type(held, value)
                and _comparable(held) != _comparable(value)
            )
        case Ordering(field, op, value):
            held = properties.get(field)
            if held is None or not _same_type(held, value):
                return False
            return bool(_ORDERING_OPS[op](_comparable(held), _comparable(value)))
        case In(field, values):
            held = properties.get(field)
            return held is not None and _same_type(held, values[0]) and held in values
        case IsMissing(field):
            return field not in properties
        case And(operands):
            return all(evaluate_filter(o, properties) for o in operands)
        case Or(operands):
            return any(evaluate_filter(o, properties) for o in operands)
        case Not(operand):
            return not evaluate_filter(operand, properties)


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors; 0.0 if either has no magnitude."""
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(a, b) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# InMemoryVectorStoreCollection
# ---------------------------------------------------------------------------


class InMemoryVectorStoreCollection(VectorStoreCollection):
    """In-memory VectorStoreCollection for testing.

    Scores by cosine similarity, evaluates FilterExpr on record properties,
    and enforces the declared schema the way a real store does.
    """

    _SUPPORTED_FILTER_NODES = frozenset(
        {Equals, NotEquals, Ordering, In, IsMissing, And, Or, Not}
    )

    def __init__(
        self,
        collection_config: VectorStoreCollectionConfig,
        indexed_properties: Mapping[str, PropertyType],
        *,
        supported_filter_nodes: Iterable[type] | None = None,
    ) -> None:
        self.collection_config = collection_config
        self._indexed_properties = dict(indexed_properties)
        self._supported_filter_nodes = (
            frozenset(supported_filter_nodes)
            if supported_filter_nodes is not None
            else InMemoryVectorStoreCollection._SUPPORTED_FILTER_NODES
        )
        self.records: dict[UUID, Record] = {}
        self.queries: list[FilterExpr | None] = []
        """The property filter of each query, in order, for tests that assert routing."""

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self.collection_config

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return self._supported_filter_nodes

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = list(records)
        for record in records:
            require_declared_properties(record.properties, self._indexed_properties)
        for record in records:
            self.records[record.uuid] = Record(
                uuid=record.uuid,
                vector=list(record.vector),
                properties=dict(record.properties),
            )

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        min_cosine_similarity: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        if property_filter is not None:
            require_supported_filter(
                property_filter, self._indexed_properties, self._supported_filter_nodes
            )
        self.queries.append(property_filter)
        results: list[QueryResult] = []
        for query_vector in query_vectors:
            qv = list(query_vector)
            matches: list[QueryMatch] = []
            for record in self.records.values():
                if property_filter is not None and not evaluate_filter(
                    property_filter, record.properties
                ):
                    continue
                cosine_similarity = _cosine_similarity(qv, record.vector)
                if (
                    min_cosine_similarity is not None
                    and cosine_similarity < min_cosine_similarity
                ):
                    continue
                matches.append(
                    QueryMatch(
                        cosine_similarity=cosine_similarity,
                        record_uuid=record.uuid,
                    )
                )
            matches.sort(key=lambda m: m.cosine_similarity, reverse=True)
            results.append(QueryResult(matches=matches[:limit]))
        return results

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        for uid in record_uuids:
            self.records.pop(uid, None)
