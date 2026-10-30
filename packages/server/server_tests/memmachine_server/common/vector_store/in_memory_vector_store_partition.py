"""In-memory VectorStorePartition implementation for testing."""

import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from uuid import UUID

from memmachine_server.common.data_types import (
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
)
from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.common.vector_store import (
    VectorStorePartition,
    VectorStoreSchema,
)
from memmachine_server.common.vector_store.data_types import (
    QueryMatch,
    QueryResult,
    Record,
)
from memmachine_server.common.vector_store.utils import (
    bind_filter,
    bind_record,
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
        case Ordering(field, op, value):
            held = properties.get(field)
            if held is None or not _same_type(held, value):
                return False
            return bool(_ORDERING_OPS[op](_comparable(held), _comparable(value)))
        case In(field, values):
            held = properties.get(field)
            return any(
                held is not None and _same_type(held, value) and held == value
                for value in values
            )
        case IsNull(field):
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


def _score(metric: SimilarityMetric, a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the similarity/distance score for the given metric."""
    match metric:
        case SimilarityMetric.COSINE:
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return _dot(a, b) / (norm_a * norm_b)
        case SimilarityMetric.DOT:
            return _dot(a, b)
        case SimilarityMetric.EUCLIDEAN:
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
        case SimilarityMetric.MANHATTAN:
            return sum(abs(x - y) for x, y in zip(a, b, strict=True))


def _passes_threshold(score: float, threshold: float, higher_is_better: bool) -> bool:
    """Check whether a score passes the threshold for the metric direction."""
    if higher_is_better:
        return score >= threshold
    return score <= threshold


# ---------------------------------------------------------------------------
# InMemoryVectorStorePartition
# ---------------------------------------------------------------------------


class InMemoryVectorStorePartition(VectorStorePartition):
    """In-memory VectorStorePartition for testing.

    Supports all similarity metrics (cosine, dot, euclidean, manhattan),
    evaluates FilterExpr on record properties, and enforces the declared
    schema the way a real store does.
    """

    def __init__(
        self,
        *,
        store_schema: VectorStoreSchema,
    ) -> None:
        self._store_schema = store_schema
        self.records: dict[UUID, Record] = {}

    @property
    def store_schema(self) -> VectorStoreSchema:
        return self._store_schema

    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = [
            bind_record(record, self._store_schema.indexed_properties)
            for record in records
        ]
        for record in records:
            self.records[record.uuid] = Record(
                uuid=record.uuid,
                vector=list(record.vector),
                properties=dict(record.properties),
            )

    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        score_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        if property_filter is not None:
            property_filter = bind_filter(
                property_filter, self._store_schema.indexed_properties
            )

        metric = self._store_schema.similarity_metric
        higher_is_better = metric.higher_is_better

        results: list[QueryResult] = []
        for query_vector in query_vectors:
            qv = list(query_vector)
            matches: list[QueryMatch] = []
            for record in self.records.values():
                if property_filter is not None and not evaluate_filter(
                    property_filter, record.properties
                ):
                    continue
                score = _score(metric, qv, record.vector)
                if score_threshold is not None and not _passes_threshold(
                    score, score_threshold, higher_is_better
                ):
                    continue
                matches.append(QueryMatch(score=score, record_uuid=record.uuid))
            matches.sort(key=lambda m: m.score, reverse=higher_is_better)
            if limit is not None:
                matches = matches[:limit]
            results.append(QueryResult(matches=matches))
        return results

    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        for uid in record_uuids:
            self.records.pop(uid, None)
