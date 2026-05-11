"""In-memory VectorStoreCollection implementation for testing."""

import math
import operator
from collections.abc import Iterable, Sequence
from uuid import UUID

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.vector_store import VectorStoreCollection
from memmachine_server.common.vector_store.data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionConfig,
)

# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------

_COMPARISON_OPS = {
    "=": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def _evaluate_comparison(prop: PropertyValue, op: str, value: PropertyValue) -> bool:
    fn = _COMPARISON_OPS.get(op)
    if fn is None:
        raise ValueError(f"Unknown comparison op: {op!r}")
    return bool(fn(prop, value))


def evaluate_filter(expr: FilterExpr, properties: dict[str, PropertyValue]) -> bool:
    """Evaluate a FilterExpr against a properties dict."""
    match expr:
        case Comparison(field=field, op=op, value=value):
            prop = properties.get(field)
            if prop is None:
                return False
            return _evaluate_comparison(prop, op, value)
        case In(field=field, values=values):
            return properties.get(field) in values
        case IsNull(field=field):
            return field not in properties
        case And(left=left, right=right):
            return evaluate_filter(left, properties) and evaluate_filter(
                right, properties
            )
        case Or(left=left, right=right):
            return evaluate_filter(left, properties) or evaluate_filter(
                right, properties
            )
        case Not(expr=inner):
            return not evaluate_filter(inner, properties)
        case _:
            raise TypeError(f"Unknown filter expression type: {type(expr)}")


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
# InMemoryVectorStoreCollection
# ---------------------------------------------------------------------------


class InMemoryVectorStoreCollection(VectorStoreCollection):
    """In-memory VectorStoreCollection for testing.

    Supports all similarity metrics (cosine, dot, euclidean, manhattan)
    and full FilterExpr evaluation on record properties.
    """

    def __init__(self, collection_config: VectorStoreCollectionConfig) -> None:
        self.collection_config = collection_config
        self.records: dict[UUID, Record] = {}

    @property
    def config(self) -> VectorStoreCollectionConfig:
        return self.collection_config

    async def upsert(self, *, records: Iterable[Record]) -> None:
        for record in records:
            self.records[record.uuid] = Record(
                uuid=record.uuid,
                vector=list(record.vector) if record.vector is not None else None,
                properties=dict(record.properties) if record.properties else {},
            )

    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        score_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        metric = self.collection_config.similarity_metric
        higher_is_better = metric.higher_is_better

        results: list[QueryResult] = []
        for query_vector in query_vectors:
            qv = list(query_vector)
            matches: list[QueryMatch] = []
            for record in self.records.values():
                if record.vector is None:
                    continue
                if property_filter is not None and not evaluate_filter(
                    property_filter, record.properties or {}
                ):
                    continue
                score = _score(metric, qv, record.vector)
                if score_threshold is not None and not _passes_threshold(
                    score, score_threshold, higher_is_better
                ):
                    continue
                matches.append(
                    QueryMatch(
                        score=score,
                        record=self._project_record(
                            record, return_vector, return_properties
                        ),
                    )
                )
            matches.sort(key=lambda m: m.score, reverse=higher_is_better)
            if limit is not None:
                matches = matches[:limit]
            results.append(QueryResult(matches=matches))
        return results

    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        out: list[Record] = []
        for uid in record_uuids:
            record = self.records.get(uid)
            if record is None:
                continue
            out.append(self._project_record(record, return_vector, return_properties))
        return out

    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        for uid in record_uuids:
            self.records.pop(uid, None)

    @staticmethod
    def _project_record(
        record: Record, return_vector: bool, return_properties: bool
    ) -> Record:
        """Return a copy of the record with only the requested fields."""
        return Record(
            uuid=record.uuid,
            vector=(
                list(record.vector)
                if return_vector and record.vector is not None
                else None
            ),
            properties=(
                dict(record.properties)
                if return_properties and record.properties
                else None
            ),
        )
