"""Qdrant-based vector store implementation."""

import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf
from qdrant_client import AsyncQdrantClient, models

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.metrics_factory import MetricsFactory

from .data_types import PropertyValue, QueryResult, Record
from .vector_store import Collection, VectorStore

_SIMILARITY_METRIC_MAP: dict[SimilarityMetric, models.Distance] = {
    SimilarityMetric.COSINE: models.Distance.COSINE,
    SimilarityMetric.DOT: models.Distance.DOT,
    SimilarityMetric.EUCLIDEAN: models.Distance.EUCLID,
    SimilarityMetric.MANHATTAN: models.Distance.MANHATTAN,
}

_PROPERTY_TYPE_TO_INDEX: dict[type[PropertyValue], models.PayloadSchemaType] = {
    bool: models.PayloadSchemaType.BOOL,
    int: models.PayloadSchemaType.INTEGER,
    float: models.PayloadSchemaType.FLOAT,
    str: models.PayloadSchemaType.KEYWORD,
    datetime: models.PayloadSchemaType.DATETIME,
}

_RANGE_OPS = {
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
}


def _build_qdrant_filter(expr: FilterExpr) -> models.Filter:
    """Convert a FilterExpr tree into a Qdrant Filter."""
    if isinstance(expr, FilterComparison):
        return _build_qdrant_comparison(expr)
    if isinstance(expr, FilterIn):
        return _in_filter(expr.field, expr.values)
    if isinstance(expr, FilterIsNull):
        return _null_filter(expr.field, negate=False)
    if isinstance(expr, FilterNot):
        return models.Filter(must_not=[_build_qdrant_filter(expr.expr)])
    if isinstance(expr, FilterAnd):
        left = _build_qdrant_filter(expr.left)
        right = _build_qdrant_filter(expr.right)
        return models.Filter(must=[left, right])
    if isinstance(expr, FilterOr):
        left = _build_qdrant_filter(expr.left)
        right = _build_qdrant_filter(expr.right)
        return models.Filter(should=[left, right])
    msg = f"Unsupported filter expression type: {type(expr)}"
    raise TypeError(msg)


def _to_match_value(value: PropertyValue) -> int | str:
    """Convert a filter value to a Qdrant MatchValue-compatible type."""
    if isinstance(value, datetime):
        msg = "datetime cannot be used with MatchValue; use a DatetimeRange filter"
        raise TypeError(msg)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    # float
    if value == int(value):
        return int(value)
    msg = (
        f"Non-integer float {value} cannot be used with MatchValue; use a range filter"
    )
    raise TypeError(msg)


def _to_range_value(value: PropertyValue) -> int | float | str:
    """Convert a filter value for Qdrant Range."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _build_qdrant_comparison(comp: FilterComparison) -> models.Filter:
    """Convert a Comparison into a Qdrant Filter."""
    field = comp.field
    op = comp.op
    value = comp.value

    if op in ("=", "!="):
        negate = op == "!="
        if isinstance(value, float) and value != int(value):
            return _float_eq_filter(field, value, negate=negate)
        if isinstance(value, datetime):
            return _datetime_eq_filter(field, value, negate=negate)
        return _match_filter(field, value, negate=negate)
    if op in _RANGE_OPS:
        return _range_filter(field, value, _RANGE_OPS[op])

    msg = f"Unsupported filter operator: {op}"
    raise ValueError(msg)


def _match_filter(
    field: str,
    value: PropertyValue,
    *,
    negate: bool,
) -> models.Filter:
    cond = models.FieldCondition(
        key=field,
        match=models.MatchValue(value=_to_match_value(value)),
    )
    if negate:
        return models.Filter(must_not=[cond])
    return models.Filter(must=[cond])


def _float_eq_filter(field: str, value: float, *, negate: bool) -> models.Filter:
    """Use a range filter for float equality since MatchValue doesn't accept floats."""
    cond = models.FieldCondition(
        key=field,
        range=models.Range(gte=value, lte=value),
    )
    if negate:
        return models.Filter(must_not=[cond])
    return models.Filter(must=[cond])


def _datetime_eq_filter(field: str, value: datetime, *, negate: bool) -> models.Filter:
    """Use a DatetimeRange filter for datetime equality since MatchValue doesn't accept datetimes."""
    cond = models.FieldCondition(
        key=field,
        range=models.DatetimeRange(gte=value, lte=value),
    )
    if negate:
        return models.Filter(must_not=[cond])
    return models.Filter(must=[cond])


def _in_filter(field: str, value: list[int] | list[str]) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=field,
                match=models.MatchAny(any=value),
            ),
        ],
    )


def _range_filter(
    field: str,
    value: PropertyValue,
    param: str,
) -> models.Filter:
    if isinstance(value, datetime):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    range=models.DatetimeRange(**{param: value}),
                ),
            ],
        )
    range_val = _to_range_value(value)
    return models.Filter(
        must=[
            models.FieldCondition(
                key=field,
                range=models.Range(**{param: range_val}),  # type: ignore[arg-type]
            ),
        ],
    )


def _null_filter(field: str, *, negate: bool) -> models.Filter:
    cond = models.IsNullCondition(
        is_null=models.PayloadField(key=field),
    )
    if negate:
        return models.Filter(must_not=[cond])
    return models.Filter(must=[cond])


class QdrantVectorStoreParams(BaseModel):
    """Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        user_metrics_labels (dict[str, str]):
            Labels to attach to the collected metrics
            (default: {}).

    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels to attach to the collected metrics",
    )


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore."""

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncQdrantClient = params.client
        self._properties_schemas: dict[str, Mapping[str, type[PropertyValue]]] = {}

        self._create_collection_calls_counter = None
        self._create_collection_latency_summary = None
        self._delete_collection_calls_counter = None
        self._delete_collection_latency_summary = None
        self._upsert_calls_counter = None
        self._upsert_latency_summary = None
        self._query_calls_counter = None
        self._query_latency_summary = None
        self._get_calls_counter = None
        self._get_latency_summary = None
        self._delete_calls_counter = None
        self._delete_latency_summary = None

        self._should_collect_metrics = False
        metrics_factory = params.metrics_factory
        if metrics_factory is not None:
            self._should_collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
            label_names = self._user_metrics_labels.keys()

            self._create_collection_calls_counter = metrics_factory.get_counter(
                "vector_store_qdrant_create_collection_calls",
                "Number of calls to create_collection in QdrantVectorStore",
                label_names=label_names,
            )
            self._create_collection_latency_summary = metrics_factory.get_summary(
                "vector_store_qdrant_create_collection_latency_seconds",
                "Latency in seconds for create_collection in QdrantVectorStore",
                label_names=label_names,
            )
            self._delete_collection_calls_counter = metrics_factory.get_counter(
                "vector_store_qdrant_delete_collection_calls",
                "Number of calls to delete_collection in QdrantVectorStore",
                label_names=label_names,
            )
            self._delete_collection_latency_summary = metrics_factory.get_summary(
                "vector_store_qdrant_delete_collection_latency_seconds",
                "Latency in seconds for delete_collection in QdrantVectorStore",
                label_names=label_names,
            )
            self._upsert_calls_counter = metrics_factory.get_counter(
                "vector_store_qdrant_upsert_calls",
                "Number of calls to upsert in QdrantVectorStore",
                label_names=label_names,
            )
            self._upsert_latency_summary = metrics_factory.get_summary(
                "vector_store_qdrant_upsert_latency_seconds",
                "Latency in seconds for upsert in QdrantVectorStore",
                label_names=label_names,
            )
            self._query_calls_counter = metrics_factory.get_counter(
                "vector_store_qdrant_query_calls",
                "Number of calls to query in QdrantVectorStore",
                label_names=label_names,
            )
            self._query_latency_summary = metrics_factory.get_summary(
                "vector_store_qdrant_query_latency_seconds",
                "Latency in seconds for query in QdrantVectorStore",
                label_names=label_names,
            )
            self._get_calls_counter = metrics_factory.get_counter(
                "vector_store_qdrant_get_calls",
                "Number of calls to get in QdrantVectorStore",
                label_names=label_names,
            )
            self._get_latency_summary = metrics_factory.get_summary(
                "vector_store_qdrant_get_latency_seconds",
                "Latency in seconds for get in QdrantVectorStore",
                label_names=label_names,
            )
            self._delete_calls_counter = metrics_factory.get_counter(
                "vector_store_qdrant_delete_calls",
                "Number of calls to delete in QdrantVectorStore",
                label_names=label_names,
            )
            self._delete_latency_summary = metrics_factory.get_summary(
                "vector_store_qdrant_delete_latency_seconds",
                "Latency in seconds for delete in QdrantVectorStore",
                label_names=label_names,
            )

    async def startup(self) -> None:
        """No-op; client is ready on construction."""

    async def shutdown(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()

    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """Create a collection in the Qdrant vector store."""
        start_time = time.monotonic()

        distance = _SIMILARITY_METRIC_MAP[similarity_metric]
        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dimensions,
                distance=distance,
            ),
        )

        if properties_schema:
            for prop_name, prop_type in properties_schema.items():
                index_type = _PROPERTY_TYPE_TO_INDEX.get(prop_type)
                if index_type is not None:
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=prop_name,
                        field_schema=index_type,
                    )
            self._properties_schemas[collection_name] = properties_schema

        end_time = time.monotonic()
        self._collect_metrics(
            self._create_collection_calls_counter,
            self._create_collection_latency_summary,
            start_time,
            end_time,
        )

    async def get_collection(self, collection_name: str) -> "QdrantCollection":
        """Get a collection handle from the vector store."""
        return QdrantCollection(store=self, collection_name=collection_name)

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the Qdrant vector store."""
        start_time = time.monotonic()

        await self._client.delete_collection(collection_name=collection_name)
        self._properties_schemas.pop(collection_name, None)

        end_time = time.monotonic()
        self._collect_metrics(
            self._delete_collection_calls_counter,
            self._delete_collection_latency_summary,
            start_time,
            end_time,
        )

    def _collect_metrics(
        self,
        calls_counter: MetricsFactory.Counter | None,
        latency_summary: MetricsFactory.Summary | None,
        start_time: float,
        end_time: float,
    ) -> None:
        """Increment calls and observe latency."""
        if self._should_collect_metrics:
            cast(MetricsFactory.Counter, calls_counter).increment(
                labels=self._user_metrics_labels,
            )
            cast(MetricsFactory.Summary, latency_summary).observe(
                value=end_time - start_time,
                labels=self._user_metrics_labels,
            )

    # ── Internal helpers for QdrantCollection ──

    def get_properties_schema(
        self,
        collection_name: str,
    ) -> Mapping[str, type[PropertyValue]]:
        """Return the cached properties schema for a collection."""
        return self._properties_schemas.get(collection_name, {})

    async def upsert_points(
        self,
        collection_name: str,
        points: list[models.PointStruct],
    ) -> None:
        """Upsert points into a collection and collect metrics."""
        start_time = time.monotonic()

        if points:
            await self._client.upsert(
                collection_name=collection_name,
                points=points,
            )

        end_time = time.monotonic()
        self._collect_metrics(
            self._upsert_calls_counter,
            self._upsert_latency_summary,
            start_time,
            end_time,
        )

    async def query_points(
        self,
        collection_name: str,
        *,
        query: list[float],
        query_filter: models.Filter | None,
        score_threshold: float | None,
        limit: int,
        with_vectors: bool,
        with_payload: bool,
    ) -> list[models.ScoredPoint]:
        """Query points from a collection and collect metrics."""
        start_time = time.monotonic()

        results = await self._client.query_points(
            collection_name=collection_name,
            query=query,
            query_filter=query_filter,
            score_threshold=score_threshold,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._query_calls_counter,
            self._query_latency_summary,
            start_time,
            end_time,
        )

        return list(results.points)

    async def retrieve_points(
        self,
        collection_name: str,
        *,
        ids: list[str],
        with_vectors: bool,
        with_payload: bool,
    ) -> list[models.Record]:
        """Retrieve points from a collection and collect metrics."""
        start_time = time.monotonic()

        points = await self._client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._get_calls_counter,
            self._get_latency_summary,
            start_time,
            end_time,
        )

        return list(points)

    async def delete_points(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """Delete points from a collection and collect metrics."""
        start_time = time.monotonic()

        if ids:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=cast("list[int | str | UUID]", ids),
                ),
            )

        end_time = time.monotonic()
        self._collect_metrics(
            self._delete_calls_counter,
            self._delete_latency_summary,
            start_time,
            end_time,
        )


class QdrantCollection(Collection):
    """A collection backed by Qdrant."""

    def __init__(
        self,
        *,
        store: QdrantVectorStore,
        collection_name: str,
    ) -> None:
        """Initialize with a back-reference to the store and collection name."""
        self._store = store
        self._collection_name = collection_name

    def _serialize_payload(
        self,
        properties: dict[str, PropertyValue] | None,
    ) -> dict[str, bool | int | float | str | None]:
        """Serialize record properties to a Qdrant-compatible payload."""
        if not properties:
            return {}
        payload: dict[str, bool | int | float | str | None] = {}
        for key, value in properties.items():
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
            else:
                payload[key] = value
        return payload

    def _deserialize_payload(
        self,
        payload: dict[str, object] | None,
    ) -> dict[str, PropertyValue] | None:
        """Deserialize a Qdrant payload back to record properties."""
        if not payload:
            return None
        schema = self._store.get_properties_schema(self._collection_name)
        result: dict[str, PropertyValue] = {}
        for key, value in payload.items():
            if schema.get(key) is datetime and isinstance(value, str):
                result[key] = datetime.fromisoformat(value)
            else:
                result[key] = cast(PropertyValue, value)
        return result

    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        points = [
            models.PointStruct(
                id=str(record.uuid),
                vector=record.vector if record.vector is not None else [],
                payload=self._serialize_payload(record.properties),
            )
            for record in records
        ]

        await self._store.upsert_points(self._collection_name, points)

    async def query(
        self,
        *,
        query_vector: Sequence[float],
        similarity_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[QueryResult]:
        """Query for records matching the criteria by vector similarity."""
        qdrant_filter = (
            _build_qdrant_filter(property_filter) if property_filter else None
        )

        effective_limit = limit if limit is not None else 10000

        scored_points = await self._store.query_points(
            self._collection_name,
            query=list(query_vector),
            query_filter=qdrant_filter,
            score_threshold=similarity_threshold,
            limit=effective_limit,
            with_vectors=return_vector,
            with_payload=return_properties,
        )

        query_results: list[QueryResult] = []
        for point in scored_points:
            vector: list[float] | None = None
            if return_vector and point.vector is not None:
                vector = cast(list[float], point.vector)

            properties: dict[str, PropertyValue] | None = None
            if return_properties and point.payload is not None:
                properties = self._deserialize_payload(
                    cast(dict[str, object], point.payload),
                )

            query_results.append(
                QueryResult(
                    score=cast(float, point.score),
                    record=Record(
                        uuid=UUID(point.id)
                        if isinstance(point.id, str)
                        else UUID(int=cast(int, point.id)),
                        vector=vector,
                        properties=properties,
                    ),
                ),
            )

        return query_results

    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records from the collection by their UUIDs."""
        uuid_list = list(record_uuids)
        if not uuid_list:
            return []

        points = await self._store.retrieve_points(
            self._collection_name,
            ids=[str(u) for u in uuid_list],
            with_vectors=return_vector,
            with_payload=return_properties,
        )

        points_by_id: dict[str, models.Record] = {str(p.id): p for p in points}

        records: list[Record] = []
        for uid in uuid_list:
            point = points_by_id.get(str(uid))
            if point is None:
                continue

            vector: list[float] | None = None
            if return_vector and point.vector is not None:
                vector = cast(list[float], point.vector)

            properties: dict[str, PropertyValue] | None = None
            if return_properties and point.payload is not None:
                properties = self._deserialize_payload(
                    cast(dict[str, object], point.payload),
                )

            records.append(
                Record(
                    uuid=uid,
                    vector=vector,
                    properties=properties,
                ),
            )

        return records

    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """Delete records from the collection by their UUIDs."""
        uuid_list = list(record_uuids)
        await self._store.delete_points(
            self._collection_name,
            ids=[str(u) for u in uuid_list],
        )
