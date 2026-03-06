"""Qdrant-based vector store implementation."""

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf
from qdrant_client import AsyncQdrantClient, models

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine_server.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine_server.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine_server.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .data_types import QueryResult, Record
from .vector_store import Collection, VectorStore

_SCHEMA_METADATA_KEY = "schema"
_SIMILARITY_METRIC_METADATA_KEY = "similarity_metric"

_RANGE_OPS = {
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
}


def _build_schema_metadata(schema: Mapping[str, type[PropertyValue]]) -> dict[str, str]:
    """Build schema metadata from properties schema."""
    return {
        name: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]
        for name, property_type in schema.items()
    }


def _parse_schema_metadata(
    schema_metadata: Mapping[str, str],
) -> dict[str, type[PropertyValue]]:
    """Parse properties schema from schema metadata."""
    return {
        name: property_type
        for name, type_name in schema_metadata.items()
        if (property_type := PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE.get(type_name))
        is not None
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


def _ensure_tz_aware(dt: datetime) -> datetime:
    """Return a timezone-aware datetime, treating naive values as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


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
        if isinstance(value, float):
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
    value: bool | int | str,
    *,
    negate: bool,
) -> models.Filter:
    cond = models.FieldCondition(
        key=field,
        match=models.MatchValue(value=value),
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
    value = _ensure_tz_aware(value)
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
        value = _ensure_tz_aware(value)
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

    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore."""

    _SIMILARITY_METRIC_TO_QDRANT_DISTANCE: ClassVar[
        dict[SimilarityMetric, models.Distance]
    ] = {
        SimilarityMetric.COSINE: models.Distance.COSINE,
        SimilarityMetric.DOT: models.Distance.DOT,
        SimilarityMetric.EUCLIDEAN: models.Distance.EUCLID,
        SimilarityMetric.MANHATTAN: models.Distance.MANHATTAN,
    }

    _PROPERTY_TYPE_TO_INDEX_TYPE: ClassVar[
        dict[type[PropertyValue], models.PayloadSchemaType]
    ] = {
        bool: models.PayloadSchemaType.BOOL,
        int: models.PayloadSchemaType.INTEGER,
        float: models.PayloadSchemaType.FLOAT,
        str: models.PayloadSchemaType.KEYWORD,
        datetime: models.PayloadSchemaType.DATETIME,
    }

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncQdrantClient = params.client
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_qdrant",
        )

    @override
    async def startup(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """Create a collection in the Qdrant vector store."""
        async with self._tracker("create_collection"):
            distance = QdrantVectorStore._SIMILARITY_METRIC_TO_QDRANT_DISTANCE[
                similarity_metric
            ]
            metadata = {
                _SCHEMA_METADATA_KEY: _build_schema_metadata(properties_schema or {}),
                _SIMILARITY_METRIC_METADATA_KEY: similarity_metric.value,
            }
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimensions,
                    distance=distance,
                ),
                metadata=metadata,
            )

            if properties_schema:
                for prop_name, prop_type in properties_schema.items():
                    index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(
                        prop_type
                    )
                    if index_type is not None:
                        await self._client.create_payload_index(
                            collection_name=collection_name,
                            field_name=prop_name,
                            field_schema=index_type,
                        )

    @override
    async def get_collection(self, collection_name: str) -> "QdrantCollection":
        """Get a collection handle from the vector store."""
        return QdrantCollection(
            client=self._client,
            collection_name=collection_name,
            tracker=self._tracker,
        )

    @override
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the Qdrant vector store."""
        async with self._tracker("delete_collection"):
            await self._client.delete_collection(collection_name=collection_name)


class QdrantCollection(Collection):
    """A collection backed by Qdrant."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection_name: str,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Qdrant client and collection name."""
        self._client = client
        self._tracker = tracker
        self._collection_name = collection_name
        self._properties_schema: Mapping[str, type[PropertyValue]] | None = None

    async def _get_properties_schema(self) -> Mapping[str, type[PropertyValue]]:
        """Lazily resolve and cache properties schema from collection metadata."""
        if self._properties_schema is not None:
            return self._properties_schema

        collection_info = await self._client.get_collection(self._collection_name)

        metadata = cast(dict[str, Any], collection_info.config.metadata or {})
        schema_metadata = cast(dict[str, str], metadata.get(_SCHEMA_METADATA_KEY, {}))
        schema = _parse_schema_metadata(schema_metadata)

        self._properties_schema = schema

        return schema

    def _build_payload(
        self,
        properties: dict[str, PropertyValue] | None,
        schema: Mapping[str, type[PropertyValue]],
    ) -> dict[str, bool | int | float | str | None]:
        """Build Qdrant-compatible payload from record properties.

        Missing schema keys are stored as explicit nulls so that
        `IsNull` filters work via the index for both omitted and
        `None` values.
        """
        payload: dict[str, bool | int | float | str | None] = dict.fromkeys(schema)
        if properties:
            for key, value in properties.items():
                if value is None:
                    continue
                if isinstance(value, datetime):
                    payload[key] = _ensure_tz_aware(value)
                else:
                    payload[key] = value
        return payload

    async def _parse_payload(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, PropertyValue] | None:
        """Parse record properties from Qdrant payload."""
        if payload is None:
            return None

        schema = await self._get_properties_schema()
        result: dict[str, PropertyValue] = {}
        for key, value in payload.items():
            if schema.get(key) is datetime and isinstance(value, str):
                result[key] = datetime.fromisoformat(value)
            else:
                result[key] = cast(PropertyValue, value)
        return result

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        async with self._tracker("upsert"):
            schema = await self._get_properties_schema()
            points = [
                models.PointStruct(
                    id=record.uuid,
                    vector=record.vector if record.vector is not None else [],
                    payload=self._build_payload(record.properties, schema),
                )
                for record in records
            ]
            if points:
                await self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        score_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Iterable[QueryResult]]:
        """Query for records matching the criteria by query vectors."""
        async with self._tracker("query"):
            qdrant_filter = (
                _build_qdrant_filter(property_filter) if property_filter else None
            )

            effective_limit = limit if limit is not None else 10000

            requests = [
                models.QueryRequest(
                    query=list(qv),
                    filter=qdrant_filter,
                    score_threshold=score_threshold,
                    limit=effective_limit,
                    with_vector=return_vector,
                    with_payload=return_properties,
                )
                for qv in query_vectors
            ]

            batch_results = await self._client.query_batch_points(
                collection_name=self._collection_name,
                requests=requests,
            )

            all_query_results: list[list[QueryResult]] = []
            for batch in batch_results:
                query_results: list[QueryResult] = []
                for point in batch.points:
                    vector: list[float] | None = None
                    if return_vector and point.vector is not None:
                        vector = cast(list[float], point.vector)

                    properties: dict[str, PropertyValue] | None = None
                    if return_properties and point.payload is not None:
                        properties = await self._parse_payload(
                            cast(dict[str, Any], point.payload),
                        )

                    query_results.append(
                        QueryResult(
                            score=cast(float, point.score),
                            record=Record(
                                uuid=UUID(str(point.id)),
                                vector=vector,
                                properties=properties,
                            ),
                        ),
                    )
                all_query_results.append(query_results)

            return all_query_results

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records from the collection by their UUIDs."""
        async with self._tracker("get"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return []

            points = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=list(uuid_list),
                with_vectors=return_vector,
                with_payload=return_properties,
            )

            points_by_uuid: dict[UUID, models.Record] = {
                UUID(str(p.id)): p for p in points
            }

            records: list[Record] = []
            for point_uuid in uuid_list:
                point = points_by_uuid.get(point_uuid)
                if point is None:
                    continue

                vector: list[float] | None = None
                if return_vector and point.vector is not None:
                    vector = cast(list[float], point.vector)

                properties: dict[str, PropertyValue] | None = None
                if return_properties and point.payload is not None:
                    properties = await self._parse_payload(
                        cast(dict[str, Any] | None, point.payload),
                    )

                records.append(
                    Record(
                        uuid=point_uuid,
                        vector=vector,
                        properties=properties,
                    ),
                )

            return records

    @override
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """Delete records from the collection by their UUIDs."""
        async with self._tracker("delete"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return

            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(
                    points=list(uuid_list),
                ),
            )
