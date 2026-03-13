"""Qdrant-based vector store implementation."""

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

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

from .data_types import QueryMatch, QueryResult, Record
from .vector_store import Collection, VectorStore

_PARTITION_KEY_FIELD = "_partition_key"
_SCHEMA_METADATA_KEY = "schema"
_SIMILARITY_METRIC_METADATA_KEY = "similarity_metric"

_RANGE_OPERATORS = {
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


def _check_field_not_reserved(field: str) -> None:
    """Raise if a filter field targets the internal partition key."""
    if field == _PARTITION_KEY_FIELD:
        message = f"Filtering on reserved field '{_PARTITION_KEY_FIELD}' is not allowed"
        raise ValueError(message)


def _build_qdrant_filter(expr: FilterExpr) -> models.Filter:
    """Convert a FilterExpr tree into a Qdrant Filter."""
    if isinstance(expr, FilterComparison):
        _check_field_not_reserved(expr.field)
        return _build_qdrant_comparison(expr)
    if isinstance(expr, FilterIn):
        _check_field_not_reserved(expr.field)
        return _in_filter(expr.field, expr.values)
    if isinstance(expr, FilterIsNull):
        _check_field_not_reserved(expr.field)
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
    message = f"Unsupported filter expression type: {type(expr)}"
    raise TypeError(message)


def _ensure_tz_aware(datetime_value: datetime) -> datetime:
    """Return a timezone-aware datetime, treating naive values as UTC."""
    if datetime_value.tzinfo is None:
        return datetime_value.replace(tzinfo=UTC)
    return datetime_value


def _to_range_value(value: PropertyValue) -> int | float | str:
    """Convert a filter value for Qdrant Range."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _build_qdrant_comparison(comparison: FilterComparison) -> models.Filter:
    """Convert a Comparison into a Qdrant Filter."""
    field = comparison.field
    operator = comparison.op
    value = comparison.value

    if operator in ("=", "!="):
        negate = operator == "!="
        if isinstance(value, float):
            return _float_eq_filter(field, value, negate=negate)
        if isinstance(value, datetime):
            return _datetime_eq_filter(field, value, negate=negate)
        return _match_filter(field, value, negate=negate)
    if operator in _RANGE_OPERATORS:
        return _range_filter(field, value, _RANGE_OPERATORS[operator])

    message = f"Unsupported filter operator: {operator}"
    raise ValueError(message)


def _match_filter(
    field: str,
    value: bool | int | str,
    *,
    negate: bool,
) -> models.Filter:
    condition = models.FieldCondition(
        key=field,
        match=models.MatchValue(value=value),
    )
    if negate:
        return models.Filter(must_not=[condition])
    return models.Filter(must=[condition])


def _float_eq_filter(field: str, value: float, *, negate: bool) -> models.Filter:
    """Use a range filter for float equality since MatchValue doesn't accept floats."""
    condition = models.FieldCondition(
        key=field,
        range=models.Range(gte=value, lte=value),
    )
    if negate:
        return models.Filter(must_not=[condition])
    return models.Filter(must=[condition])


def _datetime_eq_filter(field: str, value: datetime, *, negate: bool) -> models.Filter:
    """Use a DatetimeRange filter for datetime equality since MatchValue doesn't accept datetimes."""
    value = _ensure_tz_aware(value)
    condition = models.FieldCondition(
        key=field,
        range=models.DatetimeRange(gte=value, lte=value),
    )
    if negate:
        return models.Filter(must_not=[condition])
    return models.Filter(must=[condition])


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
    range_parameter: str,
) -> models.Filter:
    if isinstance(value, datetime):
        value = _ensure_tz_aware(value)
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    range=models.DatetimeRange(**{range_parameter: value}),
                ),
            ],
        )

    range_value = _to_range_value(value)
    return models.Filter(
        must=[
            models.FieldCondition(
                key=field,
                range=models.Range(**{range_parameter: range_value}),  # type: ignore[arg-type]
            ),
        ],
    )


def _null_filter(field: str, *, negate: bool) -> models.Filter:
    condition = models.IsNullCondition(
        is_null=models.PayloadField(key=field),
    )
    if negate:
        return models.Filter(must_not=[condition])
    return models.Filter(must=[condition])


def _partition_filter(partition_key: str) -> models.Filter:
    """Build a Qdrant filter that matches the given partition key."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PARTITION_KEY_FIELD,
                match=models.MatchValue(value=partition_key),
            ),
        ],
    )


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

    @override
    async def startup(self) -> None:
        """No-op; Qdrant collections require no explicit open."""

    @override
    async def shutdown(self) -> None:
        """No-op; Qdrant collections require no explicit close."""

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
        partition_key: str,
    ) -> dict[str, bool | int | float | str | None]:
        """Build Qdrant-compatible payload from record properties.

        Missing schema keys are stored as explicit nulls so that
        `IsNull` filters work via the index for both omitted and
        `None` values.
        """
        payload: dict[str, bool | int | float | str | None] = dict.fromkeys(schema)
        payload[_PARTITION_KEY_FIELD] = partition_key
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
            if key == _PARTITION_KEY_FIELD:
                continue
            if schema.get(key) is datetime and isinstance(value, str):
                result[key] = datetime.fromisoformat(value)
            else:
                result[key] = cast(PropertyValue, value)
        return result

    @override
    async def upsert(
        self,
        *,
        partition_key: str,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        async with self._tracker("upsert"):
            schema = await self._get_properties_schema()
            points = [
                models.PointStruct(
                    id=record.uuid,
                    vector=record.vector if record.vector is not None else [],
                    payload=self._build_payload(
                        record.properties, schema, partition_key
                    ),
                )
                for record in records
            ]
            if points:
                await self._upsert_with_backoff(points)

    async def _upsert_with_backoff(self, points: Iterable[models.PointStruct]) -> None:
        """Upsert points, splitting the batch in half on failure and retrying."""
        points = list(points)
        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
        except (ResponseHandlingException, UnexpectedResponse):
            if len(points) <= 1:
                raise
            mid = len(points) // 2
            await self._upsert_with_backoff(points[:mid])
            await self._upsert_with_backoff(points[mid:])

    @override
    async def query(
        self,
        *,
        partition_key: str,
        query_vectors: Iterable[Sequence[float]],
        score_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> Iterable[QueryResult]:
        """Query for records matching the criteria by query vectors."""
        async with self._tracker("query"):
            partition_key_filter = _partition_filter(partition_key)
            if property_filter:
                property_qdrant_filter = _build_qdrant_filter(property_filter)
                qdrant_filter = models.Filter(
                    must=[partition_key_filter, property_qdrant_filter]
                )
            else:
                qdrant_filter = partition_key_filter

            effective_limit = limit if limit is not None else 10000

            requests = [
                models.QueryRequest(
                    query=list(query_vector),
                    filter=qdrant_filter,
                    score_threshold=score_threshold,
                    limit=effective_limit,
                    with_vector=return_vector,
                    with_payload=return_properties,
                )
                for query_vector in query_vectors
            ]

            batch_results = await self._client.query_batch_points(
                collection_name=self._collection_name,
                requests=requests,
            )

            query_results: list[QueryResult] = []
            for batch in batch_results:
                matches: list[QueryMatch] = []
                for point in batch.points:
                    vector: list[float] | None = None
                    if return_vector and point.vector is not None:
                        vector = cast(list[float], point.vector)

                    properties: dict[str, PropertyValue] | None = None
                    if return_properties and point.payload is not None:
                        properties = await self._parse_payload(
                            cast(dict[str, Any], point.payload),
                        )

                    matches.append(
                        QueryMatch(
                            score=cast(float, point.score),
                            record=Record(
                                uuid=UUID(str(point.id)),
                                vector=vector,
                                properties=properties,
                            ),
                        ),
                    )
                query_results.append(QueryResult(matches=matches))

            return query_results

    @override
    async def get(
        self,
        *,
        partition_key: str,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records from the collection by their UUIDs."""
        async with self._tracker("get"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return []

            # Always get payload so we can check partition_key.
            points = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=list(uuid_list),
                with_vectors=return_vector,
                with_payload=True,
            )

            points_by_uuid: dict[UUID, models.Record] = {
                UUID(str(point.id)): point
                for point in points
                if point.payload
                and cast(dict[str, Any], point.payload).get(_PARTITION_KEY_FIELD)
                == partition_key
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
        partition_key: str,
        record_uuids: Iterable[UUID],
    ) -> None:
        """Delete records from the collection by their UUIDs."""
        async with self._tracker("delete"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return

            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            _partition_filter(partition_key),
                            models.HasIdCondition(
                                has_id=list(uuid_list),
                            ),
                        ],
                    ),
                ),
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

            await self._client.create_payload_index(
                collection_name=collection_name,
                field_name=_PARTITION_KEY_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )

            if properties_schema:
                for property_name, property_type in properties_schema.items():
                    index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(
                        property_type
                    )
                    if index_type is not None:
                        await self._client.create_payload_index(
                            collection_name=collection_name,
                            field_name=property_name,
                            field_schema=index_type,
                        )

    @override
    async def get_collection(self, collection_name: str) -> QdrantCollection | None:
        """Get a collection handle from the vector store."""
        if not await self._client.collection_exists(collection_name):
            return None
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
