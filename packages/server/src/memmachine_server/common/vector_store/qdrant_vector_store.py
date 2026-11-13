"""Qdrant-based vector store implementation."""

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar, override
from uuid import UUID, uuid5

import grpc
import grpc.aio
import numpy as np
from pydantic import Field, InstanceOf
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from memmachine_server.common.data_types import (
    OrderedValue,
    PropertyType,
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
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreSchema,
)
from .utils import (
    validate_declared_properties,
    validate_dimensions,
    validate_limit,
    validate_query_vector,
    validate_score_threshold,
    validate_supported_filter,
)
from .vector_store import VectorStore, VectorStoreParams, VectorStorePartition

# Point payload keys (stored on every Qdrant point).
# System keys start with _SYSTEM_KEY_PREFIX, whose hyphen no property key may
# contain (Record requires identifiers), so the two never collide.
_SYSTEM_KEY_PREFIX = "sys-"
_PAYLOAD_PARTITION_UUID = f"{_SYSTEM_KEY_PREFIX}partition_uuid"
"""The payload key holding the partition UUID of the partition a point belongs to.

A partition created again under a deleted one's name gets a fresh
partition UUID, so it holds only the points written under that partition UUID.
"""
_PAYLOAD_RECORD_UUID = f"{_SYSTEM_KEY_PREFIX}record_uuid"
"""The payload key holding a point's record UUID.

A point's id is derived from its partition UUID and record UUID (`_point_id`),
so a query reads the record UUID from here.
"""


def _partition_filter(partition_uuid: UUID) -> models.Filter:
    """Build a Qdrant filter that matches the points of one partition."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PAYLOAD_PARTITION_UUID,
                match=models.MatchValue(value=str(partition_uuid)),
            ),
        ],
    )


class QdrantVectorStorePartition(VectorStorePartition):
    """A partition's points in the store's Qdrant collection."""

    _RANGE_OPERATORS: ClassVar[dict[str, str]] = {
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
    }

    # A filter no point satisfies. A leaf whose value is of another type
    # than its key declares matches nothing, as on every backend; sent as
    # is, the server would refuse it for the field's index (strict mode)
    # rather than scan for it.
    _NO_MATCH: ClassVar[models.Filter] = models.Filter(
        must=[models.HasIdCondition(has_id=[])]
    )

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        vector_store_name: str,
        partition_uuid: UUID,
        store_schema: VectorStoreSchema,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Qdrant client and the partition the handle is bound to."""
        self._client = client
        self._vector_store_name = vector_store_name
        self._partition_uuid = partition_uuid
        self._store_schema = store_schema
        self._tracker = tracker

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        async with self._tracker("upsert"):
            records = list(records)
            for record in records:
                validate_declared_properties(
                    record.properties, self._store_schema.indexed_properties
                )
                validate_dimensions(record.vector, self._store_schema.vector_dimensions)
            if not records:
                return
            await self._upsert_points([self._build_point(record) for record in records])

    async def _upsert_points(self, points: list[models.PointStruct]) -> None:
        """Upsert points, halving a batch refused as sent.

        Qdrant's REST API refuses a request over its max_request_size_mb with
        a 400, the status of any request it finds invalid, and a proxy in
        front of it may refuse one with a 413. A batch refused with either is
        halved until the halves fit or a single point is refused. Any other
        error raises at once.
        """
        try:
            # Waiting for Qdrant to apply the write keeps the writes it has
            # accepted but not applied to those in flight, whose callers see
            # the wait as latency, and reports a failure to apply.
            await self._client.upsert(
                collection_name=self._vector_store_name,
                points=points,
                wait=True,
            )
        except UnexpectedResponse as err:
            if err.status_code not in (400, 413) or len(points) <= 1:
                raise
            mid = len(points) // 2
            await self._upsert_points(points[:mid])
            await self._upsert_points(points[mid:])

    def _build_point(self, record: Record) -> models.PointStruct:
        """Build a Qdrant point from a vector store record."""
        payload: dict[str, PropertyValue] = {
            _PAYLOAD_PARTITION_UUID: str(self._partition_uuid),
            _PAYLOAD_RECORD_UUID: str(record.uuid),
        }
        for key, value in record.properties.items():
            if isinstance(value, datetime):
                payload[key] = ensure_tz_aware(value)
            else:
                payload[key] = value
        return models.PointStruct(
            id=str(self._point_id(record.uuid)), vector=record.vector, payload=payload
        )

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            for query_vector in query_vectors:
                validate_query_vector(
                    query_vector, self._store_schema.vector_dimensions
                )
            validate_score_threshold(score_threshold)
            validate_limit(limit)
            if property_filter is not None:
                validate_supported_filter(
                    property_filter, self._store_schema.indexed_properties
                )
            if not query_vectors:
                return []

            qdrant_filter = self._query_filter(property_filter)
            qdrant_score_threshold = self._qdrant_score_threshold(score_threshold)
            batch_results = await self._client.query_batch_points(
                collection_name=self._vector_store_name,
                requests=[
                    models.QueryRequest(
                        query=query_vector,
                        filter=qdrant_filter,
                        score_threshold=qdrant_score_threshold,
                        limit=limit,
                        with_vector=False,
                        with_payload=models.PayloadSelectorInclude(
                            include=[_PAYLOAD_RECORD_UUID]
                        ),
                    )
                    for query_vector in query_vectors
                ],
            )
            return [
                QueryResult(matches=self._matches(batch.points, score_threshold))
                for batch in batch_results
            ]

    def _query_filter(self, property_filter: FilterExpr | None) -> models.Filter:
        """The filter a query sends: the partition's points, narrowed by the property filter when one is given."""
        qdrant_filter = _partition_filter(self._partition_uuid)
        if property_filter is None:
            return qdrant_filter
        return models.Filter(
            must=[
                qdrant_filter,
                QdrantVectorStorePartition._build_qdrant_filter(
                    property_filter, self._store_schema.indexed_properties
                ),
            ]
        )

    @staticmethod
    def _build_qdrant_filter(
        expr: FilterExpr, indexed_properties: Mapping[str, PropertyType]
    ) -> models.Filter:
        """Convert a FilterExpr tree into a Qdrant Filter over the declared keys."""
        build = QdrantVectorStorePartition._build_qdrant_filter
        if isinstance(expr, FilterComparison):
            if type(expr.value) is not indexed_properties[expr.field]:
                return QdrantVectorStorePartition._NO_MATCH
            return QdrantVectorStorePartition._build_qdrant_comparison(expr)
        if isinstance(expr, FilterIn):
            if (
                expr.values
                and type(expr.values[0]) is not indexed_properties[expr.field]
            ):
                return QdrantVectorStorePartition._NO_MATCH
            return QdrantVectorStorePartition._in_filter(expr.field, expr.values)
        if isinstance(expr, FilterIsNull):
            return QdrantVectorStorePartition._null_filter(expr.field, negate=False)
        if isinstance(expr, FilterNot):
            return models.Filter(must_not=[build(expr.expr, indexed_properties)])
        if isinstance(expr, FilterAnd):
            return models.Filter(
                must=[
                    build(expr.left, indexed_properties),
                    build(expr.right, indexed_properties),
                ]
            )
        if isinstance(expr, FilterOr):
            return models.Filter(
                should=[
                    build(expr.left, indexed_properties),
                    build(expr.right, indexed_properties),
                ]
            )
        message = f"Unsupported filter expression type: {type(expr)}"
        raise TypeError(message)

    @staticmethod
    def _build_qdrant_comparison(comparison: FilterComparison) -> models.Filter:
        """Convert a Comparison into a Qdrant Filter."""
        field = comparison.field
        operator = comparison.op
        value = comparison.value

        if operator in ("=", "!="):
            negate = operator == "!="
            if isinstance(value, float):
                return QdrantVectorStorePartition._float_eq_filter(
                    field, value, negate=negate
                )
            if isinstance(value, datetime):
                return QdrantVectorStorePartition._datetime_eq_filter(
                    field, value, negate=negate
                )
            return QdrantVectorStorePartition._match_filter(field, value, negate=negate)
        if operator in QdrantVectorStorePartition._RANGE_OPERATORS:
            if not isinstance(value, OrderedValue):
                message = (
                    f"Range filter on '{field}' requires a numeric or datetime value, "
                    f"got {type(value).__name__}"
                )
                raise TypeError(message)
            return QdrantVectorStorePartition._range_filter(
                field, value, QdrantVectorStorePartition._RANGE_OPERATORS[operator]
            )

        message = f"Unsupported filter operator: {operator}"
        raise ValueError(message)

    @staticmethod
    def _float_eq_filter(field: str, value: float, *, negate: bool) -> models.Filter:
        """Use a range filter for float equality since MatchValue doesn't accept floats."""
        condition = models.FieldCondition(
            key=field,
            range=models.Range(gte=value, lte=value),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    @staticmethod
    def _datetime_eq_filter(
        field: str, value: datetime, *, negate: bool
    ) -> models.Filter:
        """Use a DatetimeRange filter for datetime equality since MatchValue doesn't accept datetimes."""
        value = ensure_tz_aware(value)
        condition = models.FieldCondition(
            key=field,
            range=models.DatetimeRange(gte=value, lte=value),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    @staticmethod
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

    @staticmethod
    def _range_filter(
        field: str,
        value: OrderedValue,
        range_parameter: str,
    ) -> models.Filter:
        if isinstance(value, datetime):
            value = ensure_tz_aware(value)
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key=field,
                        range=models.DatetimeRange(**{range_parameter: value}),
                    ),
                ],
            )

        return models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    range=models.Range(**{range_parameter: value}),
                ),
            ],
        )

    @staticmethod
    def _in_filter(field: str, value: list[int] | list[str]) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    match=models.MatchAny(any=value),
                ),
            ],
        )

    @staticmethod
    def _null_filter(field: str, *, negate: bool) -> models.Filter:
        condition = models.IsEmptyCondition(
            is_empty=models.PayloadField(key=field),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    def _qdrant_score_threshold(self, score_threshold: float | None) -> float | None:
        """
        The threshold a query sends for the caller's, keeping every point that meets the caller's.

        Qdrant keeps only scores strictly better than its threshold, compared
        in single precision, so the query sends the next single-precision
        value on the worse side, or none when that value is not finite;
        `_matches` applies the caller's threshold exactly.
        """
        if score_threshold is None:
            return None
        worse = np.float32(
            -np.inf if self._store_schema.similarity_metric.higher_is_better else np.inf
        )
        adjacent = float(np.nextafter(np.float32(score_threshold), worse))
        return adjacent if math.isfinite(adjacent) else None

    def _matches(
        self, points: list[models.ScoredPoint], score_threshold: float | None
    ) -> list[QueryMatch]:
        """The matches of one query's points, best first as Qdrant returns them, keeping only those whose score meets the caller's threshold."""
        higher_is_better = self._store_schema.similarity_metric.higher_is_better
        return [
            QueryMatch(
                score=point.score,
                record_uuid=UUID((point.payload or {})[_PAYLOAD_RECORD_UUID]),
            )
            for point in points
            if score_threshold is None
            or (
                point.score >= score_threshold
                if higher_is_better
                else point.score <= score_threshold
            )
        ]

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        async with self._tracker("delete"):
            record_uuids = list(record_uuids)
            if not record_uuids:
                return
            await self._client.delete(
                collection_name=self._vector_store_name,
                points_selector=models.PointIdsList(
                    points=[str(self._point_id(uuid)) for uuid in record_uuids]
                ),
                wait=True,
            )

    def _point_id(self, record_uuid: UUID) -> UUID:
        """The point id of a record: a UUIDv5 of the record UUID under the partition UUID.

        Point ids are distinct across the partitions of the store's collection
        and across the partitions created under one name.
        """
        return uuid5(self._partition_uuid, str(record_uuid))


class QdrantVectorStoreParams(VectorStoreParams):
    """
    Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance, which stores of different names
            may share.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance, which stores of different names may share",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore.

    The store is one native Qdrant collection, named at construction, in
    which a partition is the points carrying it in their
    payload.
    """

    _SIMILARITY_METRIC_TO_QDRANT_DISTANCE: ClassVar[
        dict[SimilarityMetric, models.Distance]
    ] = {
        SimilarityMetric.COSINE: models.Distance.COSINE,
        SimilarityMetric.DOT: models.Distance.DOT,
        SimilarityMetric.EUCLIDEAN: models.Distance.EUCLID,
        SimilarityMetric.MANHATTAN: models.Distance.MANHATTAN,
    }

    # The store filters only on indexed keys (the declared keys and the
    # partition UUID). Strict mode makes the server refuse a filter on any other
    # key instead of scanning for it, so a missing index fails a search
    # rather than slowing it.
    _STRICT_MODE: ClassVar[models.StrictModeConfig] = models.StrictModeConfig(
        enabled=True,
        unindexed_filtering_retrieve=False,
        unindexed_filtering_update=False,
    )

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
        self._vector_store_name = params.vector_store_name
        self._store_schema = params.store_schema
        self._tracker = OperationTracker(
            params.metrics_factory, prefix="vector_store_qdrant"
        )
        self._client: AsyncQdrantClient = params.client

        self._hnsw_m = 16

    @property
    @override
    def store_schema(self) -> VectorStoreSchema:
        return self._store_schema

    @override
    async def startup(self) -> None:
        async with self._tracker("startup"):
            await self._prepare_storage()

    async def _prepare_storage(self) -> None:
        """Prepare the collection every partition lives in, and its payload indexes, each step idempotent."""
        distance = QdrantVectorStore._SIMILARITY_METRIC_TO_QDRANT_DISTANCE[
            self._store_schema.similarity_metric
        ]
        # The collection and each payload index are created under their own
        # already-exists guard, so a creation that finds the collection there
        # still creates the indexes it lacks.
        try:
            await self._client.create_collection(
                collection_name=self._vector_store_name,
                vectors_config=models.VectorParams(
                    size=self._store_schema.vector_dimensions, distance=distance
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                    payload_m=self._hnsw_m,
                ),
                strict_mode_config=QdrantVectorStore._STRICT_MODE,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise

        indexes: list[tuple[str, Any]] = [
            (
                _PAYLOAD_PARTITION_UUID,
                models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
        ]
        for (
            prop_name,
            prop_type,
        ) in self._store_schema.indexed_properties.items():
            index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(prop_type)
            if index_type is not None:
                indexes.append((prop_name, index_type))

        for field_name, field_schema in indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=self._vector_store_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 409
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.ALREADY_EXISTS
        return False

    @override
    async def shutdown(self) -> None:
        # The caller owns the client's lifecycle.
        pass

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        # Every partition lives in the collection startup
        # prepared.
        return

    @override
    async def get_partition(self, partition_uuid: UUID) -> QdrantVectorStorePartition:
        return QdrantVectorStorePartition(
            client=self._client,
            vector_store_name=self._vector_store_name,
            partition_uuid=partition_uuid,
            store_schema=self._store_schema,
            tracker=self._tracker,
        )

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        async with self._tracker("purge_partition"):
            return await self._purge_partition(partition_uuid)

    async def _purge_partition(self, partition_uuid: UUID) -> bool:
        # If a point of the partition remains, one filter-delete removes
        # them all.
        try:
            points, _ = await self._client.scroll(
                collection_name=self._vector_store_name,
                scroll_filter=_partition_filter(partition_uuid),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
            # The native collection is gone with everything in it.
            if not QdrantVectorStore._is_not_found_error(e):
                raise
            points = []
        if points:
            await self._client.delete(
                collection_name=self._vector_store_name,
                points_selector=models.FilterSelector(
                    filter=_partition_filter(partition_uuid),
                ),
                wait=True,
            )
        return bool(points)

    @staticmethod
    def _is_not_found_error(error: Exception) -> bool:
        """Check if an exception indicates a resource was not found."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 404
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.NOT_FOUND
        return False

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        # Every partition lives in the collection startup prepared.
        return
