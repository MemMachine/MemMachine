"""Qdrant-based vector store implementation."""

import hashlib
import math
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
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.utils import ensure_tz_aware

from .collection_registry import Registration
from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionConfig,
)
from .registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreCollection,
    RegistryBackedVectorStoreParams,
)

# Point payload keys (stored on every Qdrant point).
# System keys start with _SYSTEM_KEY_PREFIX, whose hyphen no property key may
# contain (Record requires identifiers), so the two never collide.
_SYSTEM_KEY_PREFIX = "sys-"
_PAYLOAD_INCARNATION = f"{_SYSTEM_KEY_PREFIX}incarnation"
"""The payload key holding the incarnation of the collection a point belongs to.

A collection created again under a deleted one's name gets a fresh
incarnation, so it holds only the points written under that incarnation.
"""
_PAYLOAD_RECORD_UUID = f"{_SYSTEM_KEY_PREFIX}record_uuid"
"""The payload key holding a point's record UUID.

A point's id is derived from its incarnation and record UUID (`_point_id`),
so a query reads the record UUID from here.
"""


def _incarnation_filter(incarnation: UUID) -> models.Filter:
    """Build a Qdrant filter that matches the points of one collection incarnation."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PAYLOAD_INCARNATION,
                match=models.MatchValue(value=str(incarnation)),
            ),
        ],
    )


class QdrantVectorStoreCollection(RegistryBackedVectorStoreCollection):
    """A collection backed by Qdrant."""

    _RANGE_OPERATORS: ClassVar[dict[str, str]] = {
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
    }

    @staticmethod
    def _build_qdrant_filter(expr: FilterExpr) -> models.Filter:
        """Convert a FilterExpr tree into a Qdrant Filter."""
        if isinstance(expr, FilterComparison):
            return QdrantVectorStoreCollection._build_qdrant_comparison(expr)
        if isinstance(expr, FilterIn):
            return QdrantVectorStoreCollection._in_filter(expr.field, expr.values)
        if isinstance(expr, FilterIsNull):
            return QdrantVectorStoreCollection._null_filter(expr.field, negate=False)
        if isinstance(expr, FilterNot):
            return models.Filter(
                must_not=[QdrantVectorStoreCollection._build_qdrant_filter(expr.expr)]
            )
        if isinstance(expr, FilterAnd):
            left = QdrantVectorStoreCollection._build_qdrant_filter(expr.left)
            right = QdrantVectorStoreCollection._build_qdrant_filter(expr.right)
            return models.Filter(must=[left, right])
        if isinstance(expr, FilterOr):
            left = QdrantVectorStoreCollection._build_qdrant_filter(expr.left)
            right = QdrantVectorStoreCollection._build_qdrant_filter(expr.right)
            return models.Filter(should=[left, right])
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
                return QdrantVectorStoreCollection._float_eq_filter(
                    field, value, negate=negate
                )
            if isinstance(value, datetime):
                return QdrantVectorStoreCollection._datetime_eq_filter(
                    field, value, negate=negate
                )
            return QdrantVectorStoreCollection._match_filter(
                field, value, negate=negate
            )
        if operator in QdrantVectorStoreCollection._RANGE_OPERATORS:
            if not isinstance(value, OrderedValue):
                message = (
                    f"Range filter on '{field}' requires a numeric or datetime value, "
                    f"got {type(value).__name__}"
                )
                raise TypeError(message)
            return QdrantVectorStoreCollection._range_filter(
                field, value, QdrantVectorStoreCollection._RANGE_OPERATORS[operator]
            )

        message = f"Unsupported filter operator: {operator}"
        raise ValueError(message)

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
    def _null_filter(field: str, *, negate: bool) -> models.Filter:
        condition = models.IsEmptyCondition(
            is_empty=models.PayloadField(key=field),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        native_collection_name: str,
        registration: Registration,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Qdrant client and the registration the handle is bound to."""
        super().__init__(registration=registration, tracker=tracker)
        self._client = client
        self._native_collection_name = native_collection_name

    def _point_id(self, record_uuid: UUID) -> UUID:
        """The point id of a record: a UUIDv5 of the record UUID under the incarnation.

        Point ids are distinct across the collections sharing a native
        collection and across a name's incarnations.
        """
        return uuid5(self._incarnation, str(record_uuid))

    def _build_point(self, record: Record) -> models.PointStruct:
        """Build a Qdrant point from a vector store record."""
        payload: dict[str, PropertyValue] = {
            _PAYLOAD_INCARNATION: str(self._incarnation),
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
    async def _upsert(self, records: list[Record]) -> None:
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
                collection_name=self._native_collection_name,
                points=points,
                wait=True,
            )
        except UnexpectedResponse as err:
            if err.status_code not in (400, 413) or len(points) <= 1:
                raise
            mid = len(points) // 2
            await self._upsert_points(points[:mid])
            await self._upsert_points(points[mid:])

    @override
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        qdrant_filter = _incarnation_filter(self._incarnation)
        if property_filter is not None:
            qdrant_filter = models.Filter(
                must=[
                    qdrant_filter,
                    QdrantVectorStoreCollection._build_qdrant_filter(property_filter),
                ]
            )

        # Qdrant keeps only scores strictly better than its threshold, compared
        # in single precision, so send the next single-precision value on the
        # worse side; the check below applies the caller's threshold exactly.
        higher_is_better = self.config.similarity_metric.higher_is_better
        qdrant_score_threshold = None
        if score_threshold is not None:
            worse = np.float32(-np.inf if higher_is_better else np.inf)
            adjacent = float(np.nextafter(np.float32(score_threshold), worse))
            if math.isfinite(adjacent):
                qdrant_score_threshold = adjacent
        requests = [
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
        ]

        batch_results = await self._client.query_batch_points(
            collection_name=self._native_collection_name,
            requests=requests,
        )

        return [
            QueryResult(
                matches=[
                    QueryMatch(
                        score=point.score,
                        record_uuid=UUID((point.payload or {})[_PAYLOAD_RECORD_UUID]),
                    )
                    for point in batch.points
                    if score_threshold is None
                    or (
                        point.score >= score_threshold
                        if higher_is_better
                        else point.score <= score_threshold
                    )
                ]
            )
            for batch in batch_results
        ]

    @override
    async def _delete(self, record_uuids: list[UUID]) -> None:
        await self._client.delete(
            collection_name=self._native_collection_name,
            points_selector=models.PointIdsList(
                points=[str(self._point_id(uuid)) for uuid in record_uuids]
            ),
            wait=True,
        )


class QdrantVectorStoreParams(RegistryBackedVectorStoreParams):
    """
    Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.
    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )


class QdrantVectorStore(RegistryBackedVectorStore[QdrantVectorStoreCollection]):
    """Asynchronous Qdrant-based implementation of VectorStore.

    A logical collection is the points carrying its incarnation in their
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

    _PROPERTY_TYPE_TO_INDEX_TYPE: ClassVar[
        dict[type[PropertyValue], models.PayloadSchemaType]
    ] = {
        bool: models.PayloadSchemaType.BOOL,
        int: models.PayloadSchemaType.INTEGER,
        float: models.PayloadSchemaType.FLOAT,
        str: models.PayloadSchemaType.KEYWORD,
        datetime: models.PayloadSchemaType.DATETIME,
    }

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 409
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.ALREADY_EXISTS
        return False

    @staticmethod
    def _is_not_found_error(error: Exception) -> bool:
        """Check if an exception indicates a resource was not found."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 404
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.NOT_FOUND
        return False

    @staticmethod
    def _build_native_collection_name(
        namespace: str, config: VectorStoreCollectionConfig
    ) -> str:
        """Build a deterministic native collection name from namespace and config."""
        digest = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        return f"{namespace}__{digest}"

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__(params, metrics_prefix="vector_store_qdrant")
        self._client: AsyncQdrantClient = params.client

        self._hnsw_m = 16

    @override
    def _build_collection_handle(
        self, registration: Registration
    ) -> QdrantVectorStoreCollection:
        return QdrantVectorStoreCollection(
            client=self._client,
            native_collection_name=QdrantVectorStore._build_native_collection_name(
                registration.namespace, registration.config
            ),
            registration=registration,
            tracker=self._tracker,
        )

    @override
    async def _prepare_storage(
        self,
        namespace: str,
        config: VectorStoreCollectionConfig,
        incarnation: UUID,
    ) -> None:
        native_collection_name = QdrantVectorStore._build_native_collection_name(
            namespace, config
        )
        distance = QdrantVectorStore._SIMILARITY_METRIC_TO_QDRANT_DISTANCE[
            config.similarity_metric
        ]
        # The collection and each payload index are created under their own
        # already-exists guard, so a creation that finds the collection there
        # still creates the indexes it lacks.
        try:
            await self._client.create_collection(
                collection_name=native_collection_name,
                vectors_config=models.VectorParams(
                    size=config.vector_dimensions, distance=distance
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                    payload_m=self._hnsw_m,
                ),
                # A filter may name a property without a payload index, which
                # the server scans for. Strict mode refuses such a filter
                # instead, and a server may turn it on for new collections by
                # default, so it is turned off here.
                strict_mode_config=models.StrictModeConfig(enabled=False),
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise

        indexes: list[tuple[str, Any]] = [
            (
                _PAYLOAD_INCARNATION,
                models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
        ]
        for prop_name, prop_type in config.indexed_properties_schema.items():
            index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(prop_type)
            if index_type is not None:
                indexes.append((prop_name, index_type))

        for field_name, field_schema in indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=native_collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    @override
    async def _purge_round(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        # If a point remains under the incarnation, one filter-delete removes
        # them all.
        native_collection_name = QdrantVectorStore._build_native_collection_name(
            namespace, config
        )
        try:
            points, _ = await self._client.scroll(
                collection_name=native_collection_name,
                scroll_filter=_incarnation_filter(incarnation),
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
                collection_name=native_collection_name,
                points_selector=models.FilterSelector(
                    filter=_incarnation_filter(incarnation),
                ),
                wait=True,
            )
        return bool(points)
