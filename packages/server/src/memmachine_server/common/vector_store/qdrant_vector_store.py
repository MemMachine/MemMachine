"""Qdrant-based vector store implementation."""

import math
from collections.abc import Mapping
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
    OrderingOp,
)
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
)
from .partition_registry import Registration
from .registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreParams,
    RegistryBackedVectorStorePartition,
)

# Point payload keys (stored on every Qdrant point).
# System keys start with _SYSTEM_KEY_PREFIX, whose hyphen no property key may
# contain (Record requires identifiers), so the two never collide.
_SYSTEM_KEY_PREFIX = "sys-"
_PAYLOAD_INCARNATION = f"{_SYSTEM_KEY_PREFIX}incarnation"
"""The payload key holding the incarnation of the partition a point belongs to.

A partition created again under a deleted one's key gets a fresh
incarnation, so it holds only the points written under that incarnation.
"""
_PAYLOAD_RECORD_UUID = f"{_SYSTEM_KEY_PREFIX}record_uuid"
"""The payload key holding a point's record UUID.

A point's id is derived from its incarnation and record UUID (`_point_id`),
so a query reads the record UUID from here.
"""


def _incarnation_filter(incarnation: UUID) -> models.Filter:
    """Build a Qdrant filter that matches the points of one partition incarnation."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PAYLOAD_INCARNATION,
                match=models.MatchValue(value=str(incarnation)),
            ),
        ],
    )


class QdrantVectorStorePartition(RegistryBackedVectorStorePartition):
    """A partition backed by Qdrant: one payload value inside the store's collection."""

    _RANGE_OPERATORS: ClassVar[dict[OrderingOp, str]] = {
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
    }

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {Equals, Ordering, In, IsNull, And, Or, Not}
    )

    @staticmethod
    def _build_qdrant_filter(expr: FilterExpr) -> models.Filter:
        """Convert a FilterExpr tree into a Qdrant Filter over the declared keys."""
        build = QdrantVectorStorePartition._build_qdrant_filter
        match expr:
            case Equals() | Ordering() | In():
                return QdrantVectorStorePartition._leaf_filter(expr)
            case IsNull(field):
                return models.Filter(
                    must=[QdrantVectorStorePartition._missing_condition(field)]
                )
            case Not(operand):
                return models.Filter(must_not=[build(operand)])
            case And(operands):
                return models.Filter(must=[build(o) for o in operands])
            case Or(operands):
                return models.Filter(should=[build(o) for o in operands])

    @staticmethod
    def _leaf_filter(expr: Equals | Ordering | In) -> models.Filter:
        """The leaf as a Qdrant Filter."""
        match expr:
            case Equals(field, value):
                condition = QdrantVectorStorePartition._eq_condition(field, value)
            case Ordering(field, op, value):
                condition = QdrantVectorStorePartition._range_condition(
                    field, value, QdrantVectorStorePartition._RANGE_OPERATORS[op]
                )
            case In(field, values):
                condition = models.FieldCondition(
                    key=field, match=models.MatchAny(any=list(values))
                )
        return models.Filter(must=[condition])

    @staticmethod
    def _eq_condition(field: str, value: PropertyValue) -> models.FieldCondition:
        """Match a field against a value, by the only condition Qdrant offers for its type."""
        if isinstance(value, float):
            # MatchValue does not accept floats.
            return models.FieldCondition(
                key=field, range=models.Range(gte=value, lte=value)
            )
        if isinstance(value, datetime):
            instant = ensure_tz_aware(value)
            return models.FieldCondition(
                key=field, range=models.DatetimeRange(gte=instant, lte=instant)
            )
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))

    @staticmethod
    def _range_condition(
        field: str,
        value: OrderedValue,
        range_parameter: str,
    ) -> models.FieldCondition:
        if isinstance(value, datetime):
            return models.FieldCondition(
                key=field,
                range=models.DatetimeRange(**{range_parameter: ensure_tz_aware(value)}),
            )
        return models.FieldCondition(
            key=field, range=models.Range(**{range_parameter: value})
        )

    @staticmethod
    def _missing_condition(field: str) -> models.IsEmptyCondition:
        """Points that do not carry the field."""
        return models.IsEmptyCondition(is_empty=models.PayloadField(key=field))

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        vector_store_name: str,
        registration: Registration,
        vector_dimensions: int,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Qdrant client and the registration the handle is bound to."""
        super().__init__(
            vector_store_name=vector_store_name,
            registration=registration,
            vector_dimensions=vector_dimensions,
            indexed_properties=indexed_properties,
            tracker=tracker,
        )
        self._client = client

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return QdrantVectorStorePartition._SUPPORTED_FILTER_NODES

    def _point_id(self, record_uuid: UUID) -> UUID:
        """The point id of a record: a UUIDv5 of the record UUID under the incarnation.

        Point ids are distinct across the partitions of the store's collection
        and across a key's incarnations.
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

    @override
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        min_cosine_similarity: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        qdrant_filter = _incarnation_filter(self._incarnation)
        if property_filter is not None:
            qdrant_filter = models.Filter(
                must=[
                    qdrant_filter,
                    QdrantVectorStorePartition._build_qdrant_filter(property_filter),
                ]
            )

        # Qdrant keeps only scores strictly above its threshold, compared in
        # single precision, so send the next single-precision value below the
        # minimum; the check below applies the caller's minimum exactly.
        qdrant_score_threshold = None
        if min_cosine_similarity is not None:
            adjacent = float(
                np.nextafter(np.float32(min_cosine_similarity), np.float32(-np.inf))
            )
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
            collection_name=self._vector_store_name,
            requests=requests,
        )

        return [
            QueryResult(
                matches=[
                    QueryMatch(
                        cosine_similarity=point.score,
                        record_uuid=UUID((point.payload or {})[_PAYLOAD_RECORD_UUID]),
                    )
                    for point in batch.points
                    if min_cosine_similarity is None
                    or point.score >= min_cosine_similarity
                ]
            )
            for batch in batch_results
        ]

    @override
    async def _delete(self, record_uuids: list[UUID]) -> None:
        await self._client.delete(
            collection_name=self._vector_store_name,
            points_selector=models.PointIdsList(
                points=[str(self._point_id(uuid)) for uuid in record_uuids]
            ),
            wait=True,
        )


class QdrantVectorStoreParams(RegistryBackedVectorStoreParams):
    """
    Parameters for QdrantVectorStore.

    The native Qdrant collection is named `vector_store_name`, and each
    declared property gets a payload index of its declared type.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.
    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )


class QdrantVectorStore(RegistryBackedVectorStore[QdrantVectorStorePartition]):
    """Asynchronous Qdrant-based implementation of VectorStore.

    The store is one native Qdrant collection, named at construction, in
    which a partition is the points carrying its incarnation in their
    payload.
    """

    _QDRANT_DISTANCE: ClassVar[models.Distance] = models.Distance.COSINE

    # Every key a filter may name is indexed (the declared keys and the
    # incarnation), so the server may refuse a filter on an unindexed one
    # instead of scanning the collection for it.
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

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__(params, metrics_prefix="vector_store_qdrant")
        self._client: AsyncQdrantClient = params.client

        self._hnsw_m = 16

    @override
    async def _prepare_storage(self) -> None:
        distance = QdrantVectorStore._QDRANT_DISTANCE
        # The collection and each payload index are created under their own
        # already-exists guard, so a creation that finds the collection there
        # still creates the indexes it lacks.
        try:
            await self._client.create_collection(
                collection_name=self.vector_store_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dimensions, distance=distance
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
                _PAYLOAD_INCARNATION,
                models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
        ]
        for prop_name, prop_type in self.indexed_properties.items():
            index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(prop_type)
            if index_type is not None:
                indexes.append((prop_name, index_type))

        for field_name, field_schema in indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=self.vector_store_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    @override
    async def _prepare_partition_storage(
        self, partition_key: str, incarnation: UUID
    ) -> None:
        # A partition is the points carrying its incarnation in the
        # store's one native collection, which startup prepared; it has no
        # storage of its own.
        pass

    @override
    def _partition_handle(
        self, registration: Registration
    ) -> QdrantVectorStorePartition:
        return QdrantVectorStorePartition(
            client=self._client,
            vector_store_name=self.vector_store_name,
            registration=registration,
            vector_dimensions=self.vector_dimensions,
            indexed_properties=self.indexed_properties,
            tracker=self._tracker,
        )

    @override
    async def _purge_round(self, incarnation: UUID) -> bool:
        # If a point remains under the incarnation, one filter-delete removes
        # them all.
        try:
            points, _ = await self._client.scroll(
                collection_name=self.vector_store_name,
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
                collection_name=self.vector_store_name,
                points_selector=models.FilterSelector(
                    filter=_incarnation_filter(incarnation),
                ),
                wait=True,
            )
        return bool(points)
