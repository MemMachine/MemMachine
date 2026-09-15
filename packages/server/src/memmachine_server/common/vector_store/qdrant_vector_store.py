"""Qdrant-based vector store implementation."""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID, uuid4, uuid5
from weakref import WeakKeyDictionary

import grpc
import grpc.aio
from pydantic import BaseModel, Field, InstanceOf, field_validator
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from memmachine_server.common.data_types import (
    OrderedValue,
    PropertyType,
    PropertyValue,
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
    IndexedProperties,
    PartitionSchema,
    QueryMatch,
    QueryResult,
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionSchemaMismatchError,
    indexed_property_names,
    validate_collection_name,
)
from .utils import validate_filter, validate_identifier
from .vector_store import VectorStore, VectorStorePartition

# Point payload keys (stored on every Qdrant point).
# System keys use _SYSTEM_KEY_PREFIX, which contains a hyphen. Hyphens are valid in
# Qdrant but forbidden by _IDENTIFIER_RE, so system keys can never collide with user keys.
_SYSTEM_KEY_PREFIX = "sys-"
_PAYLOAD_INCARNATION = f"{_SYSTEM_KEY_PREFIX}incarnation"
"""The payload key naming the partition incarnation a point belongs to.

Points carry the incarnation, never the partition key: a partition deleted
and re-created under the same key gets a fresh incarnation, and its
predecessor's points are invisible to it while the purge reclaims them.
"""


def _partition_filter(incarnation: UUID) -> models.Filter:
    """Build a Qdrant filter that matches the points of one partition incarnation."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PAYLOAD_INCARNATION,
                match=models.MatchValue(value=incarnation.hex),
            ),
        ],
    )


class QdrantVectorStorePartition(VectorStorePartition):
    """A partition backed by Qdrant: one payload value inside the store's collection."""

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
            return QdrantVectorStorePartition._build_qdrant_comparison(expr)
        if isinstance(expr, FilterIn):
            return QdrantVectorStorePartition._in_filter(expr.field, expr.values)
        if isinstance(expr, FilterIsNull):
            return QdrantVectorStorePartition._null_filter(expr.field, negate=False)
        if isinstance(expr, FilterNot):
            return models.Filter(
                must_not=[QdrantVectorStorePartition._build_qdrant_filter(expr.expr)]
            )
        if isinstance(expr, FilterAnd):
            left = QdrantVectorStorePartition._build_qdrant_filter(expr.left)
            right = QdrantVectorStorePartition._build_qdrant_filter(expr.right)
            return models.Filter(must=[left, right])
        if isinstance(expr, FilterOr):
            left = QdrantVectorStorePartition._build_qdrant_filter(expr.left)
            right = QdrantVectorStorePartition._build_qdrant_filter(expr.right)
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
        collection_name: str,
        partition_key: str,
        incarnation: UUID,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
        is_live: Callable[[str, UUID], Awaitable[bool]],
    ) -> None:
        """Initialize with a Qdrant client and the incarnation the handle is bound to."""
        self._client = client
        self._tracker = tracker
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._incarnation = incarnation
        self._indexed_properties = dict(indexed_properties)
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the partition's.

        Qdrant has no transactions, so the check and the operation are two
        calls; a deletion landing between them leaves points under a dead
        incarnation, which the purge reclaims like any other.
        """
        if not await self._is_live(self._partition_key, self._incarnation):
            raise VectorStorePartitionHandleStaleError(
                self._collection_name, self._partition_key
            )

    @property
    @override
    def partition_key(self) -> str:
        return self._partition_key

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    def _build_payload(
        self,
        properties: Mapping[str, PropertyValue],
    ) -> dict[str, PropertyValue]:
        """Build Qdrant-compatible payload from record properties."""
        payload: dict[str, PropertyValue] = {
            _PAYLOAD_INCARNATION: self._incarnation.hex,
        }
        for key, value in properties.items():
            payload[key] = (
                ensure_tz_aware(value) if isinstance(value, datetime) else value
            )
        return payload

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        async with self._tracker("upsert"):
            await self._fence()
            points = [
                models.PointStruct(
                    id=record.uuid,
                    vector=record.vector,
                    payload=self._build_payload(record.properties),
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
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        min_cosine_similarity: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        """Query for records matching the criteria by query vectors."""
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            if not query_vectors:
                return []

            await self._fence()
            partition_key_filter = _partition_filter(self._incarnation)
            if property_filter is not None:
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")
                property_qdrant_filter = (
                    QdrantVectorStorePartition._build_qdrant_filter(property_filter)
                )
                qdrant_filter = models.Filter(
                    must=[partition_key_filter, property_qdrant_filter]
                )
            else:
                qdrant_filter = partition_key_filter

            requests = [
                models.QueryRequest(
                    query=query_vector,
                    filter=qdrant_filter,
                    score_threshold=min_cosine_similarity,
                    limit=limit,
                    with_vector=False,
                    with_payload=False,
                )
                for query_vector in query_vectors
            ]

            batch_results = await self._client.query_batch_points(
                collection_name=self._collection_name,
                requests=requests,
            )

            query_results: list[QueryResult] = []
            for batch in batch_results:
                matches = [
                    QueryMatch(
                        cosine_similarity=point.score,
                        record_uuid=UUID(str(point.id)),
                    )
                    for point in batch.points
                ]
                query_results.append(QueryResult(matches=matches))

            return query_results

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

            await self._fence()
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            _partition_filter(self._incarnation),
                            models.HasIdCondition(
                                has_id=list(uuid_list),
                            ),
                        ],
                    ),
                ),
            )


class QdrantVectorStoreParams(BaseModel):
    """
    Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.
        collection (str):
            The collection this store is; the native Qdrant collection's
            name, so stores of different collections may share the client.
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        registry_replication_factor (int):
            Replication factor for registry collections. Write consistency factor is
            set to match so all replicas confirm writes before returning, guaranteeing
            read-your-writes from any available replica
            (default: 1).
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key gets a payload index of its declared type.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )
    collection: str = Field(
        ...,
        description="The collection this store is; the native Qdrant collection's name",
    )
    vector_dimensions: int = Field(
        ..., gt=0, description="Dimensionality of every vector in the store"
    )
    registry_replication_factor: int = Field(
        1,
        description=(
            "Replication factor for registry collections. Write consistency factor is "
            "set to match so all replicas confirm writes before returning, guaranteeing "
            "read-your-writes from any available replica"
        ),
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every partition of this store carries",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )

    @field_validator("collection")
    @classmethod
    def _validate_collection(cls, collection: str) -> str:
        validate_collection_name(collection)
        return collection


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore.

    The store is one native Qdrant collection, named at construction, in
    which every partition is a payload value. A registry collection beside
    it records which partitions exist and what they were created under.
    """

    _QDRANT_DISTANCE: ClassVar[models.Distance] = models.Distance.COSINE

    _PROPERTY_TYPE_TO_INDEX_TYPE: ClassVar[
        dict[type[PropertyValue], models.PayloadSchemaType]
    ] = {
        bool: models.PayloadSchemaType.BOOL,
        int: models.PayloadSchemaType.INTEGER,
        float: models.PayloadSchemaType.FLOAT,
        str: models.PayloadSchemaType.KEYWORD,
        datetime: models.PayloadSchemaType.DATETIME,
    }

    # The registry collection holds two kinds of point. A live entry, one per
    # partition, has the deterministic id of its key and carries the key, the
    # incarnation and the schema. A purge entry, one per dead incarnation, has
    # the incarnation as its id and carries the key and the deletion stamp;
    # the sweeper claims them oldest-first through the stamp's index.
    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_PARTITION_KEY: ClassVar[str] = "partition_key"
    _REGISTRY_INCARNATION: ClassVar[str] = "incarnation"
    _REGISTRY_SCHEMA: ClassVar[str] = "schema"
    _REGISTRY_DELETED_AT: ClassVar[str] = "deleted_at"

    # Fixed UUID namespace for deterministic registry point IDs.
    _REGISTRY_UUID_NAMESPACE: ClassVar[UUID] = UUID(
        "a3c1f6d2-4b8e-4f2a-9c7d-1e5f8a0b3d6c"
    )

    # Keyed by client so locks are garbage-collected when the client is.
    _partition_locks: ClassVar[
        WeakKeyDictionary[
            AsyncQdrantClient,
            defaultdict[tuple[str, str], asyncio.Lock],
        ]
    ] = WeakKeyDictionary()

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 409
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.ALREADY_EXISTS
        if isinstance(error, ValueError):
            return "already exists" in str(error).lower()
        return False

    @staticmethod
    def _is_not_found_error(error: Exception) -> bool:
        """Check if an exception indicates a resource was not found."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 404
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.NOT_FOUND
        if isinstance(error, ValueError):
            return "not found" in str(error).lower()
        return False

    @staticmethod
    def _registry_point_uuid(partition_key: str) -> UUID:
        """Return a deterministic UUID for a partition key."""
        return uuid5(QdrantVectorStore._REGISTRY_UUID_NAMESPACE, partition_key)

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncQdrantClient = params.client
        self._collection = params.collection
        self._vector_dimensions = params.vector_dimensions

        self._registry_replication_factor = params.registry_replication_factor
        self._indexed_properties = params.indexed_properties
        self._hnsw_m = 16

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_qdrant",
        )

        self._client_partition_locks = QdrantVectorStore._partition_locks.setdefault(
            self._client, defaultdict(asyncio.Lock)
        )

    @property
    @override
    def collection(self) -> str:
        return self._collection

    @property
    @override
    def vector_dimensions(self) -> int:
        return self._vector_dimensions

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @property
    def _registry_collection_name(self) -> str:
        return f"{self._collection}{QdrantVectorStore._REGISTRY_SUFFIX}"

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

    @override
    async def provision(self) -> None:
        async with self._tracker("provision"):
            await self._ensure_registry_collection()
            await self._ensure_native_collection()

    @override
    async def startup(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

    async def _ensure_registry_collection(self) -> None:
        """Idempotently create the registry collection and the purge stamp's index."""
        try:
            await self._client.create_collection(
                collection_name=self._registry_collection_name,
                vectors_config=models.VectorParams(
                    size=1,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                ),
                replication_factor=self._registry_replication_factor,
                write_consistency_factor=self._registry_replication_factor,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise
        # The sweeper orders purge entries by their stamp, which needs an index.
        try:
            await self._client.create_payload_index(
                collection_name=self._registry_collection_name,
                field_name=QdrantVectorStore._REGISTRY_DELETED_AT,
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise

    async def _ensure_native_collection(self) -> None:
        """Idempotently create the native Qdrant collection and payload indexes."""
        # The collection and its indexes are created under separate guards. Sharing
        # one meant a collection that already existed - a second worker, or a retry
        # after a crash between the two calls - raised on create_collection, took
        # the already-exists path, and left the collection with no payload indexes
        # at all.
        try:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._vector_dimensions,
                    distance=QdrantVectorStore._QDRANT_DISTANCE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                    payload_m=self._hnsw_m,
                ),
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
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
        for prop_name, prop_type in self._indexed_properties.items():
            index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(prop_type)
            if index_type is not None:
                indexes.append((prop_name, index_type))

        for field_name, field_schema in indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    async def _live_entry(self, partition_key: str) -> dict[str, Any] | None:
        """The partition's live registry entry, or None.

        Verifies the stored key matches to guard against collisions in the
        uuid5 point ID.
        """
        point_uuid = QdrantVectorStore._registry_point_uuid(partition_key)
        try:
            points = await self._client.retrieve(
                collection_name=self._registry_collection_name,
                ids=[point_uuid],
                with_payload=True,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if QdrantVectorStore._is_not_found_error(e):
                return None
            raise

        if not points:
            return None

        payload = cast(dict[str, Any], points[0].payload)
        if payload.get(QdrantVectorStore._REGISTRY_PARTITION_KEY) != partition_key:
            return None
        return payload

    async def _checked_entry(self, partition_key: str) -> dict[str, Any] | None:
        """The partition's live entry, or None; raises if its schema is not this store's."""
        payload = await self._live_entry(partition_key)
        if payload is None:
            return None
        stored_schema = PartitionSchema.model_validate(
            payload[QdrantVectorStore._REGISTRY_SCHEMA]
        )
        declared_schema = self._declared_schema()
        if stored_schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._collection, partition_key, stored_schema, declared_schema
            )
        return payload

    async def _is_live(self, partition_key: str, incarnation: UUID) -> bool:
        """Whether the partition's live entry still names this incarnation."""
        payload = await self._live_entry(partition_key)
        return (
            payload is not None
            and payload.get(QdrantVectorStore._REGISTRY_INCARNATION) == incarnation.hex
        )

    def _partition_handle(
        self, partition_key: str, incarnation: UUID
    ) -> QdrantVectorStorePartition:
        return QdrantVectorStorePartition(
            client=self._client,
            collection_name=self._collection,
            partition_key=partition_key,
            incarnation=incarnation,
            indexed_properties=self._indexed_properties,
            tracker=self._tracker,
            is_live=self._is_live,
        )

    async def _register_partition(self, partition_key: str, incarnation: UUID) -> None:
        """Write the partition's live entry to the registry."""
        await self._client.upsert(
            collection_name=self._registry_collection_name,
            points=[
                models.PointStruct(
                    id=QdrantVectorStore._registry_point_uuid(partition_key),
                    vector=[0.0],
                    payload={
                        QdrantVectorStore._REGISTRY_PARTITION_KEY: partition_key,
                        QdrantVectorStore._REGISTRY_INCARNATION: incarnation.hex,
                        QdrantVectorStore._REGISTRY_SCHEMA: self._declared_schema().model_dump(
                            mode="json"
                        ),
                    },
                ),
            ],
            wait=True,
        )

    @staticmethod
    def _require_partition_key(partition_key: str) -> None:
        if not validate_identifier(partition_key):
            raise ValueError(
                f"Partition key {partition_key!r} must match [a-z0-9_]+ and be at "
                "most 32 bytes"
            )

    @override
    async def create_partition(self, partition_key: str) -> None:
        """Create a partition in the store's collection."""
        QdrantVectorStore._require_partition_key(partition_key)
        # The lock is keyed on the client object, so it serialises callers
        # within a process and not across uvicorn workers.
        async with (
            self._client_partition_locks[(self._collection, partition_key)],
            self._tracker("create_partition"),
        ):
            if await self._checked_entry(partition_key) is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._collection, partition_key
                )
            await self._register_partition(partition_key, uuid4())

    @override
    async def get_partition(
        self, partition_key: str
    ) -> QdrantVectorStorePartition | None:
        QdrantVectorStore._require_partition_key(partition_key)
        payload = await self._checked_entry(partition_key)
        if payload is None:
            return None
        return self._partition_handle(
            partition_key, UUID(payload[QdrantVectorStore._REGISTRY_INCARNATION])
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        QdrantVectorStore._require_partition_key(partition_key)
        async with (
            self._client_partition_locks[(self._collection, partition_key)],
            self._tracker("delete_partition"),
        ):
            payload = await self._live_entry(partition_key)
            if payload is None:
                return
            incarnation = UUID(payload[QdrantVectorStore._REGISTRY_INCARNATION])

            # The live entry goes first, so the partition is unreachable
            # before anything else happens; the purge entry follows. Qdrant
            # has no transactions: a crash between the two leaves the
            # incarnation's points unreachable and unreclaimed, which is a
            # leak, never a partition the sweeper takes from under a live
            # one.
            await self._client.delete(
                collection_name=self._registry_collection_name,
                points_selector=models.PointIdsList(
                    points=[QdrantVectorStore._registry_point_uuid(partition_key)],
                ),
                wait=True,
            )
            await self._client.upsert(
                collection_name=self._registry_collection_name,
                points=[
                    models.PointStruct(
                        id=incarnation,
                        vector=[0.0],
                        payload={
                            QdrantVectorStore._REGISTRY_PARTITION_KEY: partition_key,
                            QdrantVectorStore._REGISTRY_DELETED_AT: int(
                                datetime.now(UTC).timestamp() * 1_000_000
                            ),
                        },
                    ),
                ],
                wait=True,
            )

    @override
    async def purge_deleted_partitions(self) -> bool:
        # One dead incarnation per call, oldest first: the points go by
        # filter, a single server-side operation, then the purge entry.
        # Concurrent purgers may claim the same entry; every step is
        # idempotent, so the loser does empty work.
        async with self._tracker("purge_deleted_partitions"):
            entries, _ = await self._client.scroll(
                collection_name=self._registry_collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=QdrantVectorStore._REGISTRY_DELETED_AT,
                            range=models.Range(gte=0),
                        )
                    ]
                ),
                order_by=models.OrderBy(
                    key=QdrantVectorStore._REGISTRY_DELETED_AT,
                    direction=models.Direction.ASC,
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if not entries:
                return False
            incarnation = UUID(str(entries[0].id))

            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=_partition_filter(incarnation),
                ),
                wait=True,
            )

            await self._client.delete(
                collection_name=self._registry_collection_name,
                points_selector=models.PointIdsList(points=[incarnation]),
                wait=True,
            )
            return True
