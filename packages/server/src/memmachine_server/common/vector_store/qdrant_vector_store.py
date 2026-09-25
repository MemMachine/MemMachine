"""Qdrant-based vector store implementation."""

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar, Self, override
from uuid import UUID

import grpc
import grpc.aio
from pydantic import BaseModel, Field, InstanceOf, field_validator, model_validator
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
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionSchemaMismatchError,
    indexed_property_names,
    validate_vector_store_name,
)
from .partition_registry import RegisteredPartition, VectorStorePartitionRegistry
from .utils import require_partition_key, validate_filter
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

# Consecutive lost creation races before open-or-create gives up: every
# retry requires another process to have created and then deleted the
# partition in between, so this depth means something else is wrong.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10


def _incarnation_filter(incarnation: UUID) -> models.Filter:
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
        vector_dimensions: int,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
        is_live: Callable[[UUID], Awaitable[bool]],
    ) -> None:
        """Initialize with a Qdrant client and the incarnation the handle is bound to."""
        self._client = client
        self._tracker = tracker
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._incarnation = incarnation
        self._vector_dimensions = vector_dimensions
        self._indexed_properties = dict(indexed_properties)
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the partition's.

        Called before every operation, to refuse a handle known to be
        dead, and after a write, so a write completed under an incarnation
        that died meanwhile raises instead of reporting success. Qdrant has
        no transactions, so a write can still land under a dead
        incarnation: between the two checks, or after a check that never
        ran; the tombstone's purge rounds reclaim it. A read is not checked
        after: a partition deleted while a read is in flight keeps its
        points until a purge round claims its tombstone, so the read returns
        what it saw, a snapshot from before the deletion, as a read that
        happened to run just before it would have.
        """
        if not await self._is_live(self._incarnation):
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
        properties: dict[str, PropertyValue] | None,
    ) -> dict[str, PropertyValue]:
        """Build Qdrant-compatible payload from record properties."""
        payload: dict[str, PropertyValue] = {
            _PAYLOAD_INCARNATION: self._incarnation.hex,
        }
        if properties:
            for key, value in properties.items():
                if value is None:
                    continue
                if isinstance(value, datetime):
                    payload[key] = ensure_tz_aware(value)
                else:
                    payload[key] = value
        return payload

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        async with self._tracker("upsert"):
            await self._fence()
            points: list[models.PointStruct] = []
            for record in records:
                properties = record.properties
                points.append(
                    models.PointStruct(
                        id=record.uuid,
                        vector=record.vector,
                        payload=self._build_payload(properties),
                    )
                )
            if points:
                await self._upsert_with_backoff(points)
            await self._fence()

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
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            if not query_vectors:
                return []

            await self._fence()
            partition_filter = _incarnation_filter(self._incarnation)
            if property_filter:
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")
                property_qdrant_filter = (
                    QdrantVectorStorePartition._build_qdrant_filter(property_filter)
                )
                qdrant_filter = models.Filter(
                    must=[partition_filter, property_qdrant_filter]
                )
            else:
                qdrant_filter = partition_filter

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
                            _incarnation_filter(self._incarnation),
                            models.HasIdCondition(
                                has_id=list(uuid_list),
                            ),
                        ],
                    ),
                ),
            )
            await self._fence()


class QdrantVectorStoreParams(BaseModel):
    """
    Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.
        partition_registry (VectorStorePartitionRegistry):
            The registry of this store's partitions: which exist, under which
            incarnation and schema, and which dead incarnations await purge.
            Qdrant arbitrates none of that, so the registry lives where a
            primary key and a transaction can. Every process serving this
            store uses this registry, and no other store does. The caller
            provisions it before handing it over.
        vector_store_name (str):
            The name of this store, and of the native Qdrant collection it
            is, so stores of different names may share the client.
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key gets a payload index of its declared type.
        hnsw_config (HnswConfigDiff | None):
            Optional HNSW index tuning applied to the store's collection.
            `m` must be 0 or unset: the collection holds every partition
            and disables the global graph in favor of per-partition payload
            indexing, so tune `payload_m` rather than `m`
            (default: None).
        optimizers_config (OptimizersConfigDiff | None):
            Optional optimizer tuning applied to the store's collection
            (default: None).
        quantization_config (QuantizationConfig | None):
            Optional quantization applied to the store's collection
            (default: None).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).

    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )
    partition_registry: InstanceOf[VectorStorePartitionRegistry] = Field(
        ...,
        description="The registry of this store's partitions",
    )
    vector_store_name: str = Field(
        ...,
        description="The name of this store; the native Qdrant collection's name",
    )
    vector_dimensions: int = Field(
        ..., gt=0, description="Dimensionality of every vector in the store"
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every partition of this store carries",
    )
    hnsw_config: models.HnswConfigDiff | None = Field(
        None,
        description=(
            "Optional HNSW index tuning applied to the store's collection. "
            "`m` must be 0 or unset: the collection holds every partition and "
            "disables the global graph in favor of per-partition payload "
            "indexing, so tune `payload_m` rather than `m`"
        ),
    )
    optimizers_config: models.OptimizersConfigDiff | None = Field(
        None,
        description=("Optional optimizer tuning applied to the store's collection"),
    )
    quantization_config: models.QuantizationConfig | None = Field(
        None,
        description=("Optional quantization applied to the store's collection"),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )

    @field_validator("vector_store_name")
    @classmethod
    def _validate_vector_store_name(cls, vector_store_name: str) -> str:
        validate_vector_store_name(vector_store_name)
        return vector_store_name

    @model_validator(mode="after")
    def _validate_hnsw_m(self) -> Self:
        if self.hnsw_config is not None and self.hnsw_config.m not in (None, 0):
            raise ValueError(
                "hnsw_config.m must be 0 or unset: the collection holds every "
                "partition and disables the global graph in favor of "
                "per-partition payload indexing, so tune payload_m rather than m"
            )
        return self


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore.

    The store is one native Qdrant collection, named at construction, in
    which every partition is a payload value: the incarnation of its life,
    minted by the `VectorStorePartitionRegistry` the store is given, which
    arbitrates creation, deletion and reclamation across processes, as
    Qdrant, with no transactions or unique constraints, cannot. Any process
    sharing the Qdrant backend and the registry may serve any partition.
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

    # The per-tenant graph size when no override is configured.
    _DEFAULT_NATIVE_PAYLOAD_M: ClassVar[int] = 16

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

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncQdrantClient = params.client
        self._vector_store_name = params.vector_store_name
        self._vector_dimensions = params.vector_dimensions
        self._indexed_properties = params.indexed_properties

        self._partition_registry = params.partition_registry

        self._hnsw_config = params.hnsw_config
        self._optimizers_config = params.optimizers_config
        self._quantization_config = params.quantization_config

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_qdrant",
        )

    @property
    @override
    def vector_store_name(self) -> str:
        return self._vector_store_name

    @property
    @override
    def vector_dimensions(self) -> int:
        return self._vector_dimensions

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

    def _native_hnsw_config(self) -> models.HnswConfigDiff:
        """The HNSW config of the native collection: the overrides, with `m` pinned at 0.

        The collection is multi-tenant, so the global graph is disabled and
        each partition gets its own graph of `payload_m` links.
        """
        overrides = self._hnsw_config or models.HnswConfigDiff()
        return overrides.model_copy(
            update={
                "m": 0,
                "payload_m": overrides.payload_m
                if overrides.payload_m is not None
                else QdrantVectorStore._DEFAULT_NATIVE_PAYLOAD_M,
            }
        )

    @override
    async def provision(self) -> None:
        async with self._tracker("provision"):
            await self._ensure_native_collection()

    @override
    async def startup(self) -> None:
        # The caller owns the client's and the registry's lifecycles.
        pass

    @override
    async def shutdown(self) -> None:
        # The caller owns the client's and the registry's lifecycles.
        pass

    async def _ensure_native_collection(self) -> None:
        """Idempotently create the native Qdrant collection and payload indexes."""
        distance = QdrantVectorStore._QDRANT_DISTANCE
        # The collection and its indexes are created under separate guards. Sharing
        # one meant a collection that already existed - a second worker, or a retry
        # after a crash between the two calls - raised on create_collection, took
        # the already-exists path, and left the collection with no payload indexes
        # at all.
        try:
            await self._client.create_collection(
                collection_name=self._vector_store_name,
                vectors_config=models.VectorParams(
                    size=self._vector_dimensions, distance=distance
                ),
                hnsw_config=self._native_hnsw_config(),
                optimizers_config=self._optimizers_config,
                quantization_config=self._quantization_config,
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
                    collection_name=self._vector_store_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    async def _checked_entry(self, partition_key: str) -> RegisteredPartition | None:
        """The live partition under the key, or None; raises if its schema is not this store's."""
        registered = await self._partition_registry.get(partition_key)
        if registered is None:
            return None
        declared_schema = self._declared_schema()
        if registered.schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._vector_store_name,
                partition_key,
                registered.schema,
                declared_schema,
            )
        return registered

    def _partition_handle(
        self, partition_key: str, incarnation: UUID
    ) -> QdrantVectorStorePartition:
        return QdrantVectorStorePartition(
            client=self._client,
            collection_name=self._vector_store_name,
            partition_key=partition_key,
            incarnation=incarnation,
            vector_dimensions=self._vector_dimensions,
            indexed_properties=self._indexed_properties,
            tracker=self._tracker,
            is_live=self._partition_registry.is_live,
        )

    @override
    async def create_partition(self, partition_key: str) -> None:
        require_partition_key(partition_key)
        async with self._tracker("create_partition"):
            # The registry's primary key is the arbiter: a racing creator
            # on any process loses here, never in Qdrant.
            try:
                await self._partition_registry.register(
                    partition_key, self._declared_schema()
                )
            except VectorStorePartitionAlreadyExistsError:
                # A key taken under another schema is reported as such.
                await self._checked_entry(partition_key)
                raise

    @override
    async def open_or_create_partition(
        self, partition_key: str
    ) -> QdrantVectorStorePartition:
        require_partition_key(partition_key)
        async with self._tracker("open_or_create_partition"):
            attempts = 0
            # Read-then-create, retried: losing the create means a racing
            # creator won (open its row), and finding no row after losing
            # means a racing deleter removed the winner (create again).
            while True:
                registered = await self._checked_entry(partition_key)
                if registered is not None:
                    return self._partition_handle(partition_key, registered.incarnation)
                try:
                    incarnation = await self._partition_registry.register(
                        partition_key, self._declared_schema()
                    )
                except VectorStorePartitionAlreadyExistsError as err:
                    attempts += 1
                    if attempts >= _MAX_OPEN_OR_CREATE_ATTEMPTS:
                        raise VectorStoreAttemptsExhaustedError(
                            f"Opening or creating partition {partition_key!r} of "
                            f"vector store {self._vector_store_name!r} made no progress after "
                            f"{_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
                        ) from err
                    continue
                return self._partition_handle(partition_key, incarnation)

    @override
    async def get_partition(
        self, partition_key: str
    ) -> QdrantVectorStorePartition | None:
        require_partition_key(partition_key)
        registered = await self._checked_entry(partition_key)
        if registered is None:
            return None
        return self._partition_handle(partition_key, registered.incarnation)

    @override
    async def close_partition(self, *, partition: VectorStorePartition) -> None:
        # Qdrant partition handles hold nothing to release.
        pass

    @override
    async def delete_partition(self, partition_key: str) -> None:
        require_partition_key(partition_key)
        async with self._tracker("delete_partition"):
            # One registry transaction: the partition is unreachable when
            # it commits, and its points wait on the queue for the purge.
            await self._partition_registry.unregister(partition_key)

    @override
    async def purge_deleted_partitions(self) -> bool:
        # One purge round per call, on the tombstone that came due first: the
        # claim is a row lock the registry holds while one point is looked for
        # and, if there is one, the points go by filter in a single
        # server-side operation. The registry keeps or removes the tombstone
        # by what the round found.
        async with (
            self._tracker("purge_deleted_partitions"),
            self._partition_registry.claim_due() as claim,
        ):
            if claim is None:
                return False
            points, _ = await self._client.scroll(
                collection_name=self._vector_store_name,
                scroll_filter=_incarnation_filter(claim.incarnation),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            claim.found = bool(points)
            if claim.found:
                await self._client.delete(
                    collection_name=self._vector_store_name,
                    points_selector=models.FilterSelector(
                        filter=_incarnation_filter(claim.incarnation),
                    ),
                    wait=True,
                )
            return claim.found
