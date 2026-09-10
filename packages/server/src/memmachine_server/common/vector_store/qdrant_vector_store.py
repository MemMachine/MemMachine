"""Qdrant-based vector store implementation."""

import asyncio
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar, Self, cast, override
from uuid import UUID, uuid5
from weakref import WeakKeyDictionary

import grpc
import grpc.aio
from pydantic import BaseModel, Field, InstanceOf, model_validator
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

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
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
    OrderingOp,
)
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    IndexedProperties,
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from .declared_properties import require_declared_properties, require_supported_filter
from .utils import validate_identifier
from .vector_store import VectorStore, VectorStoreCollection

# Point payload keys (stored on every Qdrant point).
# System keys use _SYSTEM_KEY_PREFIX, which contains a hyphen. Hyphens are valid in
# Qdrant but forbidden by _IDENTIFIER_RE, so system keys can never collide with user keys.
_SYSTEM_KEY_PREFIX = "sys-"
_PAYLOAD_PARTITION_KEY = f"{_SYSTEM_KEY_PREFIX}partition_key"


def _partition_filter(partition_key: str) -> models.Filter:
    """Build a Qdrant filter that matches the given partition key."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PAYLOAD_PARTITION_KEY,
                match=models.MatchValue(value=partition_key),
            ),
        ],
    )


class QdrantVectorStoreCollection(VectorStoreCollection):
    """A collection backed by Qdrant."""

    _RANGE_OPERATORS: ClassVar[dict[OrderingOp, str]] = {
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
    }

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {Equals, NotEquals, Ordering, In, IsMissing, And, Or, Not}
    )

    @staticmethod
    def _build_qdrant_filter(expr: FilterExpr) -> models.Filter:
        """Convert a FilterExpr tree into a Qdrant Filter."""
        build = QdrantVectorStoreCollection._build_qdrant_filter
        eq_condition = QdrantVectorStoreCollection._eq_condition
        missing_condition = QdrantVectorStoreCollection._missing_condition
        match expr:
            case Equals(field, value):
                return models.Filter(must=[eq_condition(field, value)])
            case NotEquals(field, value):
                # `must_not` of the match alone would also admit points
                # lacking the field; a differing value is one the point holds.
                return models.Filter(
                    must_not=[eq_condition(field, value), missing_condition(field)]
                )
            case Ordering(field, op, value):
                return models.Filter(
                    must=[
                        QdrantVectorStoreCollection._range_condition(
                            field,
                            value,
                            QdrantVectorStoreCollection._RANGE_OPERATORS[op],
                        )
                    ]
                )
            case In(field, values):
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field, match=models.MatchAny(any=list(values))
                        )
                    ]
                )
            case IsMissing(field):
                return models.Filter(must=[missing_condition(field)])
            case Not(operand):
                return models.Filter(must_not=[build(operand)])
            case And(operands):
                return models.Filter(must=[build(o) for o in operands])
            case Or(operands):
                return models.Filter(should=[build(o) for o in operands])

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
        collection_name: str,
        partition_key: str,
        config: VectorStoreCollectionConfig,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
        shard_key: str | None = None,
    ) -> None:
        """Initialize with a Qdrant client and collection name."""
        self._client = client
        self._tracker = tracker
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._config = config
        self._indexed_properties = dict(indexed_properties)
        self._shard_key = shard_key

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        return self._config

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return QdrantVectorStoreCollection._SUPPORTED_FILTER_NODES

    def _build_payload(
        self,
        properties: Mapping[str, PropertyValue],
    ) -> dict[str, PropertyValue]:
        """Build Qdrant-compatible payload from record properties."""
        payload: dict[str, PropertyValue] = {
            _PAYLOAD_PARTITION_KEY: self._partition_key,
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
            points: list[models.PointStruct] = []
            for record in records:
                require_declared_properties(record.properties, self._indexed_properties)
                points.append(
                    models.PointStruct(
                        id=record.uuid,
                        vector=record.vector,
                        payload=self._build_payload(record.properties),
                    )
                )
            if points:
                await self._upsert_with_backoff(points)

    async def _upsert_with_backoff(self, points: Iterable[models.PointStruct]) -> None:
        """Upsert points, splitting the batch in half on failure and retrying."""
        points = list(points)
        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                shard_key_selector=self._shard_key,
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

            partition_key_filter = _partition_filter(self._partition_key)
            if property_filter is not None:
                require_supported_filter(
                    property_filter,
                    self._indexed_properties,
                    QdrantVectorStoreCollection._SUPPORTED_FILTER_NODES,
                )
                property_qdrant_filter = (
                    QdrantVectorStoreCollection._build_qdrant_filter(property_filter)
                )
                qdrant_filter = models.Filter(
                    must=[partition_key_filter, property_qdrant_filter]
                )
            else:
                qdrant_filter = partition_key_filter

            requests = [
                models.QueryRequest(
                    shard_key=self._shard_key,
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

            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            _partition_filter(self._partition_key),
                            models.HasIdCondition(
                                has_id=list(uuid_list),
                            ),
                        ],
                    ),
                ),
                shard_key_selector=self._shard_key,
            )


class QdrantVectorStoreParams(BaseModel):
    """
    Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.
        is_distributed (bool):
            Whether the Qdrant cluster is running in distributed mode.
            If True, native collections use custom sharding
            so each logical collection maps to a dedicated shard key.
            This enables logical collection deletion via shard drop
            instead of filter-based deletion.
        registry_replication_factor (int):
            Replication factor for registry collections. Write consistency factor is
            set to match so all replicas confirm writes before returning, guaranteeing
            read-your-writes from any available replica
            (default: 1).
        indexed_properties (IndexedProperties):
            The declared schema every collection of this store carries: each
            key gets a payload index of its declared type, and a record or a
            filter naming any other key is rejected.
        hnsw_config (HnswConfigDiff | None):
            Optional HNSW index tuning applied to native collections.
            `m` must be 0 or unset: native collections are multi-tenant
            and disable the global graph in favor of per-tenant payload indexing,
            so tune `payload_m` rather than `m`.
            Does not apply to registry collections
            (default: None).
        optimizers_config (OptimizersConfigDiff | None):
            Optional optimizer tuning applied to native collections.
            Does not apply to registry collections
            (default: None).
        quantization_config (QuantizationConfig | None):
            Optional quantization applied to native collections.
            Does not apply to registry collections
            (default: None).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )
    is_distributed: bool = Field(
        False,
        description=(
            "Whether the Qdrant cluster is running in distributed mode. "
            "If True, native collections use custom sharding "
            "so each logical collection maps to a dedicated shard key. "
            "This enables logical collection deletion via shard drop "
            "instead of filter-based deletion"
        ),
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
        description="The declared schema every collection of this store carries",
    )
    hnsw_config: models.HnswConfigDiff | None = Field(
        None,
        description=(
            "Optional HNSW index tuning applied to native collections. "
            "`m` must be 0 or unset: native collections are multi-tenant "
            "and disable the global graph in favor of per-tenant payload indexing, "
            "so tune `payload_m` rather than `m`. "
            "Does not apply to registry collections"
        ),
    )
    optimizers_config: models.OptimizersConfigDiff | None = Field(
        None,
        description=(
            "Optional optimizer tuning applied to native collections. "
            "Does not apply to registry collections"
        ),
    )
    quantization_config: models.QuantizationConfig | None = Field(
        None,
        description=(
            "Optional quantization applied to native collections. "
            "Does not apply to registry collections"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )

    @model_validator(mode="after")
    def _validate_hnsw_m(self) -> Self:
        if self.hnsw_config is not None and self.hnsw_config.m not in (None, 0):
            raise ValueError(
                "hnsw_config.m must be 0 or unset: native collections are "
                "multi-tenant and disable the global graph in favor of per-tenant "
                "payload indexing, so tune payload_m rather than m"
            )
        return self


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore."""

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

    # Registry collection keys (stored on registry points, one per logical collection)
    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_NAME: ClassVar[str] = "name"
    _REGISTRY_VECTOR_DIMENSIONS: ClassVar[str] = "vector_dimensions"

    # The per-tenant graph size when no override is configured.
    _DEFAULT_NATIVE_PAYLOAD_M: ClassVar[int] = 16

    # Fixed UUID namespace for deterministic registry point IDs.
    _REGISTRY_UUID_NAMESPACE: ClassVar[UUID] = UUID(
        "a3c1f6d2-4b8e-4f2a-9c7d-1e5f8a0b3d6c"
    )

    # Keyed by client so locks are garbage-collected when the client is.
    _name_locks: ClassVar[
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
    def _registry_collection_name(namespace: str) -> str:
        """Return the registry collection name for a namespace."""
        return f"{namespace}{QdrantVectorStore._REGISTRY_SUFFIX}"

    @staticmethod
    def _registry_point_uuid(name: str) -> UUID:
        """Return a deterministic UUID for a logical collection name."""
        return uuid5(QdrantVectorStore._REGISTRY_UUID_NAMESPACE, name)

    @staticmethod
    def _build_native_collection_name(
        namespace: str, config: VectorStoreCollectionConfig
    ) -> str:
        """Build a deterministic native collection name from namespace and config."""
        digest = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        return f"{namespace}__{digest}"

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncQdrantClient = params.client
        self._is_distributed = params.is_distributed

        self._registry_replication_factor = params.registry_replication_factor
        self._indexed_properties = params.indexed_properties
        self._hnsw_config = params.hnsw_config
        self._optimizers_config = params.optimizers_config
        self._quantization_config = params.quantization_config

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_qdrant",
        )

        self._client_name_locks = QdrantVectorStore._name_locks.setdefault(
            self._client, defaultdict(asyncio.Lock)
        )

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @override
    async def startup(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

    def _native_hnsw_config(self) -> models.HnswConfigDiff:
        """The HNSW config of a native collection: the overrides, with `m` pinned at 0.

        A native collection is multi-tenant, so the global graph is disabled
        and each tenant gets its own graph of `payload_m` links.
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

    async def _ensure_namespace_registry_collection(self, namespace: str) -> None:
        """Idempotently create the registry collection for a namespace."""
        registry_collection_name = QdrantVectorStore._registry_collection_name(
            namespace
        )
        try:
            await self._client.create_collection(
                collection_name=registry_collection_name,
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

    async def _get_registry_entry(
        self, namespace: str, name: str
    ) -> dict[str, Any] | None:
        """
        Retrieve the registry entry for a logical collection name.

        Verifies the stored name matches
        to guard against SHA-1 collisions in the uuid5 point ID.
        """
        registry_collection_name = QdrantVectorStore._registry_collection_name(
            namespace
        )
        point_uuid = QdrantVectorStore._registry_point_uuid(name)
        try:
            points = await self._client.retrieve(
                collection_name=registry_collection_name,
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
        if payload.get(QdrantVectorStore._REGISTRY_NAME) != name:
            return None

        return payload

    @staticmethod
    def _parse_entry(entry: Mapping[str, Any]) -> VectorStoreCollectionConfig:
        """Parse a VectorStoreCollectionConfig from a registry entry."""
        return VectorStoreCollectionConfig(
            vector_dimensions=entry[QdrantVectorStore._REGISTRY_VECTOR_DIMENSIONS],
        )

    def _build_collection_handle(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> QdrantVectorStoreCollection:
        """Build a QdrantVectorStoreCollection handle."""
        return QdrantVectorStoreCollection(
            client=self._client,
            collection_name=QdrantVectorStore._build_native_collection_name(
                namespace, config
            ),
            partition_key=name,
            config=config,
            indexed_properties=self._indexed_properties,
            tracker=self._tracker,
            shard_key=name if self._is_distributed else None,
        )

    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Idempotently create the native Qdrant collection and payload indexes."""
        native_collection_name = QdrantVectorStore._build_native_collection_name(
            namespace, config
        )
        distance = QdrantVectorStore._QDRANT_DISTANCE
        # The collection and its indexes are created under separate guards. Sharing
        # one meant a collection that already existed - a second worker, or a retry
        # after a crash between the two calls - raised on create_collection, took
        # the already-exists path, and left the collection with no payload indexes
        # at all. The lock above is keyed on the client object, so it serialises
        # callers within a process and not across uvicorn workers.
        try:
            await self._client.create_collection(
                collection_name=native_collection_name,
                vectors_config=models.VectorParams(
                    size=config.vector_dimensions, distance=distance
                ),
                hnsw_config=self._native_hnsw_config(),
                optimizers_config=self._optimizers_config,
                quantization_config=self._quantization_config,
                sharding_method=(
                    models.ShardingMethod.CUSTOM if self._is_distributed else None
                ),
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise

        indexes: list[tuple[str, Any]] = [
            (
                _PAYLOAD_PARTITION_KEY,
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
                    collection_name=native_collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    async def _ensure_shard_key(
        self, native_collection_name: str, shard_key: str
    ) -> None:
        """Idempotently create a shard key on a native collection."""
        try:
            await self._client.create_shard_key(
                native_collection_name, shard_key=shard_key
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
            if "already exists" not in str(e).lower():
                raise

    async def _register_collection(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Write the logical collection entry to the registry."""
        registry_name = QdrantVectorStore._registry_collection_name(namespace)
        point_uuid = QdrantVectorStore._registry_point_uuid(name)
        await self._client.upsert(
            collection_name=registry_name,
            points=[
                models.PointStruct(
                    id=point_uuid,
                    vector=[0.0],
                    payload={
                        QdrantVectorStore._REGISTRY_NAME: name,
                        QdrantVectorStore._REGISTRY_VECTOR_DIMENSIONS: config.vector_dimensions,
                    },
                ),
            ],
            wait=True,
        )

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        """Create a logical collection in the Qdrant vector store."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        async with (
            self._client_name_locks[(namespace, name)],
            self._tracker("create_collection"),
        ):
            await self._ensure_namespace_registry_collection(namespace)
            if await self._get_registry_entry(namespace, name) is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name)
            await self._create_native_collection(namespace, config)
            if self._is_distributed:
                native_collection_name = (
                    QdrantVectorStore._build_native_collection_name(namespace, config)
                )
                await self._ensure_shard_key(native_collection_name, name)
            await self._register_collection(namespace, name, config)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> QdrantVectorStoreCollection:
        """Open the collection if it exists, or create and return it."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        async with (
            self._client_name_locks[(namespace, name)],
            self._tracker("open_or_create_collection"),
        ):
            entry = await self._get_registry_entry(namespace, name)
            if entry is not None:
                existing_config = QdrantVectorStore._parse_entry(entry)
                if existing_config != config:
                    raise VectorStoreCollectionConfigMismatchError(
                        namespace, name, existing_config, config
                    )
                return self._build_collection_handle(namespace, name, existing_config)

            await self._ensure_namespace_registry_collection(namespace)
            await self._create_native_collection(namespace, config)
            if self._is_distributed:
                native_collection_name = (
                    QdrantVectorStore._build_native_collection_name(namespace, config)
                )
                await self._ensure_shard_key(native_collection_name, name)
            await self._register_collection(namespace, name, config)
            return self._build_collection_handle(namespace, name, config)

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> QdrantVectorStoreCollection | None:
        """Get a collection handle from the vector store."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        entry = await self._get_registry_entry(namespace, name)
        if entry is None:
            return None
        return self._build_collection_handle(
            namespace, name, QdrantVectorStore._parse_entry(entry)
        )

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        """No-op; Qdrant collection handles require no explicit close."""

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """Delete a logical collection from the Qdrant vector store."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        async with (
            self._client_name_locks[(namespace, name)],
            self._tracker("delete_collection"),
        ):
            entry = await self._get_registry_entry(namespace, name)
            if entry is None:
                return

            config = QdrantVectorStore._parse_entry(entry)
            native_collection_name = QdrantVectorStore._build_native_collection_name(
                namespace, config
            )

            # Delete partition data, then registry entry.
            if self._is_distributed:
                try:
                    await self._client.delete_shard_key(
                        native_collection_name, shard_key=name
                    )
                except (UnexpectedResponse, grpc.aio.AioRpcError) as e:
                    if "does not exist" not in str(e).lower():
                        raise
            else:
                await self._client.delete(
                    collection_name=native_collection_name,
                    points_selector=models.FilterSelector(
                        filter=_partition_filter(name),
                    ),
                )

            registry_name = QdrantVectorStore._registry_collection_name(namespace)
            point_uuid = QdrantVectorStore._registry_point_uuid(name)
            await self._client.delete(
                collection_name=registry_name,
                points_selector=models.PointIdsList(
                    points=[point_uuid],
                ),
                wait=True,
            )
