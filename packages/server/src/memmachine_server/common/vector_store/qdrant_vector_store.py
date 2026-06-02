"""Qdrant-based vector store implementation."""

import asyncio
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID, uuid5
from weakref import WeakKeyDictionary

import grpc
import grpc.aio
from pydantic import BaseModel, Field, InstanceOf
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

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
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from .utils import validate_filter, validate_identifier
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
        collection_name: str,
        partition_key: str,
        config: VectorStoreCollectionConfig,
        tracker: OperationTracker,
        shard_key: str | None = None,
    ) -> None:
        """Initialize with a Qdrant client and collection name."""
        self._client = client
        self._tracker = tracker
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._config = config
        self._shard_key = shard_key

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        return self._config

    def _build_payload(
        self,
        properties: dict[str, PropertyValue] | None,
    ) -> dict[str, PropertyValue]:
        """Build Qdrant-compatible payload from record properties."""
        payload: dict[str, PropertyValue] = {
            _PAYLOAD_PARTITION_KEY: self._partition_key,
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

    def _parse_payload(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, PropertyValue] | None:
        """Parse record properties from Qdrant payload."""
        if payload is None:
            return None

        indexed_properties_schema = self._config.indexed_properties_schema
        result: dict[str, PropertyValue] = {}
        for key, value in payload.items():
            if key == _PAYLOAD_PARTITION_KEY or value is None:
                continue
            if indexed_properties_schema.get(key) is datetime and isinstance(
                value, str
            ):
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
            points: list[models.PointStruct] = []
            for record in records:
                if record.vector is None:
                    raise ValueError(
                        f"Record {record.uuid} has vector=None, which is not allowed on input."
                    )
                properties = record.properties if record.properties is not None else {}
                points.append(
                    models.PointStruct(
                        id=record.uuid,
                        vector=record.vector,
                        payload=self._build_payload(properties),
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
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        """Query for records matching the criteria by query vectors."""
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            if not query_vectors:
                return []

            partition_key_filter = _partition_filter(self._partition_key)
            if property_filter:
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")
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
                    score_threshold=score_threshold,
                    limit=limit,
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
                        properties = self._parse_payload(point.payload)

                    matches.append(
                        QueryMatch(
                            score=point.score,
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
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
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
                shard_key_selector=self._shard_key,
            )

            points_by_uuid: dict[UUID, models.Record] = {
                UUID(str(point.id)): point
                for point in points
                if point.payload
                and cast(dict[str, Any], point.payload).get(_PAYLOAD_PARTITION_KEY)
                == self._partition_key
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
                    properties = self._parse_payload(
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
            read-your-writes from any available replica.
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

    # Registry collection keys (stored on registry points, one per logical collection)
    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_NAME: ClassVar[str] = "name"
    _REGISTRY_VECTOR_DIMENSIONS: ClassVar[str] = "vector_dimensions"
    _REGISTRY_SIMILARITY_METRIC: ClassVar[str] = "similarity_metric"
    _REGISTRY_INDEXED_PROPERTIES_SCHEMA: ClassVar[str] = "indexed_properties_schema"

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

        self._hnsw_m = 16

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_qdrant",
        )

        self._client_name_locks = QdrantVectorStore._name_locks.setdefault(
            self._client, defaultdict(asyncio.Lock)
        )

    @override
    async def startup(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

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
            similarity_metric=entry[QdrantVectorStore._REGISTRY_SIMILARITY_METRIC],
            indexed_properties_schema=entry[
                QdrantVectorStore._REGISTRY_INDEXED_PROPERTIES_SCHEMA
            ],
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
        distance = QdrantVectorStore._SIMILARITY_METRIC_TO_QDRANT_DISTANCE[
            config.similarity_metric
        ]
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
                sharding_method=(
                    models.ShardingMethod.CUSTOM if self._is_distributed else None
                ),
            )
            await self._client.create_payload_index(
                collection_name=native_collection_name,
                field_name=_PAYLOAD_PARTITION_KEY,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            for prop_name, prop_type in config.indexed_properties_schema.items():
                index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(
                    prop_type
                )
                if index_type is not None:
                    await self._client.create_payload_index(
                        collection_name=native_collection_name,
                        field_name=prop_name,
                        field_schema=index_type,
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
                        QdrantVectorStore._REGISTRY_SIMILARITY_METRIC: config.similarity_metric.value,
                        QdrantVectorStore._REGISTRY_INDEXED_PROPERTIES_SCHEMA: config.model_dump(
                            mode="json"
                        )["indexed_properties_schema"],
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
