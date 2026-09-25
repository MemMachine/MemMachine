"""Qdrant-based vector store implementation."""

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

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

from .collection_registry import RegisteredCollection, VectorStoreCollectionRegistry
from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStoreCollectionHandleStaleError,
)
from .utils import require_identifiers, validate_filter
from .vector_store import VectorStore, VectorStoreCollection

# Point payload keys (stored on every Qdrant point).
# System keys use _SYSTEM_KEY_PREFIX, which contains a hyphen. Hyphens are valid in
# Qdrant but forbidden by _IDENTIFIER_RE, so system keys can never collide with user keys.
_SYSTEM_KEY_PREFIX = "sys-"
_PAYLOAD_INCARNATION = f"{_SYSTEM_KEY_PREFIX}incarnation"
"""The payload key naming the collection incarnation a point belongs to.

Points carry the incarnation, never the collection's name: a collection
deleted and re-created under the same name gets a fresh incarnation, and
its predecessor's points are invisible to it while the purge reclaims them.
"""

# Consecutive lost creation races before open-or-create gives up: every
# retry requires another process to have created and then deleted the
# collection in between, so this depth means something else is wrong.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10


def _incarnation_filter(incarnation: UUID) -> models.Filter:
    """Build a Qdrant filter that matches the points of one collection incarnation."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PAYLOAD_INCARNATION,
                match=models.MatchValue(value=incarnation.hex),
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
        native_collection_name: str,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
        tracker: OperationTracker,
        is_live: Callable[[UUID], Awaitable[bool]],
    ) -> None:
        """Initialize with a Qdrant client and the incarnation the handle is bound to."""
        self._client = client
        self._tracker = tracker
        self._native_collection_name = native_collection_name
        self._namespace = namespace
        self._name = name
        self._incarnation = incarnation
        self._config = config
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the collection's.

        Called before every operation, to refuse a handle known to be
        dead, and after a write, so a write completed under an incarnation
        that died meanwhile raises instead of reporting success. Qdrant has
        no transactions, so a write can still land under a dead
        incarnation: between the two checks, or after a check that never
        ran; the tombstone's purge rounds reclaim it. A read is not checked
        after: a collection deleted while a read is in flight keeps its
        points until a purge round claims its tombstone, so the read returns
        what it saw, a snapshot from before the deletion, as a read that
        happened to run just before it would have.
        """
        if not await self._is_live(self._incarnation):
            raise VectorStoreCollectionHandleStaleError(self._namespace, self._name)

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

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
            if key == _PAYLOAD_INCARNATION or value is None:
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
        async with self._tracker("upsert"):
            await self._fence()
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
            await self._fence()

    async def _upsert_with_backoff(self, points: Iterable[models.PointStruct]) -> None:
        """Upsert points, splitting the batch in half on failure and retrying."""
        points = list(points)
        try:
            await self._client.upsert(
                collection_name=self._native_collection_name,
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
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            if not query_vectors:
                return []

            await self._fence()
            partition_key_filter = _incarnation_filter(self._incarnation)
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
                collection_name=self._native_collection_name,
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
        async with self._tracker("get"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return []

            await self._fence()
            # Always get payload so we can check the incarnation.
            points = await self._client.retrieve(
                collection_name=self._native_collection_name,
                ids=list(uuid_list),
                with_vectors=return_vector,
                with_payload=True,
            )

            points_by_uuid: dict[UUID, models.Record] = {
                UUID(str(point.id)): point
                for point in points
                if point.payload
                and cast(dict[str, Any], point.payload).get(_PAYLOAD_INCARNATION)
                == self._incarnation.hex
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
        async with self._tracker("delete"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return

            await self._fence()
            await self._client.delete(
                collection_name=self._native_collection_name,
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
        collection_registry (VectorStoreCollectionRegistry):
            The registry of the Qdrant deployment the client reaches: which
            collections exist, under which incarnation and configuration,
            and which dead incarnations await purge. Qdrant arbitrates none
            of that, so the registry lives where a primary key and a
            transaction can. Every store on the deployment, in any
            process, uses this registry, and no store on another
            deployment does. The caller starts it before handing it over.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).

    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )
    collection_registry: InstanceOf[VectorStoreCollectionRegistry] = Field(
        ...,
        description="The registry of the deployment the client reaches",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore.

    A logical collection is a payload value, the incarnation of its life,
    inside a native collection shared by the logical collections of one
    namespace and configuration. The catalog is the `VectorStoreCollectionRegistry`
    the store is given: it mints the incarnations and arbitrates creation,
    deletion and reclamation across processes, which Qdrant, with no
    transactions or unique constraints, cannot. Any process sharing the
    Qdrant backend and the registry may serve any collection.
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

        self._collection_registry = params.collection_registry

        self._hnsw_m = 16

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_qdrant",
        )

    @override
    async def startup(self) -> None:
        # The caller owns the client's and the registry's lifecycles.
        pass

    @override
    async def shutdown(self) -> None:
        # The caller owns the client's and the registry's lifecycles.
        pass

    def _build_collection_handle(
        self, namespace: str, name: str, registered: RegisteredCollection
    ) -> QdrantVectorStoreCollection:
        """Build a QdrantVectorStoreCollection handle bound to the registered incarnation."""
        return QdrantVectorStoreCollection(
            client=self._client,
            native_collection_name=QdrantVectorStore._build_native_collection_name(
                namespace, registered.config
            ),
            namespace=namespace,
            name=name,
            incarnation=registered.incarnation,
            config=registered.config,
            tracker=self._tracker,
            is_live=self._collection_registry.is_live,
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
        # The collection and its indexes are created under separate guards. Sharing
        # one meant a collection that already existed - a second worker, or a retry
        # after a crash between the two calls - raised on create_collection, took
        # the already-exists path, and left the collection with no payload indexes
        # at all.
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
            except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("create_collection"):
            # The native collection first, the registry row last: a crash
            # between the two leaves an empty native collection the next
            # creation of the same configuration adopts, never a row whose
            # points have nowhere to go. The registry's primary key is the
            # arbiter: a racing creator on any process loses here, never in
            # Qdrant.
            await self._create_native_collection(namespace, config)
            await self._collection_registry.register(namespace, name, config)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> QdrantVectorStoreCollection:
        require_identifiers(namespace, name)
        async with self._tracker("open_or_create_collection"):
            attempts = 0
            # Read-then-create, retried: losing the create means a racing
            # creator won (open its row), and finding no row after losing
            # means a racing deleter removed the winner (create again).
            while True:
                registered = await self._collection_registry.get(namespace, name)
                if registered is not None:
                    if registered.config != config:
                        raise VectorStoreCollectionConfigMismatchError(
                            namespace, name, registered.config, config
                        )
                    return self._build_collection_handle(namespace, name, registered)
                await self._create_native_collection(namespace, config)
                try:
                    incarnation = await self._collection_registry.register(
                        namespace, name, config
                    )
                except VectorStoreCollectionAlreadyExistsError as err:
                    attempts += 1
                    if attempts >= _MAX_OPEN_OR_CREATE_ATTEMPTS:
                        raise VectorStoreAttemptsExhaustedError(
                            f"Opening or creating collection ({namespace!r}, "
                            f"{name!r}) made no progress after "
                            f"{_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
                        ) from err
                    continue
                return self._build_collection_handle(
                    namespace,
                    name,
                    RegisteredCollection(incarnation=incarnation, config=config),
                )

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> QdrantVectorStoreCollection | None:
        require_identifiers(namespace, name)
        registered = await self._collection_registry.get(namespace, name)
        if registered is None:
            return None
        return self._build_collection_handle(namespace, name, registered)

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        # Qdrant collection handles hold nothing to release.
        pass

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("delete_collection"):
            # One registry transaction: the collection is unreachable when
            # it commits, and its points wait on the queue for the purge.
            await self._collection_registry.unregister(namespace, name)

    @override
    async def purge_deleted_collections(self) -> bool:
        # One purge round per call, on the tombstone that came due first: the
        # claim is a row lock the registry holds while one point is looked for
        # and, if there is one, the points go by filter in a single
        # server-side operation. The registry keeps or removes the tombstone
        # by what the round found.
        async with (
            self._tracker("purge_deleted_collections"),
            self._collection_registry.claim_purgeable_incarnation() as claim,
        ):
            if claim is None:
                return False
            native_collection_name = QdrantVectorStore._build_native_collection_name(
                claim.namespace, claim.config
            )
            try:
                points, _ = await self._client.scroll(
                    collection_name=native_collection_name,
                    scroll_filter=_incarnation_filter(claim.incarnation),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
                # The native collection is gone with everything in it.
                if not QdrantVectorStore._is_not_found_error(e):
                    raise
                points = []
            claim.found = bool(points)
            if claim.found:
                await self._client.delete(
                    collection_name=native_collection_name,
                    points_selector=models.FilterSelector(
                        filter=_incarnation_filter(claim.incarnation),
                    ),
                    wait=True,
                )
            return claim.found
