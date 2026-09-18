"""Milvus-based vector store implementation."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, cast, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, field_validator
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.data_types import PropertyType, PropertyValue
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
from memmachine_server.common.utils import compute_cosine_similarity, ensure_tz_aware

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
from .declared_properties import require_declared_properties, require_supported_filter
from .sql_partition_registry import RegisteredPartition, SqlPartitionRegistry
from .utils import validate_identifier
from .vector_store import VectorStore, VectorStorePartition

_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
"""The native partition-key field; holds the incarnation, never the caller's key.

A partition deleted and re-created under the same key gets a fresh
incarnation, and its predecessor's entities are invisible to it while the
purge reclaims them.
"""
_VECTOR_FIELD = "vector"
_PROPERTY_FILTER_PREFIX = "_p_"

_MAX_UUID_LENGTH = 36
_MAX_PRIMARY_ID_LENGTH = 128
_MAX_PARTITION_KEY_LENGTH = 32
_INCARNATION_HEX_LENGTH = 32
_FALSE_EXPR = f'{_ID_FIELD} == "__memmachine_no_match__"'


def _expr_string(value: str) -> str:
    """Return a Milvus expression string literal."""
    return json.dumps(value)


def _literal(value: PropertyValue) -> str:
    """Return a Milvus expression literal for a property value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, datetime):
        return _expr_string(ensure_tz_aware(value).astimezone(UTC).isoformat())
    return _expr_string(value)


def _property_field(field: str) -> str:
    """Return the dynamic Milvus field used for property filtering."""
    return f"{_PROPERTY_FILTER_PREFIX}{field}"


def _normalize_property_filter_value(value: PropertyValue) -> PropertyValue:
    """Normalize property values stored in dynamic filter fields."""
    if isinstance(value, datetime):
        return ensure_tz_aware(value).astimezone(UTC).isoformat()
    return value


class MilvusVectorStorePartition(VectorStorePartition):
    """A partition backed by Milvus: one partition-key value inside the store's collection."""

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {FilterComparison, FilterIn, FilterIsNull, FilterAnd, FilterOr, FilterNot}
    )

    _RANGE_OPERATORS: ClassVar[set[str]] = {">", ">=", "<", "<="}

    @staticmethod
    def _build_milvus_filter(expr: FilterExpr) -> str:
        """Convert a FilterExpr tree into a Milvus filter expression."""
        if isinstance(expr, FilterComparison):
            return MilvusVectorStorePartition._build_milvus_comparison(expr)
        if isinstance(expr, FilterIn):
            if not expr.values:
                return _FALSE_EXPR
            values = ", ".join(_literal(value) for value in expr.values)
            return f"{_property_field(expr.field)} in [{values}]"
        if isinstance(expr, FilterIsNull):
            return f"{_property_field(expr.field)} is null"
        if isinstance(expr, FilterNot):
            return f"not ({MilvusVectorStorePartition._build_milvus_filter(expr.expr)})"
        if isinstance(expr, FilterAnd):
            left = MilvusVectorStorePartition._build_milvus_filter(expr.left)
            right = MilvusVectorStorePartition._build_milvus_filter(expr.right)
            return f"({left}) && ({right})"
        if isinstance(expr, FilterOr):
            left = MilvusVectorStorePartition._build_milvus_filter(expr.left)
            right = MilvusVectorStorePartition._build_milvus_filter(expr.right)
            return f"({left}) || ({right})"
        message = f"Unsupported filter expression type: {type(expr)}"
        raise TypeError(message)

    @staticmethod
    def _build_milvus_comparison(comparison: FilterComparison) -> str:
        """Convert a Comparison into a Milvus filter expression."""
        field = _property_field(comparison.field)
        operator = "==" if comparison.op == "=" else comparison.op
        return f"{field} {operator} {_literal(comparison.value)}"

    @staticmethod
    def _primary_id(incarnation: UUID, record_uuid: UUID) -> str:
        """Build a native primary key unique within a shared native collection."""
        return f"{incarnation.hex}:{record_uuid}"

    def __init__(
        self,
        *,
        client: MilvusClient,
        collection_name: str,
        partition_key: str,
        incarnation: UUID,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
        is_live: Callable[[UUID], Awaitable[bool]],
        request_timeout_seconds: int,
    ) -> None:
        """Initialize with a Milvus client and the incarnation the handle is bound to."""
        self._client = client
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._incarnation = incarnation
        self._indexed_properties = dict(indexed_properties)
        self._tracker = tracker
        self._request_timeout_seconds = request_timeout_seconds
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the partition's.

        Called before an operation, to refuse a handle known to be dead,
        and after it, so an operation completed under an incarnation that
        died meanwhile raises instead of reporting success. Milvus has no
        transactions, so a write can still land under a dead incarnation:
        between the two checks, or after a check that never ran; the
        tombstone's purge rounds reclaim it.
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

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return MilvusVectorStorePartition._SUPPORTED_FILTER_NODES

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(self._incarnation, record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: self._incarnation.hex,
            _VECTOR_FIELD: record.vector,
        }
        # Explicit nulls clear stale dynamic fields during native Milvus upserts.
        for key in self._indexed_properties:
            entity[_property_field(key)] = None
        for key, value in record.properties.items():
            entity[_property_field(key)] = _normalize_property_filter_value(value)
        return entity

    def _output_fields(self) -> list[str]:
        # The vector always comes back: search scores are recomputed from it.
        return [_RECORD_UUID_FIELD, _VECTOR_FIELD]

    @staticmethod
    def _cosine_similarity_from_entity_vector(
        query_vector: Sequence[float],
        entity: Mapping[str, Any],
    ) -> float:
        raw_vector = entity.get(_VECTOR_FIELD)
        if raw_vector is None:
            raise ValueError("Milvus search result did not include the vector field")
        return compute_cosine_similarity(
            list(query_vector),
            [list(cast(Sequence[float], raw_vector))],
        )[0]

    def _partition_filter(self) -> str:
        return f"{_PARTITION_KEY_FIELD} == {_expr_string(self._incarnation.hex)}"

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        async with self._tracker("upsert"):
            records = list(records)
            if not records:
                return

            await self._fence()
            for record in records:
                require_declared_properties(record.properties, self._indexed_properties)
            entities = [self._build_entity(record) for record in records]

            def _upsert() -> None:
                self._client.upsert(
                    collection_name=self._collection_name,
                    data=entities,
                    timeout=self._request_timeout_seconds,
                )

            await asyncio.to_thread(_upsert)
            await self._fence()

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
            if limit <= 0:
                return [QueryResult(matches=[]) for _ in query_vectors]

            await self._fence()
            filter_expr = self._partition_filter()
            if property_filter is not None:
                require_supported_filter(
                    property_filter,
                    self._indexed_properties,
                    MilvusVectorStorePartition._SUPPORTED_FILTER_NODES,
                )
                property_expr = self._build_milvus_filter(property_filter)
                filter_expr = f"({filter_expr}) && ({property_expr})"

            raw_results = await asyncio.to_thread(
                self._client.search,
                collection_name=self._collection_name,
                data=query_vectors,
                filter=filter_expr,
                limit=limit,
                # Milvus Lite returns COSINE as distance, while Zilliz Cloud
                # returns it as similarity. Fetch vectors and compute scores
                # locally so MemMachine score semantics stay consistent.
                output_fields=self._output_fields(),
                anns_field=_VECTOR_FIELD,
                timeout=self._request_timeout_seconds,
            )

            results: list[QueryResult] = []
            for query_vector, raw_matches in zip(
                query_vectors, raw_results, strict=True
            ):
                matches: list[QueryMatch] = []
                for raw_match in raw_matches:
                    entity = cast(Mapping[str, Any], raw_match["entity"])
                    cosine_similarity = self._cosine_similarity_from_entity_vector(
                        query_vector,
                        entity,
                    )
                    if (
                        min_cosine_similarity is not None
                        and cosine_similarity < min_cosine_similarity
                    ):
                        continue

                    matches.append(
                        QueryMatch(
                            cosine_similarity=cosine_similarity,
                            record_uuid=UUID(str(entity[_RECORD_UUID_FIELD])),
                        )
                    )

                matches.sort(key=lambda match: match.cosine_similarity, reverse=True)
                results.append(QueryResult(matches=matches))

            await self._fence()
            return results

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
            primary_ids = [
                self._primary_id(self._incarnation, uuid) for uuid in uuid_list
            ]
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection_name,
                ids=primary_ids,
                timeout=self._request_timeout_seconds,
            )
            await self._fence()


class MilvusVectorStoreParams(BaseModel):
    """
    Parameters for MilvusVectorStore.

    Attributes:
        client (MilvusClient): Milvus client instance.
        collection (str):
            The collection this store is; the native Milvus collection's
            name, so stores of different collections may share the client.
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        registry_engine (AsyncEngine):
            The relational database holding the partition registry: which
            partitions exist, under which incarnation, and which dead
            incarnations await purge. Milvus arbitrates none of that, so
            the registry lives where a primary key and a transaction can.
        tombstone_retention_seconds (int):
            How long a deleted partition's registry entry outlives the
            first purge round that finds nothing under it, so a write that
            landed after that round is still reclaimed; orders of magnitude
            above the request timeout.
        consistency_level (str): Consistency level for the collection this store creates.
        request_timeout_seconds (int): Seconds any request to Milvus may take.
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key is a dynamic field a search filters on, and a record or a
            filter naming any other key is rejected.
        metrics_factory (MetricsFactory | None): Metrics factory for collecting usage metrics.
    """

    client: InstanceOf[MilvusClient] = Field(
        ...,
        description="Milvus client instance",
    )
    collection: str = Field(
        ...,
        description="The collection this store is; the native Milvus collection's name",
    )
    vector_dimensions: int = Field(
        ..., gt=0, description="Dimensionality of every vector in the store"
    )
    registry_engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="The relational database holding the partition registry",
    )
    tombstone_retention_seconds: int = Field(
        ...,
        gt=0,
        description=(
            "Seconds a deleted partition's tombstone outlives the first purge "
            "round that finds nothing under it"
        ),
    )
    consistency_level: str = Field(
        default="Session",
        description="Milvus consistency level for the collection this store creates",
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every partition of this store carries",
    )
    request_timeout_seconds: int = Field(
        ..., gt=0, description="Seconds any request to Milvus may take"
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


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore.

    The store is one native Milvus collection, named at construction, in
    which every partition is a partition-key value: the incarnation of its
    life, minted by the registry. The registry is `SqlPartitionRegistry`
    in the deployment's relational database, shared by every Milvus store
    there: it arbitrates creation and deletion across processes, which
    Milvus, with no transactions or unique constraints, cannot.
    """

    _MILVUS_METRIC_TYPE: ClassVar[str] = "COSINE"

    # Every Milvus store in one relational database shares these tables.
    _REGISTRY_TABLE_PREFIX: ClassVar[str] = "vector_store_milvus"

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        message = str(error).lower()
        return "already exist" in message or "already exists" in message

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client = params.client
        self._collection = params.collection
        self._vector_dimensions = params.vector_dimensions
        self._consistency_level = params.consistency_level
        self._request_timeout_seconds = params.request_timeout_seconds
        self._registry = SqlPartitionRegistry(
            engine=params.registry_engine,
            table_prefix=MilvusVectorStore._REGISTRY_TABLE_PREFIX,
            collection=self._collection,
            tombstone_retention=timedelta(seconds=params.tombstone_retention_seconds),
        )
        self._indexed_properties = params.indexed_properties
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_milvus",
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

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

    @override
    async def provision(self) -> None:
        async with self._tracker("provision"):
            await self._registry.provision()
            await self._ensure_native_collection()

    @override
    async def startup(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

    async def _ensure_native_collection(self) -> None:
        """Idempotently create the native Milvus collection."""
        if await asyncio.to_thread(
            self._client.has_collection,
            self._collection,
            timeout=self._request_timeout_seconds,
        ):
            return

        def _create_collection() -> None:
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(
                field_name=_ID_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=_MAX_PRIMARY_ID_LENGTH,
            )
            schema.add_field(
                field_name=_RECORD_UUID_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_MAX_UUID_LENGTH,
            )
            schema.add_field(
                field_name=_PARTITION_KEY_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_INCARNATION_HEX_LENGTH,
                is_partition_key=True,
            )
            schema.add_field(
                field_name=_VECTOR_FIELD,
                datatype=DataType.FLOAT_VECTOR,
                dim=self._vector_dimensions,
            )

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name=_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type=MilvusVectorStore._MILVUS_METRIC_TYPE,
            )

            self._client.create_collection(
                collection_name=self._collection,
                schema=schema,
                index_params=index_params,
                consistency_level=self._consistency_level,
                timeout=self._request_timeout_seconds,
            )

        try:
            await asyncio.to_thread(_create_collection)
        except MilvusException as exc:
            if not MilvusVectorStore._is_already_exists_error(exc):
                raise

    async def _checked_entry(self, partition_key: str) -> RegisteredPartition | None:
        """The live partition under the key, or None; raises if its schema is not this store's."""
        registered = await self._registry.get(partition_key)
        if registered is None:
            return None
        declared_schema = self._declared_schema()
        if registered.schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._collection, partition_key, registered.schema, declared_schema
            )
        return registered

    def _partition_handle(
        self, partition_key: str, incarnation: UUID
    ) -> MilvusVectorStorePartition:
        return MilvusVectorStorePartition(
            client=self._client,
            collection_name=self._collection,
            partition_key=partition_key,
            incarnation=incarnation,
            indexed_properties=self._indexed_properties,
            tracker=self._tracker,
            is_live=self._registry.is_live,
            request_timeout_seconds=self._request_timeout_seconds,
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
        MilvusVectorStore._require_partition_key(partition_key)
        async with self._tracker("create_partition"):
            # The registry's primary key is the arbiter: a racing creator
            # on any process loses here, never in Milvus.
            try:
                await self._registry.create(partition_key, self._declared_schema())
            except VectorStorePartitionAlreadyExistsError:
                # A key taken under another schema is reported as such.
                await self._checked_entry(partition_key)
                raise

    @override
    async def get_partition(
        self, partition_key: str
    ) -> MilvusVectorStorePartition | None:
        MilvusVectorStore._require_partition_key(partition_key)
        registered = await self._checked_entry(partition_key)
        if registered is None:
            return None
        return self._partition_handle(partition_key, registered.incarnation)

    @override
    async def delete_partition(self, partition_key: str) -> None:
        MilvusVectorStore._require_partition_key(partition_key)
        async with self._tracker("delete_partition"):
            # One registry transaction: the partition is unreachable when
            # it commits, and its entities wait on the queue for the purge.
            await self._registry.delete(partition_key)

    @override
    async def purge_deleted_partitions(self) -> bool:
        # One purge round per call, on the oldest tombstone due: the claim
        # is a row lock the registry holds while one entity is looked for
        # and, if there is one, the entities go by filter in a single
        # server-side operation. The registry keeps or removes the
        # tombstone by what the round found.
        async with (
            self._tracker("purge_deleted_partitions"),
            self._registry.claim_oldest() as claim,
        ):
            if claim is None:
                return False
            partition_filter = (
                f"{_PARTITION_KEY_FIELD} == {_expr_string(claim.incarnation.hex)}"
            )
            held = await asyncio.to_thread(
                self._client.query,
                collection_name=self._collection,
                filter=partition_filter,
                output_fields=[_ID_FIELD],
                limit=1,
                timeout=self._request_timeout_seconds,
            )
            claim.found = bool(list(held))
            if claim.found:
                await asyncio.to_thread(
                    self._client.delete,
                    collection_name=self._collection,
                    filter=partition_filter,
                    timeout=self._request_timeout_seconds,
                )
            return claim.found
