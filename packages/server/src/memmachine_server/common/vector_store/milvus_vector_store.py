"""Milvus-based vector store implementation."""

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID, uuid4
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field, InstanceOf, field_validator
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

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
from memmachine_server.common.properties_json import (
    encode_properties,
)
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
from .utils import validate_filter, validate_identifier
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
_PROPERTIES_FIELD = "properties"
_PROPERTY_FILTER_PREFIX = "_p_"

_MAX_UUID_LENGTH = 36
_MAX_PRIMARY_ID_LENGTH = 128
_MAX_PARTITION_KEY_LENGTH = 32
_INCARNATION_HEX_LENGTH = 32
_REGISTRY_VECTOR_DIMENSION = 2
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
        is_live: Callable[[str, UUID], Awaitable[bool]],
    ) -> None:
        """Initialize with a Milvus client and the incarnation the handle is bound to."""
        self._client = client
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._incarnation = incarnation
        self._indexed_properties = dict(indexed_properties)
        self._tracker = tracker
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the partition's.

        Milvus has no transactions, so the check and the operation are two
        calls; a deletion landing between them leaves entities under a dead
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

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(self._incarnation, record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: self._incarnation.hex,
            _VECTOR_FIELD: record.vector,
            _PROPERTIES_FIELD: encode_properties(record.properties),
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
            entities = [self._build_entity(record) for record in records]

            def _upsert() -> None:
                self._client.upsert(
                    collection_name=self._collection_name,
                    data=entities,
                )

            await asyncio.to_thread(_upsert)

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
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")
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
            )


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
        consistency_level (str): Consistency level for the collections this store creates.
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key is a dynamic field a search filters on.
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
    consistency_level: str = Field(
        default="Session",
        description="Milvus consistency level for the collections this store creates",
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


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore.

    The store is one native Milvus collection, named at construction, in
    which every partition is a partition-key value. A registry collection
    beside it records which partitions exist and what they were created
    under.
    """

    _MILVUS_METRIC_TYPE: ClassVar[str] = "COSINE"

    # The registry collection holds two kinds of entity. A live entry, one
    # per partition, has the partition key as its id and carries the
    # incarnation and the schema. A purge entry, one per dead incarnation,
    # has the incarnation as its id and carries the key and the deletion
    # stamp, which orders the sweeper's claims.
    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_SCHEMA: ClassVar[str] = "schema"
    _REGISTRY_PARTITION_KEY: ClassVar[str] = "partition_key"
    _REGISTRY_INCARNATION: ClassVar[str] = "incarnation"
    _REGISTRY_DELETED_AT: ClassVar[str] = "deleted_at"
    # How many purge entries a claim reads to pick the oldest: Milvus queries
    # do not order, so the choice is made here over a bounded page.
    _PURGE_CLAIM_PAGE: ClassVar[int] = 100

    _partition_locks: ClassVar[
        WeakKeyDictionary[
            MilvusClient,
            defaultdict[tuple[str, str], asyncio.Lock],
        ]
    ] = WeakKeyDictionary()

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        message = str(error).lower()
        return "already exist" in message or "already exists" in message

    @staticmethod
    def _is_not_found_error(error: Exception) -> bool:
        """Check if an exception indicates a resource was not found."""
        message = str(error).lower()
        return "not found" in message or "can't find" in message

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client = params.client
        self._collection = params.collection
        self._vector_dimensions = params.vector_dimensions
        self._consistency_level = params.consistency_level
        self._indexed_properties = params.indexed_properties
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_milvus",
        )
        self._client_partition_locks = MilvusVectorStore._partition_locks.setdefault(
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
        return f"{self._collection}{MilvusVectorStore._REGISTRY_SUFFIX}"

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
        """Idempotently create the registry collection."""
        registry_collection_name = self._registry_collection_name
        if await asyncio.to_thread(
            self._client.has_collection, registry_collection_name
        ):
            return

        def _create_registry() -> None:
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            # A live entry's id is the partition key; a purge entry's is the
            # incarnation hex.
            schema.add_field(
                field_name=_ID_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=max(_MAX_PARTITION_KEY_LENGTH, _INCARNATION_HEX_LENGTH),
            )
            schema.add_field(
                field_name=_VECTOR_FIELD,
                datatype=DataType.FLOAT_VECTOR,
                dim=_REGISTRY_VECTOR_DIMENSION,
            )
            schema.add_field(
                field_name=self._REGISTRY_SCHEMA,
                datatype=DataType.JSON,
            )
            schema.add_field(
                field_name=self._REGISTRY_PARTITION_KEY,
                datatype=DataType.VARCHAR,
                max_length=_MAX_PARTITION_KEY_LENGTH,
            )
            schema.add_field(
                field_name=self._REGISTRY_INCARNATION,
                datatype=DataType.VARCHAR,
                max_length=_INCARNATION_HEX_LENGTH,
            )
            schema.add_field(
                field_name=self._REGISTRY_DELETED_AT,
                datatype=DataType.INT64,
            )

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name=_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type=MilvusVectorStore._MILVUS_METRIC_TYPE,
            )

            self._client.create_collection(
                collection_name=registry_collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self._consistency_level,
            )

        try:
            await asyncio.to_thread(_create_registry)
        except MilvusException as exc:
            if not MilvusVectorStore._is_already_exists_error(exc):
                raise

    async def _ensure_native_collection(self) -> None:
        """Idempotently create the native Milvus collection."""
        if await asyncio.to_thread(self._client.has_collection, self._collection):
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
            schema.add_field(
                field_name=_PROPERTIES_FIELD,
                datatype=DataType.JSON,
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
            )

        try:
            await asyncio.to_thread(_create_collection)
        except MilvusException as exc:
            if not MilvusVectorStore._is_already_exists_error(exc):
                raise

    async def _live_entry(self, partition_key: str) -> dict[str, Any] | None:
        """The partition's live registry entry, or None."""
        registry_collection_name = self._registry_collection_name
        if not await asyncio.to_thread(
            self._client.has_collection, registry_collection_name
        ):
            return None

        output_fields = [
            _ID_FIELD,
            self._REGISTRY_SCHEMA,
            self._REGISTRY_INCARNATION,
            self._REGISTRY_DELETED_AT,
        ]
        try:
            result = await asyncio.to_thread(
                self._client.get,
                collection_name=registry_collection_name,
                ids=[partition_key],
                output_fields=output_fields,
            )
        except MilvusException as exc:
            if MilvusVectorStore._is_not_found_error(exc):
                return None
            raise

        entries = list(result)
        if not entries:
            return None
        entry = cast(dict[str, Any], entries[0])
        entry_id = entry.get(_ID_FIELD)
        if entry_id is not None and entry_id != partition_key:
            return None
        if entry.get(self._REGISTRY_SCHEMA) is None:
            # Older clients may not include every field in get() output
            # unless queried.
            rows = await asyncio.to_thread(
                self._client.query,
                collection_name=registry_collection_name,
                filter=f"{_ID_FIELD} == {_expr_string(partition_key)}",
                output_fields=output_fields,
            )
            rows = list(rows)
            if not rows:
                return None
            entry = cast(dict[str, Any], rows[0])
        if entry.get(self._REGISTRY_DELETED_AT):
            # A purge entry whose incarnation collides with a key: not live.
            return None
        return entry

    async def _checked_entry(self, partition_key: str) -> dict[str, Any] | None:
        """The partition's live entry, or None; raises if its schema is not this store's."""
        entry = await self._live_entry(partition_key)
        if entry is None:
            return None
        stored_schema = PartitionSchema.model_validate(entry[self._REGISTRY_SCHEMA])
        declared_schema = self._declared_schema()
        if stored_schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._collection, partition_key, stored_schema, declared_schema
            )
        return entry

    async def _is_live(self, partition_key: str, incarnation: UUID) -> bool:
        """Whether the partition's live entry still names this incarnation."""
        entry = await self._live_entry(partition_key)
        return (
            entry is not None
            and entry.get(self._REGISTRY_INCARNATION) == incarnation.hex
        )

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
            is_live=self._is_live,
        )

    async def _register_partition(self, partition_key: str, incarnation: UUID) -> None:
        """Write the partition's live entry to the registry."""
        await asyncio.to_thread(
            self._client.insert,
            collection_name=self._registry_collection_name,
            data=[
                {
                    _ID_FIELD: partition_key,
                    _VECTOR_FIELD: [0.0] * _REGISTRY_VECTOR_DIMENSION,
                    self._REGISTRY_SCHEMA: self._declared_schema().model_dump(
                        mode="json"
                    ),
                    self._REGISTRY_PARTITION_KEY: partition_key,
                    self._REGISTRY_INCARNATION: incarnation.hex,
                    self._REGISTRY_DELETED_AT: 0,
                }
            ],
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
        MilvusVectorStore._require_partition_key(partition_key)
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
    ) -> MilvusVectorStorePartition | None:
        MilvusVectorStore._require_partition_key(partition_key)
        entry = await self._checked_entry(partition_key)
        if entry is None:
            return None
        return self._partition_handle(
            partition_key, UUID(entry[self._REGISTRY_INCARNATION])
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        MilvusVectorStore._require_partition_key(partition_key)
        async with (
            self._client_partition_locks[(self._collection, partition_key)],
            self._tracker("delete_partition"),
        ):
            entry = await self._live_entry(partition_key)
            if entry is None:
                return
            incarnation = UUID(entry[self._REGISTRY_INCARNATION])

            # The live entry goes first, so the partition is unreachable
            # before anything else happens; the purge entry follows. Milvus
            # has no transactions: a crash between the two leaves the
            # incarnation's entities unreachable and unreclaimed, which is a
            # leak, never a partition the sweeper takes from under a live
            # one.
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._registry_collection_name,
                ids=[partition_key],
            )
            await asyncio.to_thread(
                self._client.insert,
                collection_name=self._registry_collection_name,
                data=[
                    {
                        _ID_FIELD: incarnation.hex,
                        _VECTOR_FIELD: [0.0] * _REGISTRY_VECTOR_DIMENSION,
                        self._REGISTRY_SCHEMA: {},
                        self._REGISTRY_PARTITION_KEY: partition_key,
                        self._REGISTRY_INCARNATION: incarnation.hex,
                        self._REGISTRY_DELETED_AT: int(
                            datetime.now(UTC).timestamp() * 1_000_000
                        ),
                    }
                ],
            )

    @override
    async def purge_deleted_partitions(self) -> bool:
        # One dead incarnation per call, the oldest of a bounded page of
        # purge entries: its entities go by filter, a single server-side
        # operation, then the entry. Concurrent purgers may claim the same
        # entry; every step is idempotent, so the loser does empty work.
        async with self._tracker("purge_deleted_partitions"):
            if not await asyncio.to_thread(
                self._client.has_collection, self._registry_collection_name
            ):
                return False
            rows = list(
                await asyncio.to_thread(
                    self._client.query,
                    collection_name=self._registry_collection_name,
                    filter=f"{self._REGISTRY_DELETED_AT} > 0",
                    output_fields=[_ID_FIELD, self._REGISTRY_DELETED_AT],
                    limit=self._PURGE_CLAIM_PAGE,
                )
            )
            if not rows:
                return False
            oldest = min(rows, key=lambda row: row[self._REGISTRY_DELETED_AT])
            incarnation_hex = str(oldest[_ID_FIELD])

            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection,
                filter=f"{_PARTITION_KEY_FIELD} == {_expr_string(incarnation_hex)}",
            )
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._registry_collection_name,
                ids=[incarnation_hex],
            )
            return True
