"""Milvus-based vector store implementation."""

import asyncio
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID
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
    VectorStorePartitionSchemaMismatchError,
    indexed_property_names,
    validate_collection_name,
)
from .utils import validate_filter, validate_identifier
from .vector_store import VectorStore, VectorStorePartition

_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
_VECTOR_FIELD = "vector"
_PROPERTIES_FIELD = "properties"
_PROPERTY_FILTER_PREFIX = "_p_"

_MAX_UUID_LENGTH = 36
_MAX_PRIMARY_ID_LENGTH = 128
_MAX_PARTITION_KEY_LENGTH = 32
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
    def _primary_id(partition_key: str, record_uuid: UUID) -> str:
        """Build a native primary key unique within a shared native collection."""
        return f"{partition_key}:{record_uuid}"

    def __init__(
        self,
        *,
        client: MilvusClient,
        collection_name: str,
        partition_key: str,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Milvus client and the collection and partition it is bound to."""
        self._client = client
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._indexed_properties = dict(indexed_properties)
        self._tracker = tracker

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
            _ID_FIELD: self._primary_id(self._partition_key, record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: self._partition_key,
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
        return f"{_PARTITION_KEY_FIELD} == {_expr_string(self._partition_key)}"

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
            primary_ids = [
                self._primary_id(self._partition_key, uuid) for uuid in uuid_list
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

    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_SCHEMA: ClassVar[str] = "schema"

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
            schema.add_field(
                field_name=_ID_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=_MAX_PARTITION_KEY_LENGTH,
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
                max_length=_MAX_PARTITION_KEY_LENGTH,
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

    async def _stored_schema(self, partition_key: str) -> PartitionSchema | None:
        """The schema the partition was created under; raises if it is not this store's."""
        registry_collection_name = self._registry_collection_name
        if not await asyncio.to_thread(
            self._client.has_collection, registry_collection_name
        ):
            return None

        try:
            result = await asyncio.to_thread(
                self._client.get,
                collection_name=registry_collection_name,
                ids=[partition_key],
                output_fields=[_ID_FIELD, self._REGISTRY_SCHEMA],
            )
        except MilvusException as exc:
            if MilvusVectorStore._is_not_found_error(exc):
                return None
            raise

        entries = list(result)
        if not entries:
            return None
        entry = entries[0]
        entry_id = entry.get(_ID_FIELD)
        if entry_id is not None and entry_id != partition_key:
            return None
        stored = cast(dict[str, Any] | None, entry.get(self._REGISTRY_SCHEMA))
        if stored is None:
            # Older clients may not include the primary key in get() output unless queried.
            rows = await asyncio.to_thread(
                self._client.query,
                collection_name=registry_collection_name,
                filter=f"{_ID_FIELD} == {_expr_string(partition_key)}",
                output_fields=[_ID_FIELD, self._REGISTRY_SCHEMA],
            )
            rows = list(rows)
            if not rows:
                return None
            stored = cast(dict[str, Any], rows[0][self._REGISTRY_SCHEMA])

        stored_schema = PartitionSchema.model_validate(stored)
        declared_schema = self._declared_schema()
        if stored_schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._collection, partition_key, stored_schema, declared_schema
            )
        return stored_schema

    def _partition_handle(self, partition_key: str) -> MilvusVectorStorePartition:
        return MilvusVectorStorePartition(
            client=self._client,
            collection_name=self._collection,
            partition_key=partition_key,
            indexed_properties=self._indexed_properties,
            tracker=self._tracker,
        )

    async def _register_partition(self, partition_key: str) -> None:
        """Write the partition's entry to the registry."""
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
            if await self._stored_schema(partition_key) is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._collection, partition_key
                )
            await self._register_partition(partition_key)

    @override
    async def get_partition(
        self, partition_key: str
    ) -> MilvusVectorStorePartition | None:
        """Get a handle bound to an existing partition."""
        MilvusVectorStore._require_partition_key(partition_key)
        if await self._stored_schema(partition_key) is None:
            return None
        return self._partition_handle(partition_key)

    @override
    async def delete_partition(self, partition_key: str) -> None:
        """Delete a partition and its records from the store's collection."""
        MilvusVectorStore._require_partition_key(partition_key)
        async with (
            self._client_partition_locks[(self._collection, partition_key)],
            self._tracker("delete_partition"),
        ):
            if await self._stored_schema(partition_key) is None:
                return

            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._collection,
                filter=f"{_PARTITION_KEY_FIELD} == {_expr_string(partition_key)}",
            )
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._registry_collection_name,
                ids=[partition_key],
            )
