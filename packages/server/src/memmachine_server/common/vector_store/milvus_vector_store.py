"""Milvus-based vector store implementation."""

import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field, InstanceOf
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from memmachine_server.common.data_types import PropertyType, PropertyValue
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
from memmachine_server.common.utils import compute_cosine_similarity, ensure_tz_aware

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

_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
_VECTOR_FIELD = "vector"
_PROPERTY_FILTER_PREFIX = "_p_"

_INVERSE_ORDERING: dict[OrderingOp, OrderingOp] = {
    ">": "<=",
    ">=": "<",
    "<": ">=",
    "<=": ">",
}

_MAX_UUID_LENGTH = 36
_MAX_PRIMARY_ID_LENGTH = 128
_MAX_PARTITION_KEY_LENGTH = 32
_REGISTRY_VECTOR_DIMENSION = 2


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


class MilvusVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by Milvus."""

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {Equals, NotEquals, Ordering, In, IsMissing, And, Or, Not}
    )

    @staticmethod
    def _build_milvus_filter(expr: FilterExpr) -> str:
        """Convert a FilterExpr tree into a Milvus filter expression."""
        build = MilvusVectorStoreCollection._build_milvus_filter
        match expr:
            case Equals(field, value):
                return f"{_property_field(field)} == {_literal(value)}"
            case NotEquals(field, value):
                return f"{_property_field(field)} != {_literal(value)}"
            case Ordering(field, op, value):
                return f"{_property_field(field)} {op} {_literal(value)}"
            case In(field, values):
                literals = ", ".join(_literal(value) for value in values)
                return f"{_property_field(field)} in [{literals}]"
            case IsMissing(field):
                return f"{_property_field(field)} is null"
            case Not(operand):
                return MilvusVectorStoreCollection._negated(operand)
            case And(operands):
                return " && ".join(f"({build(o)})" for o in operands)
            case Or(operands):
                return " || ".join(f"({build(o)})" for o in operands)

    @staticmethod
    def _negated(expr: FilterExpr) -> str:
        """The complement of a tree as a Milvus expression, pushed to the leaves.

        Milvus's own `not` does not admit entities lacking the field, so a
        negated predicate is rendered as the inverse predicate or the field's
        absence, which is the complement the filter language defines.
        """
        build = MilvusVectorStoreCollection._build_milvus_filter
        negated = MilvusVectorStoreCollection._negated
        match expr:
            case Equals(field, value):
                return (
                    f"({_property_field(field)} != {_literal(value)}) || "
                    f"({_property_field(field)} is null)"
                )
            case NotEquals(field, value):
                return (
                    f"({_property_field(field)} == {_literal(value)}) || "
                    f"({_property_field(field)} is null)"
                )
            case Ordering(field, op, value):
                return (
                    f"({_property_field(field)} {_INVERSE_ORDERING[op]} {_literal(value)}) || "
                    f"({_property_field(field)} is null)"
                )
            case In(field, values):
                literals = ", ".join(_literal(value) for value in values)
                return (
                    f"({_property_field(field)} not in [{literals}]) || "
                    f"({_property_field(field)} is null)"
                )
            case IsMissing(field):
                return f"{_property_field(field)} is not null"
            case Not(operand):
                return build(operand)
            case And(operands):
                return " || ".join(f"({negated(o)})" for o in operands)
            case Or(operands):
                return " && ".join(f"({negated(o)})" for o in operands)

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
        config: VectorStoreCollectionConfig,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Milvus client and collection name."""
        self._client = client
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._config = config
        self._indexed_properties = dict(indexed_properties)
        self._tracker = tracker

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
        return MilvusVectorStoreCollection._SUPPORTED_FILTER_NODES

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(self._partition_key, record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: self._partition_key,
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

            for record in records:
                require_declared_properties(record.properties, self._indexed_properties)
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
                require_supported_filter(
                    property_filter,
                    self._indexed_properties,
                    MilvusVectorStoreCollection._SUPPORTED_FILTER_NODES,
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
        consistency_level (str): Collection consistency level for newly created collections.
        indexed_properties (IndexedProperties):
            The declared schema every collection of this store carries: each
            key is a dynamic field a search filters on, and a record or a
            filter naming any other key is rejected.
        metrics_factory (MetricsFactory | None): Metrics factory for collecting usage metrics.
    """

    client: InstanceOf[MilvusClient] = Field(
        ...,
        description="Milvus client instance",
    )
    consistency_level: str = Field(
        default="Session",
        description="Milvus consistency level for newly created collections",
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every collection of this store carries",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore."""

    _MILVUS_METRIC_TYPE: ClassVar[str] = "COSINE"

    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_VECTOR_DIMENSIONS: ClassVar[str] = "vector_dimensions"
    _REGISTRY_CONFIG: ClassVar[str] = "config"

    _name_locks: ClassVar[
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

    @staticmethod
    def _registry_collection_name(namespace: str) -> str:
        """Return the registry collection name for a namespace."""
        return f"memmachine_{namespace}{MilvusVectorStore._REGISTRY_SUFFIX}"

    @staticmethod
    def _build_native_collection_name(
        namespace: str, config: VectorStoreCollectionConfig
    ) -> str:
        """Build a deterministic native collection name from namespace and config."""
        digest = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        return f"memmachine_{namespace}__{digest}"

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client = params.client
        self._consistency_level = params.consistency_level
        self._indexed_properties = params.indexed_properties
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_milvus",
        )
        self._client_name_locks = MilvusVectorStore._name_locks.setdefault(
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

    async def _ensure_namespace_registry_collection(self, namespace: str) -> None:
        """Idempotently create the registry collection for a namespace."""
        registry_collection_name = MilvusVectorStore._registry_collection_name(
            namespace
        )
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
                field_name=self._REGISTRY_CONFIG,
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

    async def _get_registry_entry(
        self, namespace: str, name: str
    ) -> dict[str, Any] | None:
        """Retrieve the registry entry for a logical collection name."""
        registry_collection_name = MilvusVectorStore._registry_collection_name(
            namespace
        )
        if not await asyncio.to_thread(
            self._client.has_collection, registry_collection_name
        ):
            return None

        filter_expr = f"{_ID_FIELD} == {_expr_string(name)}"
        try:
            result = await asyncio.to_thread(
                self._client.get,
                collection_name=registry_collection_name,
                ids=[name],
                output_fields=[_ID_FIELD, self._REGISTRY_CONFIG],
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
        if entry_id is not None and entry_id != name:
            return None
        config = cast(dict[str, Any], entry.get(self._REGISTRY_CONFIG))
        if config is None:
            # Older clients may not include the primary key in get() output unless queried.
            rows = await asyncio.to_thread(
                self._client.query,
                collection_name=registry_collection_name,
                filter=filter_expr,
                output_fields=[_ID_FIELD, self._REGISTRY_CONFIG],
            )
            rows = list(rows)
            if not rows:
                return None
            config = cast(dict[str, Any], rows[0][self._REGISTRY_CONFIG])
        return config

    @staticmethod
    def _parse_entry(entry: Mapping[str, Any]) -> VectorStoreCollectionConfig:
        """Parse a VectorStoreCollectionConfig from a registry entry."""
        return VectorStoreCollectionConfig(
            vector_dimensions=entry[MilvusVectorStore._REGISTRY_VECTOR_DIMENSIONS],
        )

    def _build_collection_handle(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> MilvusVectorStoreCollection:
        """Build a MilvusVectorStoreCollection handle."""
        return MilvusVectorStoreCollection(
            client=self._client,
            collection_name=MilvusVectorStore._build_native_collection_name(
                namespace, config
            ),
            partition_key=name,
            config=config,
            indexed_properties=self._indexed_properties,
            tracker=self._tracker,
        )

    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Idempotently create the native Milvus collection."""
        native_collection_name = MilvusVectorStore._build_native_collection_name(
            namespace, config
        )
        if await asyncio.to_thread(self._client.has_collection, native_collection_name):
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
                dim=config.vector_dimensions,
            )

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name=_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type=MilvusVectorStore._MILVUS_METRIC_TYPE,
            )

            self._client.create_collection(
                collection_name=native_collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self._consistency_level,
            )

        try:
            await asyncio.to_thread(_create_collection)
        except MilvusException as exc:
            if not MilvusVectorStore._is_already_exists_error(exc):
                raise

    async def _register_collection(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Write the logical collection entry to the registry."""
        registry_name = MilvusVectorStore._registry_collection_name(namespace)
        await asyncio.to_thread(
            self._client.insert,
            collection_name=registry_name,
            data=[
                {
                    _ID_FIELD: name,
                    _VECTOR_FIELD: [0.0] * _REGISTRY_VECTOR_DIMENSION,
                    self._REGISTRY_CONFIG: {
                        self._REGISTRY_VECTOR_DIMENSIONS: config.vector_dimensions,
                    },
                }
            ],
        )

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        """Create a logical collection in the Milvus vector store."""
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
            await self._register_collection(namespace, name, config)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> MilvusVectorStoreCollection:
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
                existing_config = MilvusVectorStore._parse_entry(entry)
                if existing_config != config:
                    raise VectorStoreCollectionConfigMismatchError(
                        namespace, name, existing_config, config
                    )
                return self._build_collection_handle(namespace, name, existing_config)

            await self._ensure_namespace_registry_collection(namespace)
            await self._create_native_collection(namespace, config)
            await self._register_collection(namespace, name, config)
            return self._build_collection_handle(namespace, name, config)

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> MilvusVectorStoreCollection | None:
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
            namespace, name, MilvusVectorStore._parse_entry(entry)
        )

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        """No-op; Milvus collection handles require no explicit close."""

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """Delete a logical collection from the Milvus vector store."""
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

            config = MilvusVectorStore._parse_entry(entry)
            native_collection_name = MilvusVectorStore._build_native_collection_name(
                namespace, config
            )
            registry_name = MilvusVectorStore._registry_collection_name(namespace)

            await asyncio.to_thread(
                self._client.delete,
                collection_name=native_collection_name,
                filter=f"{_PARTITION_KEY_FIELD} == {_expr_string(name)}",
            )
            await asyncio.to_thread(
                self._client.delete,
                collection_name=registry_name,
                ids=[name],
            )
