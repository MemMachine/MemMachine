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

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
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
    decode_properties,
    encode_properties,
)
from memmachine_server.common.utils import compute_similarity, ensure_tz_aware

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


class MilvusVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by Milvus."""

    _RANGE_OPERATORS: ClassVar[set[str]] = {">", ">=", "<", "<="}

    @staticmethod
    def _build_milvus_filter(expr: FilterExpr) -> str:
        """Convert a FilterExpr tree into a Milvus filter expression."""
        if isinstance(expr, FilterComparison):
            return MilvusVectorStoreCollection._build_milvus_comparison(expr)
        if isinstance(expr, FilterIn):
            if not expr.values:
                return _FALSE_EXPR
            values = ", ".join(_literal(value) for value in expr.values)
            return f"{_property_field(expr.field)} in [{values}]"
        if isinstance(expr, FilterIsNull):
            return f"{_property_field(expr.field)} is null"
        if isinstance(expr, FilterNot):
            return (
                f"not ({MilvusVectorStoreCollection._build_milvus_filter(expr.expr)})"
            )
        if isinstance(expr, FilterAnd):
            left = MilvusVectorStoreCollection._build_milvus_filter(expr.left)
            right = MilvusVectorStoreCollection._build_milvus_filter(expr.right)
            return f"({left}) && ({right})"
        if isinstance(expr, FilterOr):
            left = MilvusVectorStoreCollection._build_milvus_filter(expr.left)
            right = MilvusVectorStoreCollection._build_milvus_filter(expr.right)
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
    def _passes_threshold(
        score: float,
        threshold: float | None,
        similarity_metric: SimilarityMetric,
    ) -> bool:
        if threshold is None:
            return True
        if similarity_metric.higher_is_better:
            return score >= threshold
        return score <= threshold

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
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Milvus client and collection name."""
        self._client = client
        self._collection_name = collection_name
        self._partition_key = partition_key
        self._config = config
        self._tracker = tracker

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        return self._config

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        if record.vector is None:
            raise ValueError(
                f"Record {record.uuid} has vector=None, which is not allowed on input."
            )

        properties = record.properties if record.properties is not None else {}
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(self._partition_key, record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: self._partition_key,
            _VECTOR_FIELD: record.vector,
            _PROPERTIES_FIELD: encode_properties(properties),
        }
        # Explicit nulls clear stale dynamic fields during native Milvus upserts.
        for key in self._config.indexed_properties_schema:
            entity[_property_field(key)] = None
        for key, value in properties.items():
            entity[_property_field(key)] = _normalize_property_filter_value(value)
        return entity

    @staticmethod
    def _parse_record(
        entity: Mapping[str, Any],
        *,
        return_vector: bool,
        return_properties: bool,
    ) -> Record:
        """Parse a Milvus entity into a vector store record."""
        vector: list[float] | None = None
        if return_vector:
            raw_vector = entity.get(_VECTOR_FIELD)
            if raw_vector is not None:
                vector = list(cast(Sequence[float], raw_vector))

        properties: dict[str, PropertyValue] | None = None
        if return_properties:
            properties = decode_properties(
                cast(Mapping | None, entity.get(_PROPERTIES_FIELD))
            )

        return Record(
            uuid=UUID(str(entity[_RECORD_UUID_FIELD])),
            vector=vector,
            properties=properties,
        )

    def _output_fields(
        self, *, return_vector: bool, return_properties: bool
    ) -> list[str]:
        fields = [_RECORD_UUID_FIELD]
        if return_vector:
            fields.append(_VECTOR_FIELD)
        if return_properties:
            fields.append(_PROPERTIES_FIELD)
        return fields

    @staticmethod
    def _score_from_entity_vector(
        query_vector: Sequence[float],
        entity: Mapping[str, Any],
        similarity_metric: SimilarityMetric,
    ) -> float:
        raw_vector = entity.get(_VECTOR_FIELD)
        if raw_vector is None:
            raise ValueError("Milvus search result did not include the vector field")
        return compute_similarity(
            list(query_vector),
            [list(cast(Sequence[float], raw_vector))],
            similarity_metric,
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
                output_fields=self._output_fields(
                    return_vector=True,
                    return_properties=return_properties,
                ),
                anns_field=_VECTOR_FIELD,
            )

            results: list[QueryResult] = []
            for query_vector, raw_matches in zip(
                query_vectors, raw_results, strict=True
            ):
                matches: list[QueryMatch] = []
                for raw_match in raw_matches:
                    entity = cast(Mapping[str, Any], raw_match["entity"])
                    score = self._score_from_entity_vector(
                        query_vector,
                        entity,
                        self._config.similarity_metric,
                    )
                    if not self._passes_threshold(
                        score, score_threshold, self._config.similarity_metric
                    ):
                        continue

                    matches.append(
                        QueryMatch(
                            score=score,
                            record=self._parse_record(
                                entity,
                                return_vector=return_vector,
                                return_properties=return_properties,
                            ),
                        )
                    )

                matches.sort(
                    key=lambda match: match.score,
                    reverse=self._config.similarity_metric.higher_is_better,
                )
                results.append(QueryResult(matches=matches))

            return results

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

            primary_ids = [
                self._primary_id(self._partition_key, uuid) for uuid in uuid_list
            ]
            raw_records = await asyncio.to_thread(
                self._client.get,
                collection_name=self._collection_name,
                ids=primary_ids,
                output_fields=self._output_fields(
                    return_vector=return_vector,
                    return_properties=return_properties,
                ),
            )

            records_by_uuid = {
                record.uuid: record
                for record in (
                    self._parse_record(
                        cast(Mapping[str, Any], raw_record),
                        return_vector=return_vector,
                        return_properties=return_properties,
                    )
                    for raw_record in raw_records
                )
            }
            return [
                records_by_uuid[record_uuid]
                for record_uuid in uuid_list
                if record_uuid in records_by_uuid
            ]

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
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore."""

    _SIMILARITY_METRIC_TO_MILVUS_METRIC: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "COSINE",
        SimilarityMetric.DOT: "IP",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_VECTOR_DIMENSIONS: ClassVar[str] = "vector_dimensions"
    _REGISTRY_SIMILARITY_METRIC: ClassVar[str] = "similarity_metric"
    _REGISTRY_INDEXED_PROPERTIES_SCHEMA: ClassVar[str] = "indexed_properties_schema"
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

    @staticmethod
    def _validate_metric(similarity_metric: SimilarityMetric) -> None:
        if (
            similarity_metric
            not in MilvusVectorStore._SIMILARITY_METRIC_TO_MILVUS_METRIC
        ):
            supported = ", ".join(
                metric.value
                for metric in MilvusVectorStore._SIMILARITY_METRIC_TO_MILVUS_METRIC
            )
            raise ValueError(
                f"Milvus only supports {supported} similarity metrics, "
                f"got {similarity_metric.value!r}"
            )

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client = params.client
        self._consistency_level = params.consistency_level
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_milvus",
        )
        self._client_name_locks = MilvusVectorStore._name_locks.setdefault(
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
                metric_type="COSINE",
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
            similarity_metric=entry[MilvusVectorStore._REGISTRY_SIMILARITY_METRIC],
            indexed_properties_schema=entry[
                MilvusVectorStore._REGISTRY_INDEXED_PROPERTIES_SCHEMA
            ],
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
            tracker=self._tracker,
        )

    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Idempotently create the native Milvus collection."""
        self._validate_metric(config.similarity_metric)
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
            schema.add_field(
                field_name=_PROPERTIES_FIELD,
                datatype=DataType.JSON,
            )

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name=_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type=self._SIMILARITY_METRIC_TO_MILVUS_METRIC[
                    config.similarity_metric
                ],
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
                        self._REGISTRY_SIMILARITY_METRIC: config.similarity_metric.value,
                        self._REGISTRY_INDEXED_PROPERTIES_SCHEMA: config.model_dump(
                            mode="json"
                        )["indexed_properties_schema"],
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
        self._validate_metric(config.similarity_metric)
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
        self._validate_metric(config.similarity_metric)
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
