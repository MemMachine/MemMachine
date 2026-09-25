"""Milvus-based vector store implementation."""

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, ClassVar, cast, override
from uuid import UUID

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
    PROPERTY_VALUE_KEY,
    decode_properties,
    encode_properties,
)
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds

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

_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
"""The native partition-key field; holds the incarnation, never the collection's name.

A collection deleted and re-created under the same name gets a fresh
incarnation, and its predecessor's entities are invisible to it while the
purge reclaims them.
"""
_VECTOR_FIELD = "vector"
_PROPERTIES_FIELD = "properties"
"""A JSON field holding the properties the collection's schema does not declare."""
_DECLARED_FIELD_PREFIX = "_p_"
"""The prefix of the typed field holding a declared property."""
_OFFSET_FIELD_PREFIX = "_tz_"
"""The prefix of the field holding a declared datetime property's UTC offset.

A TIMESTAMPTZ field keeps the instant and returns it in UTC; the offset, in
seconds, restores the timezone the value was written in.
"""

_MAX_UUID_LENGTH = 36
_MAX_PRIMARY_ID_LENGTH = 128
_INCARNATION_HEX_LENGTH = 32
_FALSE_EXPR = f'{_ID_FIELD} == "__memmachine_no_match__"'

_DECLARED_DATA_TYPES: dict[type[PropertyValue], DataType] = {
    bool: DataType.BOOL,
    int: DataType.INT64,
    float: DataType.DOUBLE,
    str: DataType.VARCHAR,
    datetime: DataType.TIMESTAMPTZ,
}
# Term lookups on strings, a bitmap for two values, sorted order for ranges.
_DECLARED_INDEX_TYPES: dict[type[PropertyValue], str] = {
    bool: "BITMAP",
    int: "STL_SORT",
    float: "STL_SORT",
    str: "INVERTED",
    datetime: "STL_SORT",
}

# The vector index: HNSW with 4-bit codes, rescored against half-precision
# copies of the vectors. The collection isolates tenants by partition key,
# which only the HNSW family supports, so each tenant's search walks an index
# of its own group of tenants rather than every tenant's rows.
_VECTOR_INDEX_TYPE = "HNSW_SQ"
_VECTOR_INDEX_PARAMS: dict[str, Any] = {
    "M": 16,
    "efConstruction": 200,
    "sq_type": "SQ4U",
    "refine": True,
    "refine_type": "FP16",
}
# `refine_k` rescores that many candidates per result.
_SEARCH_PARAMS: dict[str, Any] = {"ef": 64, "refine_k": 2}

# Consecutive lost creation races before open-or-create gives up: every
# retry requires another process to have created and then deleted the
# collection in between, so this depth means something else is wrong.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10


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


def _declared_literal(value: PropertyValue) -> str:
    """Return a Milvus expression literal for a declared property's value.

    A datetime is a TIMESTAMPTZ instant literal, which compares instants
    whatever the offsets.
    """
    if isinstance(value, datetime):
        return f"ISO '{ensure_tz_aware(value).astimezone(UTC).isoformat()}'"
    return _literal(value)


def _fits(value: PropertyValue, declared_type: type[PropertyValue]) -> bool:
    """Whether a value can be compared with, or stored in, a declared property."""
    if isinstance(value, bool):
        return declared_type is bool
    if isinstance(value, int | float):
        return declared_type in (int, float)
    return isinstance(value, declared_type)


def _absent(key: str, declared: Mapping[str, type[PropertyValue]]) -> str:
    """A Milvus expression true exactly where the property has no value."""
    if key in declared:
        return f"{_DECLARED_FIELD_PREFIX}{key} is null"
    return f"not exists {_PROPERTIES_FIELD}[{_expr_string(key)}]"


def _condition(
    expr: FilterComparison | FilterIn,
    declared: Mapping[str, type[PropertyValue]],
) -> str:
    """A Milvus expression for one condition; false or null where the property has no value."""
    values = list(expr.values) if isinstance(expr, FilterIn) else [expr.value]
    declared_type = declared.get(expr.field)
    if declared_type is None:
        target = (
            f"{_PROPERTIES_FIELD}[{_expr_string(expr.field)}]"
            f"[{_expr_string(PROPERTY_VALUE_KEY)}]"
        )
        render = _literal
    else:
        # A value of another type never equals or orders against the property.
        values = [value for value in values if _fits(value, declared_type)]
        target = f"{_DECLARED_FIELD_PREFIX}{expr.field}"
        render = _declared_literal
    if not values:
        return _FALSE_EXPR
    if isinstance(expr, FilterIn):
        return f"{target} in [{', '.join(render(value) for value in values)}]"
    operator = "==" if expr.op == "=" else expr.op
    return f"{target} {operator} {render(values[0])}"


def _milvus_filter(
    expr: FilterExpr,
    declared: Mapping[str, type[PropertyValue]],
    *,
    negate: bool = False,
) -> str:
    """Compile a filter, or with `negate` its complement, into a Milvus expression.

    A condition on a property with no value is false, and a negation is the
    complement, true wherever the negated expression is not, missing values
    included. Milvus evaluates a condition on a null the SQL way, so negation
    is pushed down to the conditions, each of which, negated, also holds
    where its property has no value. `!=` is the negation of `=`.
    """
    if isinstance(expr, FilterNot):
        return _milvus_filter(expr.expr, declared, negate=not negate)
    if isinstance(expr, FilterAnd | FilterOr):
        left = _milvus_filter(expr.left, declared, negate=negate)
        right = _milvus_filter(expr.right, declared, negate=negate)
        operator = "&&" if isinstance(expr, FilterAnd) != negate else "||"
        return f"({left}) {operator} ({right})"
    if isinstance(expr, FilterIsNull):
        absent = _absent(expr.field, declared)
        return f"not ({absent})" if negate else absent
    if isinstance(expr, FilterComparison) and expr.op == "!=":
        equal = FilterComparison(field=expr.field, op="=", value=expr.value)
        return _milvus_filter(equal, declared, negate=not negate)
    if isinstance(expr, FilterComparison | FilterIn):
        condition = _condition(expr, declared)
        if negate:
            return f"(not ({condition})) || ({_absent(expr.field, declared)})"
        return condition
    message = f"Unsupported filter expression type: {type(expr)}"
    raise TypeError(message)


def _incarnation_filter(incarnation: UUID) -> str:
    """A Milvus expression matching the entities of one collection incarnation."""
    return f"{_PARTITION_KEY_FIELD} == {_expr_string(incarnation.hex)}"


class MilvusVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by Milvus."""

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
    def _primary_id(incarnation: UUID, record_uuid: UUID) -> str:
        """Build a native primary key unique within a shared native collection."""
        return f"{incarnation.hex}:{record_uuid}"

    def __init__(
        self,
        *,
        client: MilvusClient,
        native_collection_name: str,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
        tracker: OperationTracker,
        is_live: Callable[[UUID], Awaitable[bool]],
        request_timeout_seconds: int,
    ) -> None:
        """Initialize with a Milvus client and the incarnation the handle is bound to."""
        self._client = client
        self._native_collection_name = native_collection_name
        self._namespace = namespace
        self._name = name
        self._incarnation = incarnation
        self._config = config
        self._tracker = tracker
        self._is_live = is_live
        self._request_timeout_seconds = request_timeout_seconds

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the collection's.

        Called before every operation, to refuse a handle known to be
        dead, and after a write, so a write completed under an incarnation
        that died meanwhile raises instead of reporting success. Milvus has
        no transactions, so a write can still land under a dead
        incarnation: between the two checks, or after a check that never
        ran; the tombstone's purge rounds reclaim it. A read is not checked
        after: a collection deleted while a read is in flight keeps its
        entities until a purge round claims its tombstone, so the read returns
        what it saw, a snapshot from before the deletion, as a read that
        happened to run just before it would have.
        """
        if not await self._is_live(self._incarnation):
            raise VectorStoreCollectionHandleStaleError(self._namespace, self._name)

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        if record.vector is None:
            raise ValueError(
                f"Record {record.uuid} has vector=None, which is not allowed on input."
            )

        properties = record.properties if record.properties is not None else {}
        declared = self._config.indexed_properties_schema
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(self._incarnation, record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: self._incarnation.hex,
            _VECTOR_FIELD: record.vector,
            _PROPERTIES_FIELD: encode_properties(
                {key: value for key, value in properties.items() if key not in declared}
            ),
        }
        for key, declared_type in declared.items():
            value = properties.get(key)
            if value is not None and not _fits(value, declared_type):
                raise TypeError(
                    f"Property {key!r} is declared {declared_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            if isinstance(value, datetime):
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = ensure_tz_aware(
                    value
                ).isoformat()
                entity[f"{_OFFSET_FIELD_PREFIX}{key}"] = utc_offset_seconds(value)
            elif declared_type is datetime:
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = None
                entity[f"{_OFFSET_FIELD_PREFIX}{key}"] = None
            elif declared_type is float and isinstance(value, int | float):
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = float(value)
            else:
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = value
        return entity

    def _parse_record(
        self,
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
            for key, declared_type in self._config.indexed_properties_schema.items():
                value = entity.get(f"{_DECLARED_FIELD_PREFIX}{key}")
                if value is None:
                    continue
                if declared_type is datetime:
                    offset = timedelta(seconds=entity[f"{_OFFSET_FIELD_PREFIX}{key}"])
                    value = datetime.fromisoformat(value).astimezone(timezone(offset))
                properties[key] = value

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
            for key, declared_type in self._config.indexed_properties_schema.items():
                fields.append(f"{_DECLARED_FIELD_PREFIX}{key}")
                if declared_type is datetime:
                    fields.append(f"{_OFFSET_FIELD_PREFIX}{key}")
        return fields

    def _score(self, distance: float) -> float:
        """The store's score for a distance Milvus returned.

        Milvus returns cosine similarity and inner product as they are, and
        the squared Euclidean distance.
        """
        if self._config.similarity_metric is SimilarityMetric.EUCLIDEAN:
            return math.sqrt(max(distance, 0.0))
        return distance

    def _partition_filter(self) -> str:
        return _incarnation_filter(self._incarnation)

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        async with self._tracker("upsert"):
            records = list(records)
            if not records:
                return

            await self._fence()
            entities = [self._build_entity(record) for record in records]

            def _upsert() -> None:
                self._client.upsert(
                    collection_name=self._native_collection_name,
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
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
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
                property_expr = _milvus_filter(
                    property_filter, self._config.indexed_properties_schema
                )
                filter_expr = f"({filter_expr}) && ({property_expr})"

            raw_results = await asyncio.to_thread(
                self._client.search,
                collection_name=self._native_collection_name,
                data=query_vectors,
                filter=filter_expr,
                limit=limit,
                search_params={"params": _SEARCH_PARAMS},
                output_fields=self._output_fields(
                    return_vector=return_vector,
                    return_properties=return_properties,
                ),
                anns_field=_VECTOR_FIELD,
                timeout=self._request_timeout_seconds,
            )

            results: list[QueryResult] = []
            for raw_matches in raw_results:
                matches: list[QueryMatch] = []
                for raw_match in raw_matches:
                    entity = cast(Mapping[str, Any], raw_match["entity"])
                    score = self._score(raw_match["distance"])
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
        async with self._tracker("get"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return []

            await self._fence()
            primary_ids = [
                self._primary_id(self._incarnation, uuid) for uuid in uuid_list
            ]
            raw_records = await asyncio.to_thread(
                self._client.get,
                collection_name=self._native_collection_name,
                ids=primary_ids,
                output_fields=self._output_fields(
                    return_vector=return_vector,
                    return_properties=return_properties,
                ),
                timeout=self._request_timeout_seconds,
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
            records = [
                records_by_uuid[record_uuid]
                for record_uuid in uuid_list
                if record_uuid in records_by_uuid
            ]
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
            primary_ids = [
                self._primary_id(self._incarnation, uuid) for uuid in uuid_list
            ]
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._native_collection_name,
                ids=primary_ids,
                timeout=self._request_timeout_seconds,
            )
            await self._fence()


class MilvusVectorStoreParams(BaseModel):
    """
    Parameters for MilvusVectorStore.

    Attributes:
        client (MilvusClient): Milvus client instance.
        collection_registry (VectorStoreCollectionRegistry):
            The registry of the Milvus deployment the client reaches: which
            collections exist, under which incarnation and configuration,
            and which dead incarnations await purge. Milvus arbitrates none
            of that, so the registry lives where a primary key and a
            transaction can. Every store on the deployment, in any
            process, uses this registry, and no store on another
            deployment does. The caller starts it before handing it over.
        consistency_level (str): Collection consistency level for newly created collections.
        request_timeout_seconds (int): Seconds any request to Milvus may take.
        max_varchar_length (int):
            Bytes a declared string property can hold: the length of its
            VARCHAR field. Milvus refuses a length above its
            proxy.maxVarCharLength.
        purge_batch_size (int):
            The most entities one purge round lists and deletes. Milvus
            refuses a query whose limit exceeds its
            quotaAndLimits.limits.maxQueryResultWindow.
        metrics_factory (MetricsFactory | None): Metrics factory for collecting usage metrics.
    """

    client: InstanceOf[MilvusClient] = Field(
        ...,
        description="Milvus client instance",
    )
    collection_registry: InstanceOf[VectorStoreCollectionRegistry] = Field(
        ...,
        description="The registry of the deployment the client reaches",
    )
    consistency_level: str = Field(
        default="Session",
        description="Milvus consistency level for newly created collections",
    )
    request_timeout_seconds: int = Field(
        ..., gt=0, description="Seconds any request to Milvus may take"
    )
    max_varchar_length: int = Field(
        ..., gt=0, description="Bytes a declared string property can hold"
    )
    purge_batch_size: int = Field(
        ..., gt=0, description="The most entities one purge round lists and deletes"
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore.

    A logical collection is a partition-key value, the incarnation of its
    life, inside a native collection shared by the logical collections of
    one namespace and configuration. The catalog is the `VectorStoreCollectionRegistry`
    the store is given: it mints the incarnations and arbitrates creation,
    deletion and reclamation across processes, which Milvus, with no
    transactions or unique constraints, cannot. Any process sharing the
    Milvus backend and the registry may serve any collection.
    """

    _SIMILARITY_METRIC_TO_MILVUS_METRIC: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "COSINE",
        SimilarityMetric.DOT: "IP",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        message = str(error).lower()
        return "already exist" in message or "already exists" in message

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
        self._collection_registry = params.collection_registry
        self._request_timeout_seconds = params.request_timeout_seconds
        self._max_varchar_length = params.max_varchar_length
        self._purge_batch_size = params.purge_batch_size
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_milvus",
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
    ) -> MilvusVectorStoreCollection:
        """Build a MilvusVectorStoreCollection handle bound to the registered incarnation."""
        return MilvusVectorStoreCollection(
            client=self._client,
            native_collection_name=MilvusVectorStore._build_native_collection_name(
                namespace, registered.config
            ),
            namespace=namespace,
            name=name,
            incarnation=registered.incarnation,
            config=registered.config,
            tracker=self._tracker,
            is_live=self._collection_registry.is_live,
            request_timeout_seconds=self._request_timeout_seconds,
        )

    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Idempotently create the native Milvus collection."""
        self._validate_metric(config.similarity_metric)
        native_collection_name = MilvusVectorStore._build_native_collection_name(
            namespace, config
        )

        def _create_collection() -> None:
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
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
                dim=config.vector_dimensions,
            )
            schema.add_field(
                field_name=_PROPERTIES_FIELD,
                datatype=DataType.JSON,
            )
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name=_VECTOR_FIELD,
                index_type=_VECTOR_INDEX_TYPE,
                metric_type=self._SIMILARITY_METRIC_TO_MILVUS_METRIC[
                    config.similarity_metric
                ],
                params=_VECTOR_INDEX_PARAMS,
            )
            for key, declared_type in config.indexed_properties_schema.items():
                if declared_type is str:
                    schema.add_field(
                        field_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                        datatype=DataType.VARCHAR,
                        max_length=self._max_varchar_length,
                        nullable=True,
                    )
                else:
                    schema.add_field(
                        field_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                        datatype=_DECLARED_DATA_TYPES[declared_type],
                        nullable=True,
                    )
                if declared_type is datetime:
                    schema.add_field(
                        field_name=f"{_OFFSET_FIELD_PREFIX}{key}",
                        datatype=DataType.INT32,
                        nullable=True,
                    )
                index_params.add_index(
                    field_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                    index_type=_DECLARED_INDEX_TYPES[declared_type],
                )

            self._client.create_collection(
                collection_name=native_collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self._consistency_level,
                properties={"partitionkey.isolation": True},
                timeout=self._request_timeout_seconds,
            )

        if await asyncio.to_thread(
            self._client.has_collection,
            native_collection_name,
            timeout=self._request_timeout_seconds,
        ):
            return
        try:
            await asyncio.to_thread(_create_collection)
        except MilvusException as exc:
            if not MilvusVectorStore._is_already_exists_error(exc):
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
        self._validate_metric(config.similarity_metric)
        async with self._tracker("create_collection"):
            # The native collection first, the registry row last: a crash
            # between the two leaves an empty native collection the next
            # creation of the same configuration adopts, never a row whose
            # entities have nowhere to go. The registry's primary key is the
            # arbiter: a racing creator on any process loses here, never in
            # Milvus.
            await self._create_native_collection(namespace, config)
            await self._collection_registry.register(namespace, name, config)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> MilvusVectorStoreCollection:
        require_identifiers(namespace, name)
        self._validate_metric(config.similarity_metric)
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
    ) -> MilvusVectorStoreCollection | None:
        require_identifiers(namespace, name)
        registered = await self._collection_registry.get(namespace, name)
        if registered is None:
            return None
        return self._build_collection_handle(namespace, name, registered)

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        # Milvus collection handles hold nothing to release.
        pass

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("delete_collection"):
            # One registry transaction: the collection is unreachable when
            # it commits, and its entities wait on the queue for the purge.
            await self._collection_registry.unregister(namespace, name)

    @override
    async def purge_deleted_collections(self) -> bool:
        # One purge round per call, on the tombstone that came due first: the
        # claim is a row lock the registry holds while the round lists up to a
        # batch of the incarnation's entities and deletes them by primary key.
        # Deleting the whole incarnation by filter would be one request, but
        # Milvus applies it as one burst that every Session read waits
        # behind; a batch keeps each burst small. The registry keeps or
        # removes the tombstone by what the round found.
        async with (
            self._tracker("purge_deleted_collections"),
            self._collection_registry.claim_purgeable_incarnation() as claim,
        ):
            if claim is None:
                return False
            native_collection_name = MilvusVectorStore._build_native_collection_name(
                claim.namespace, claim.config
            )
            if await asyncio.to_thread(
                self._client.has_collection,
                native_collection_name,
                timeout=self._request_timeout_seconds,
            ):
                # Primary keys begin with the incarnation and a colon, so its
                # keys are exactly those between that prefix and the prefix
                # ending in the character after the colon: a key-range query
                # reads the sorted key index instead of scanning the
                # partition-key column of the incarnation's whole partition.
                prefix = claim.incarnation.hex
                listed = await asyncio.to_thread(
                    self._client.query,
                    collection_name=native_collection_name,
                    filter=(
                        f"{_ID_FIELD} > {_expr_string(f'{prefix}:')}"
                        f" and {_ID_FIELD} < {_expr_string(f'{prefix};')}"
                    ),
                    output_fields=[_ID_FIELD],
                    limit=self._purge_batch_size,
                    timeout=self._request_timeout_seconds,
                )
                primary_ids = [entity[_ID_FIELD] for entity in listed]
            else:
                # The native collection is gone with everything in it.
                primary_ids = []
            claim.found = bool(primary_ids)
            if claim.found:
                await asyncio.to_thread(
                    self._client.delete,
                    collection_name=native_collection_name,
                    ids=primary_ids,
                    timeout=self._request_timeout_seconds,
                )
            return claim.found
