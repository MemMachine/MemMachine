"""Milvus-based vector store implementation."""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

import grpc
import grpc.aio
from pydantic import Field, InstanceOf
from pymilvus import AsyncMilvusClient, DataType
from pymilvus.exceptions import MilvusException

from memmachine_server.common.data_types import (
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
    VectorStoreSchema,
)
from .utils import (
    validate_declared_properties,
    validate_dimensions,
    validate_limit,
    validate_query_vector,
    validate_score_threshold,
    validate_supported_filter,
)
from .vector_store import VectorStore, VectorStoreParams, VectorStorePartition

_COLLECTION_NAME_PREFIX = "sys_"
"""The prefix of the native collection's name, before the vector store name.

Milvus requires a collection name to begin with a letter or an underscore,
and a vector store name may begin with a digit. Milvus names its own
internals with a leading underscore, so the prefix begins with a letter.
"""
_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
"""The native partition-key field, holding the partition UUID of the partition an entity belongs to.

A partition created again under a deleted one's name gets a fresh
partition UUID, so it holds only the entities written under that partition UUID.
"""
_VECTOR_FIELD = "vector"
_DECLARED_FIELD_PREFIX = "_p_"
"""The prefix of the typed field holding a declared property."""

_UUID_LENGTH = 36
"""The length of a UUID's hyphenated text form (RFC 9562), the form stored."""
_PRIMARY_ID_LENGTH = 2 * _UUID_LENGTH + 1
"""The length of a primary key, `"{partition_uuid}:{record_uuid}"`."""
_FALSE_EXPR = f"{_ID_FIELD} in []"
"""A Milvus expression no entity satisfies; Milvus refuses a bare `false`."""

_DECLARED_DATA_TYPES: dict[type[PropertyValue], DataType] = {
    bool: DataType.BOOL,
    int: DataType.INT64,
    float: DataType.DOUBLE,
    str: DataType.VARCHAR,
    datetime: DataType.TIMESTAMPTZ,
}

# HNSW_SQ with explicit parameters, so every server builds the same index
# whatever its version or AUTOINDEX configuration; partition-key isolation
# needs the HNSW family. refine gives refine_k FP16 vectors to rescore
# against.
_VECTOR_INDEX_TYPE = "HNSW_SQ"
_VECTOR_INDEX_PARAMS: dict[str, Any] = {
    "M": 18,
    "efConstruction": 240,
    "sq_type": "SQ4U",
    "refine": True,
    "refine_type": "FP16",
}
# Candidates per result rescored against the half-precision vectors.
_SEARCH_REFINE_K = 8


def _expression_string_literal(value: str) -> str:
    """Return a Milvus expression string literal.

    Characters stay UTF-8: Milvus's parser refuses the surrogate pair an
    ASCII escape writes for a character outside the Basic Multilingual Plane.
    """
    return json.dumps(value, ensure_ascii=False)


def _declared_property_value_literal(value: PropertyValue) -> str:
    """Return a Milvus expression literal for a declared property's value.

    A datetime is a TIMESTAMPTZ instant literal, which compares instants
    whatever the offsets.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, datetime):
        return f"ISO '{ensure_tz_aware(value).astimezone(UTC).isoformat()}'"
    return _expression_string_literal(value)


def _property_absent_expression(key: str) -> str:
    """A Milvus expression true exactly where the property has no value."""
    return f"{_DECLARED_FIELD_PREFIX}{key} is null"


def _condition_expression(
    expr: FilterComparison | FilterIn,
    declared: Mapping[str, type[PropertyValue]],
) -> str:
    """A Milvus expression for one condition; false or null where the property has no value."""
    values = list(expr.values) if isinstance(expr, FilterIn) else [expr.value]
    # A value of another type never equals or orders against the property.
    values = [value for value in values if type(value) is declared[expr.field]]
    if not values:
        return _FALSE_EXPR
    target = f"{_DECLARED_FIELD_PREFIX}{expr.field}"
    match expr:
        case FilterIn():
            return f"{target} in [{', '.join(_declared_property_value_literal(value) for value in values)}]"
        case FilterComparison(op="="):
            return f"{target} == {_declared_property_value_literal(values[0])}"
        case FilterComparison(op=op):
            return f"{target} {op} {_declared_property_value_literal(values[0])}"


def _filter_expression(
    expr: FilterExpr,
    declared: Mapping[str, type[PropertyValue]],
    *,
    negate: bool = False,
) -> str:
    """Compile a filter, or with `negate` its complement, into a Milvus expression.

    A negated comparison or membership test holds where the property has no
    value. Milvus evaluates a condition on a null the SQL way, so negation is
    pushed down to the conditions.
    """
    match expr:
        case FilterNot(operand):
            return _filter_expression(operand, declared, negate=not negate)
        case FilterAnd(left, right) | FilterOr(left, right):
            operator = "&&" if isinstance(expr, FilterAnd) != negate else "||"
            return f" {operator} ".join(
                f"({_filter_expression(operand, declared, negate=negate)})"
                for operand in (left, right)
            )
        case FilterIsNull(field):
            absent = _property_absent_expression(field)
            return f"not ({absent})" if negate else absent
        case FilterComparison(field, "!=", value):
            equal = FilterComparison(field=field, op="=", value=value)
            return _filter_expression(equal, declared, negate=not negate)
        case FilterComparison() | FilterIn():
            condition = _condition_expression(expr, declared)
            if negate:
                return f"(not ({condition})) || ({_property_absent_expression(expr.field)})"
            return condition
        case _:
            raise TypeError(f"Unsupported filter expression type: {type(expr)}")


def _validate_every_key_accepted(result: Mapping[str, int], sent: int) -> None:
    """Raise unless Milvus accepted the delete of every primary key sent.

    Milvus counts the primary keys a delete accepts, present or not, so a
    delete it rejected counts fewer, and pymilvus's async client returns
    that count.
    """
    accepted = result["delete_count"]
    if accepted != sent:
        raise MilvusException(
            message=f"Milvus accepted the delete of {accepted} of {sent} primary keys"
        )


def _partition_filter(partition_uuid: UUID) -> str:
    """A Milvus expression matching the entities of one partition."""
    return (
        f"{_PARTITION_KEY_FIELD} == {_expression_string_literal(str(partition_uuid))}"
    )


class MilvusVectorStorePartition(VectorStorePartition):
    """A partition's entities in the store's Milvus collection."""

    def __init__(
        self,
        *,
        client: AsyncMilvusClient,
        collection_name: str,
        partition_uuid: UUID,
        store_schema: VectorStoreSchema,
        tracker: OperationTracker,
        request_timeout_seconds: int,
    ) -> None:
        """Initialize with a Milvus client and the partition the handle is bound to."""
        self._client = client
        self._collection_name = collection_name
        self._partition_uuid = partition_uuid
        self._store_schema = store_schema
        self._tracker = tracker
        self._request_timeout_seconds = request_timeout_seconds

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        async with self._tracker("upsert"):
            records = list(records)
            for record in records:
                validate_declared_properties(
                    record.properties, self._store_schema.indexed_properties
                )
                validate_dimensions(record.vector, self._store_schema.vector_dimensions)
            if not records:
                return
            await self._upsert_entities(
                [self._build_entity(record) for record in records]
            )

    async def _upsert_entities(self, entities: list[dict[str, Any]]) -> None:
        """Upsert entities, halving a batch refused as too large.

        Milvus refuses a request over its proxy's gRPC receive limit with
        RESOURCE_EXHAUSTED, before writing any of it. A batch refused so is
        halved until the halves fit or a single entity is refused. Any other
        error raises at once.
        """
        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                data=entities,
                timeout=self._request_timeout_seconds,
            )
        except grpc.aio.AioRpcError as err:
            if err.code() != grpc.StatusCode.RESOURCE_EXHAUSTED or len(entities) <= 1:
                raise
            mid = len(entities) // 2
            await self._upsert_entities(entities[:mid])
            await self._upsert_entities(entities[mid:])

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        declared = self._store_schema.indexed_properties
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: str(self._partition_uuid),
            _VECTOR_FIELD: record.vector,
        }
        for key in declared:
            value = record.properties.get(key)
            if isinstance(value, datetime):
                # The instant in UTC, which is all a filter compares: Milvus
                # refuses an offset with a seconds component.
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = (
                    ensure_tz_aware(value).astimezone(UTC).isoformat()
                )
            else:
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = value
        return entity

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            for query_vector in query_vectors:
                validate_query_vector(
                    query_vector, self._store_schema.vector_dimensions
                )
            validate_score_threshold(score_threshold)
            validate_limit(limit)
            if property_filter is not None:
                validate_supported_filter(
                    property_filter, self._store_schema.indexed_properties
                )
            if not query_vectors:
                return []

            raw_results = await self._client.search(
                collection_name=self._collection_name,
                data=query_vectors,
                filter=self._query_filter(property_filter),
                limit=limit,
                search_params={"params": {"refine_k": _SEARCH_REFINE_K}},
                output_fields=[_RECORD_UUID_FIELD],
                anns_field=_VECTOR_FIELD,
                timeout=self._request_timeout_seconds,
            )
            return [
                QueryResult(matches=self._matches(raw_matches, score_threshold))
                for raw_matches in raw_results
            ]

    def _query_filter(self, property_filter: FilterExpr | None) -> str:
        """The filter expression a query sends: the partition's entities, narrowed by the property filter when one is given."""
        filter_expr = _partition_filter(self._partition_uuid)
        if property_filter is None:
            return filter_expr
        property_expr = _filter_expression(
            property_filter, self._store_schema.indexed_properties
        )
        return f"({filter_expr}) && ({property_expr})"

    def _matches(
        self, raw_matches: list[dict], score_threshold: float | None
    ) -> list[QueryMatch]:
        """
        The matches of one query's hits, best first, keeping only those whose score meets the caller's threshold.

        Milvus returns a query's hits best first, and the square root taken
        of a Euclidean distance keeps their order.
        """
        matches: list[QueryMatch] = []
        for raw_match in raw_matches:
            entity = cast(Mapping[str, Any], raw_match["entity"])
            score = self._score_from_distance(raw_match["distance"])
            if not self._passes_threshold(
                score, score_threshold, self._store_schema.similarity_metric
            ):
                continue
            matches.append(
                QueryMatch(
                    score=score,
                    record_uuid=UUID(str(entity[_RECORD_UUID_FIELD])),
                )
            )
        return matches

    def _score_from_distance(self, distance: float) -> float:
        """The store's score for a distance Milvus returned.

        Milvus returns cosine similarity and inner product as they are, and
        the squared Euclidean distance.
        """
        if self._store_schema.similarity_metric is SimilarityMetric.EUCLIDEAN:
            return math.sqrt(max(distance, 0.0))
        return distance

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

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        async with self._tracker("delete"):
            primary_ids = [self._primary_id(uuid) for uuid in record_uuids]
            if not primary_ids:
                return
            result = await self._client.delete(
                collection_name=self._collection_name,
                ids=primary_ids,
                timeout=self._request_timeout_seconds,
            )
            _validate_every_key_accepted(result, len(primary_ids))

    def _primary_id(self, record_uuid: UUID) -> str:
        """The primary key of a record: the partition UUID and the record UUID.

        Primary keys are distinct across the partitions sharing the native
        collection and across the partitions created under one name.
        """
        return f"{self._partition_uuid}:{record_uuid}"


class MilvusVectorStoreParams(VectorStoreParams):
    """
    Parameters for MilvusVectorStore.

    The schema's similarity metric must be cosine, dot, or euclidean.

    Attributes:
        client (AsyncMilvusClient):
            Async Milvus client instance, which stores of different names
            may share.
        request_timeout_seconds (int):
            Seconds any request to Milvus may take (default: 30).
        max_varchar_length (int):
            Bytes a declared string property can hold: the length of its
            VARCHAR field, at most the server's proxy.maxVarCharLength
            (default: 65535).
        purge_batch_size (int):
            The most entities one purge round lists and deletes, at most the
            server's quotaAndLimits.limits.maxQueryResultWindow (default:
            10000).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    client: InstanceOf[AsyncMilvusClient] = Field(
        ...,
        description="Async Milvus client instance, which stores of different names may share",
    )
    request_timeout_seconds: int = Field(
        30, gt=0, description="Seconds any request to Milvus may take"
    )
    max_varchar_length: int = Field(
        65535,
        gt=0,
        description=(
            "Bytes a declared string property can hold: the length of its "
            "VARCHAR field, at most the server's proxy.maxVarCharLength"
        ),
    )
    purge_batch_size: int = Field(
        10000,
        gt=0,
        description=(
            "The most entities one purge round lists and deletes, at most the "
            "server's quotaAndLimits.limits.maxQueryResultWindow"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore.

    The store is one native Milvus collection, named at construction, in
    which a partition is the entities carrying it in the
    partition-key field.

    Reads run at Bounded, the consistency level the store creates its
    native collection at: a query reflects every write that returned at
    least the server's `common.gracefulTime` before it began.
    """

    _SIMILARITY_METRIC_TO_MILVUS_METRIC: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "COSINE",
        SimilarityMetric.DOT: "IP",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        MilvusVectorStore._validate_metric(params.store_schema.similarity_metric)
        self._store_schema = params.store_schema
        self._tracker = OperationTracker(
            params.metrics_factory, prefix="vector_store_milvus"
        )
        self._client = params.client
        self._collection_name = f"{_COLLECTION_NAME_PREFIX}{params.vector_store_name}"
        self._request_timeout_seconds = params.request_timeout_seconds
        self._max_varchar_length = params.max_varchar_length
        self._purge_batch_size = params.purge_batch_size

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

    @property
    @override
    def store_schema(self) -> VectorStoreSchema:
        return self._store_schema

    @override
    async def startup(self) -> None:
        async with self._tracker("startup"):
            await self._prepare_storage()

    async def _prepare_storage(self) -> None:
        """Prepare the collection every partition lives in, its indexes, and its load, each step idempotent."""
        # Each step runs when it is missing, and the collection is loaded, so
        # the next startup completes one that failed part way.
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_name=_VECTOR_FIELD,
            index_type=_VECTOR_INDEX_TYPE,
            metric_type=self._SIMILARITY_METRIC_TO_MILVUS_METRIC[
                self._store_schema.similarity_metric
            ],
            params=_VECTOR_INDEX_PARAMS,
        )
        for key in self._store_schema.indexed_properties:
            index_params.add_index(
                field_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                index_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                index_type="AUTOINDEX",
            )

        async def _create_collection() -> None:
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name=_ID_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=_PRIMARY_ID_LENGTH,
            )
            schema.add_field(
                field_name=_RECORD_UUID_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_UUID_LENGTH,
            )
            schema.add_field(
                field_name=_PARTITION_KEY_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_UUID_LENGTH,
                is_partition_key=True,
            )
            schema.add_field(
                field_name=_VECTOR_FIELD,
                datatype=DataType.FLOAT_VECTOR,
                dim=self._store_schema.vector_dimensions,
            )
            for key, declared_type in self._store_schema.indexed_properties.items():
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

            # Without index_params, so the indexes and the load below are
            # steps of their own.
            await self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
                properties={"partitionkey.isolation": True},
                consistency_level="Bounded",
                timeout=self._request_timeout_seconds,
            )

        # Milvus answers a create of an existing collection with the same
        # schema with success, so racing creators both go on.
        if not await self._client.has_collection(
            self._collection_name,
            timeout=self._request_timeout_seconds,
        ):
            await _create_collection()

        # Index names are their field names, so the ones present say which
        # fields are indexed. An index a racing creator made is created again
        # without error.
        existing = set(
            await self._client.list_indexes(
                self._collection_name,
                timeout=self._request_timeout_seconds,
            )
        )
        missing = self._client.prepare_index_params()
        missing.extend(
            index for index in index_params if index.index_name not in existing
        )
        if missing:
            await self._client.create_index(
                self._collection_name,
                missing,
                timeout=self._request_timeout_seconds,
            )
        # A no-op when the collection is already loaded.
        await self._client.load_collection(
            self._collection_name,
            timeout=self._request_timeout_seconds,
        )

    @override
    async def shutdown(self) -> None:
        # The caller owns the client's lifecycle.
        pass

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        # Every partition lives in the collection startup
        # prepared.
        return

    @override
    async def get_partition(self, partition_uuid: UUID) -> MilvusVectorStorePartition:
        return MilvusVectorStorePartition(
            client=self._client,
            collection_name=self._collection_name,
            partition_uuid=partition_uuid,
            store_schema=self._store_schema,
            tracker=self._tracker,
            request_timeout_seconds=self._request_timeout_seconds,
        )

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        async with self._tracker("purge_partition"):
            return await self._purge_partition(partition_uuid)

    async def _purge_partition(self, partition_uuid: UUID) -> bool:
        # Deletes the partition's entities by primary key, one listed batch
        # per round, keeping each delete short for every tenant of the native
        # collection.
        if not await self._client.has_collection(
            self._collection_name,
            timeout=self._request_timeout_seconds,
        ):
            # The native collection is gone with everything in it.
            return False
        listed = await self._client.query(
            collection_name=self._collection_name,
            filter=_partition_filter(partition_uuid),
            output_fields=[_ID_FIELD],
            limit=self._purge_batch_size,
            timeout=self._request_timeout_seconds,
        )
        primary_ids = [entity[_ID_FIELD] for entity in listed]
        if primary_ids:
            result = await self._client.delete(
                collection_name=self._collection_name,
                ids=primary_ids,
                timeout=self._request_timeout_seconds,
            )
            _validate_every_key_accepted(result, len(primary_ids))
        return bool(primary_ids)

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        # Every partition lives in the collection startup prepared.
        return
