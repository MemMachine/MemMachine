"""Milvus-based vector store implementation."""

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

import grpc
import grpc.aio
from pydantic import Field, InstanceOf
from pymilvus import AsyncMilvusClient, DataType
from pymilvus.exceptions import MilvusException

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
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
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.properties_json import (
    PROPERTY_TYPE_KEY,
    PROPERTY_VALUE_KEY,
    encode_properties,
)
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
)
from .partition_registry import Registration
from .registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreParams,
    RegistryBackedVectorStorePartition,
)

_COLLECTION_NAME_PREFIX = "sys_"
"""The prefix of the native collection's name, before the vector store name.

Milvus requires a collection name to begin with a letter or an underscore,
and a vector store name may begin with a digit. Milvus names its own
internals with a leading underscore (`_default`, `__virtual_pk__`), so the
prefix begins with a letter.
"""
_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
"""The native partition-key field, holding the incarnation of the partition an entity belongs to.

A partition created again under a deleted one's key gets a fresh
incarnation, so it holds only the entities written under that incarnation.
"""
_VECTOR_FIELD = "vector"
_PROPERTIES_FIELD = "properties"
"""A JSON field holding the properties the collection's schema does not declare."""
_DECLARED_FIELD_PREFIX = "_p_"
"""The prefix of the typed field holding a declared property."""

_UUID_LENGTH = 36
"""The length of a UUID's hyphenated text form (RFC 9562), the form stored."""
_PRIMARY_ID_LENGTH = 2 * _UUID_LENGTH + 1
"""The length of a primary key, `"{incarnation}:{record_uuid}"`."""
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


def _property_value_literal(value: PropertyValue) -> str:
    """Return a Milvus expression literal for a property value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, datetime):
        return _expression_string_literal(
            ensure_tz_aware(value).astimezone(UTC).isoformat()
        )
    return _expression_string_literal(value)


def _declared_property_value_literal(value: PropertyValue) -> str:
    """Return a Milvus expression literal for a declared property's value.

    A datetime is a TIMESTAMPTZ instant literal, which compares instants
    whatever the offsets.
    """
    if isinstance(value, datetime):
        return f"ISO '{ensure_tz_aware(value).astimezone(UTC).isoformat()}'"
    return _property_value_literal(value)


def _is_comparable_with_type(
    value: PropertyValue, property_type: type[PropertyValue]
) -> bool:
    """Whether a filter value can be compared with a property of the type.

    An int compares with an int or a float, and a float only with a float:
    Milvus refuses a float literal against an INT64 field.
    """
    if isinstance(value, bool):
        return property_type is bool
    if isinstance(value, int):
        return property_type in (int, float)
    if isinstance(value, float):
        return property_type is float
    return isinstance(value, property_type)


def _property_absent_expression(
    key: str, declared: Mapping[str, type[PropertyValue]]
) -> str:
    """A Milvus expression true exactly where the property has no value."""
    if key in declared:
        return f"{_DECLARED_FIELD_PREFIX}{key} is null"
    return f"not exists {_PROPERTIES_FIELD}[{_expression_string_literal(key)}]"


def _condition_expression(
    expr: FilterComparison | FilterIn,
    declared: Mapping[str, type[PropertyValue]],
) -> str:
    """A Milvus expression for one condition; false or null where the property has no value."""
    values = list(expr.values) if isinstance(expr, FilterIn) else [expr.value]
    declared_type = declared.get(expr.field)
    # A value of another type never equals or orders against the property.
    if declared_type is None:
        entry = f"{_PROPERTIES_FIELD}[{_expression_string_literal(expr.field)}]"
        target = f"{entry}[{_expression_string_literal(PROPERTY_VALUE_KEY)}]"
        type_names = ", ".join(
            _expression_string_literal(type_name)
            for property_type, type_name in PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME.items()
            if any(_is_comparable_with_type(value, property_type) for value in values)
        )
        type_check = (
            f"{entry}[{_expression_string_literal(PROPERTY_TYPE_KEY)}] "
            f"in [{type_names}] && "
        )
        render = _property_value_literal
    else:
        values = [
            value for value in values if _is_comparable_with_type(value, declared_type)
        ]
        target = f"{_DECLARED_FIELD_PREFIX}{expr.field}"
        type_check = ""
        render = _declared_property_value_literal
    if not values:
        return _FALSE_EXPR
    match expr:
        case FilterIn():
            literals = ", ".join(render(value) for value in values)
            return f"{type_check}{target} in [{literals}]"
        case FilterComparison(op="="):
            return f"{type_check}{target} == {render(values[0])}"
        case FilterComparison(op=op):
            return f"{type_check}{target} {op} {render(values[0])}"


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
            absent = _property_absent_expression(field, declared)
            return f"not ({absent})" if negate else absent
        case FilterComparison(field, "!=", value):
            equal = FilterComparison(field=field, op="=", value=value)
            return _filter_expression(equal, declared, negate=not negate)
        case FilterComparison() | FilterIn():
            condition = _condition_expression(expr, declared)
            if negate:
                return f"(not ({condition})) || ({_property_absent_expression(expr.field, declared)})"
            return condition
        case _:
            raise TypeError(f"Unsupported filter expression type: {type(expr)}")


def _require_every_key_accepted(result: Mapping[str, int], sent: int) -> None:
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


def _incarnation_filter(incarnation: UUID) -> str:
    """A Milvus expression matching the entities of one partition incarnation."""
    return f"{_PARTITION_KEY_FIELD} == {_expression_string_literal(str(incarnation))}"


class MilvusVectorStorePartition(RegistryBackedVectorStorePartition):
    """A partition backed by Milvus: one partition-key value inside the store's collection."""

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

    def __init__(
        self,
        *,
        client: AsyncMilvusClient,
        collection_name: str,
        vector_store_name: str,
        registration: Registration,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
        request_timeout_seconds: int,
    ) -> None:
        """Initialize with a Milvus client and the registration the handle is bound to."""
        super().__init__(
            vector_store_name=vector_store_name,
            registration=registration,
            vector_dimensions=vector_dimensions,
            similarity_metric=similarity_metric,
            indexed_properties=indexed_properties,
            tracker=tracker,
        )
        self._client = client
        self._collection_name = collection_name
        self._request_timeout_seconds = request_timeout_seconds

    def _primary_id(self, record_uuid: UUID) -> str:
        """The primary key of a record: the incarnation and the record UUID.

        Primary keys are distinct across the partitions sharing the native
        collection and across a key's incarnations.
        """
        return f"{self._incarnation}:{record_uuid}"

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        declared = self.indexed_properties
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: str(self._incarnation),
            _VECTOR_FIELD: record.vector,
            _PROPERTIES_FIELD: encode_properties(
                {
                    key: value
                    for key, value in record.properties.items()
                    if key not in declared
                }
            ),
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

    def _score_from_distance(self, distance: float) -> float:
        """The store's score for a distance Milvus returned.

        Milvus returns cosine similarity and inner product as they are, and
        the squared Euclidean distance.
        """
        if self.similarity_metric is SimilarityMetric.EUCLIDEAN:
            return math.sqrt(max(distance, 0.0))
        return distance

    @override
    async def _upsert(self, records: list[Record]) -> None:
        await self._upsert_entities([self._build_entity(record) for record in records])

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

    @override
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        filter_expr = _incarnation_filter(self._incarnation)
        if property_filter is not None:
            property_expr = _filter_expression(property_filter, self.indexed_properties)
            filter_expr = f"({filter_expr}) && ({property_expr})"

        raw_results = await self._client.search(
            collection_name=self._collection_name,
            data=query_vectors,
            filter=filter_expr,
            limit=limit,
            search_params={"params": {"refine_k": _SEARCH_REFINE_K}},
            output_fields=[_RECORD_UUID_FIELD],
            anns_field=_VECTOR_FIELD,
            timeout=self._request_timeout_seconds,
        )

        # Milvus returns each query's hits best first, and the square root
        # taken of a Euclidean distance keeps their order.
        results: list[QueryResult] = []
        for raw_matches in raw_results:
            matches: list[QueryMatch] = []
            for raw_match in raw_matches:
                entity = cast(Mapping[str, Any], raw_match["entity"])
                score = self._score_from_distance(raw_match["distance"])
                if not self._passes_threshold(
                    score, score_threshold, self.similarity_metric
                ):
                    continue

                matches.append(
                    QueryMatch(
                        score=score,
                        record_uuid=UUID(str(entity[_RECORD_UUID_FIELD])),
                    )
                )

            results.append(QueryResult(matches=matches))

        return results

    @override
    async def _delete(self, record_uuids: list[UUID]) -> None:
        primary_ids = [self._primary_id(uuid) for uuid in record_uuids]
        result = await self._client.delete(
            collection_name=self._collection_name,
            ids=primary_ids,
            timeout=self._request_timeout_seconds,
        )
        _require_every_key_accepted(result, len(primary_ids))


class MilvusVectorStoreParams(RegistryBackedVectorStoreParams):
    """
    Parameters for MilvusVectorStore.

    The native Milvus collection is named `sys_` followed by
    `vector_store_name`. Milvus scores by cosine, dot or euclidean only. Each
    declared property is a nullable typed field, `_p_<key>`, with a scalar
    index.

    Attributes:
        client (AsyncMilvusClient):
            Async Milvus client instance.
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
    """

    client: InstanceOf[AsyncMilvusClient] = Field(
        ...,
        description="Async Milvus client instance",
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


class MilvusVectorStore(RegistryBackedVectorStore[MilvusVectorStorePartition]):
    """Asynchronous Milvus-based implementation of VectorStore.

    The store is one native Milvus collection, named at construction, in
    which a partition is the entities carrying its incarnation in the
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
        MilvusVectorStore._validate_metric(params.similarity_metric)
        super().__init__(params, metrics_prefix="vector_store_milvus")
        self._client = params.client
        self._collection_name = f"{_COLLECTION_NAME_PREFIX}{params.vector_store_name}"
        self._request_timeout_seconds = params.request_timeout_seconds
        self._max_varchar_length = params.max_varchar_length
        self._purge_batch_size = params.purge_batch_size

    @override
    async def _prepare_storage(self) -> None:
        # Each step runs when it is missing, and the collection is loaded, so
        # the next startup completes one that failed part way.
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_name=_VECTOR_FIELD,
            index_type=_VECTOR_INDEX_TYPE,
            metric_type=self._SIMILARITY_METRIC_TO_MILVUS_METRIC[
                self.similarity_metric
            ],
            params=_VECTOR_INDEX_PARAMS,
        )
        for key in self.indexed_properties:
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
                dim=self.vector_dimensions,
            )
            schema.add_field(
                field_name=_PROPERTIES_FIELD,
                datatype=DataType.JSON,
            )
            for key, declared_type in self.indexed_properties.items():
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
    async def _prepare_partition_storage(
        self, partition_key: str, incarnation: UUID
    ) -> None:
        # A partition is the entities carrying its incarnation in the
        # store's one native collection, which startup prepared; it has no
        # storage of its own.
        pass

    @override
    def _partition_handle(
        self, registration: Registration
    ) -> MilvusVectorStorePartition:
        return MilvusVectorStorePartition(
            client=self._client,
            collection_name=self._collection_name,
            vector_store_name=self.vector_store_name,
            registration=registration,
            vector_dimensions=self.vector_dimensions,
            similarity_metric=self.similarity_metric,
            indexed_properties=self.indexed_properties,
            tracker=self._tracker,
            request_timeout_seconds=self._request_timeout_seconds,
        )

    @override
    async def _purge_round(self, incarnation: UUID) -> bool:
        # Deletes the incarnation's entities by primary key, one listed batch
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
            filter=_incarnation_filter(incarnation),
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
            _require_every_key_accepted(result, len(primary_ids))
        return bool(primary_ids)
