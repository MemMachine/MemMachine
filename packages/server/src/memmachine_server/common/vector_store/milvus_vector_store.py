"""Milvus-based vector store implementation."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

from pydantic import Field, InstanceOf
from pymilvus import AsyncMilvusClient, DataType
from pymilvus.exceptions import MilvusException

from memmachine_server.common.data_types import (
    PropertyType,
    PropertyValue,
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
    PROPERTY_VALUE_KEY,
    encode_properties,
)
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
)
from .partition_registry import LiveRegistration
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
_OFFSET_FIELD_PREFIX = "_tz_"
"""The prefix of the field holding a declared datetime's UTC offset in seconds, which TIMESTAMPTZ drops."""

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
    """Whether a filter value can be compared with a declared property."""
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
    match expr:
        case FilterIn():
            return f"{target} in [{', '.join(render(value) for value in values)}]"
        case FilterComparison(op="="):
            return f"{target} == {render(values[0])}"
        case FilterComparison(op=op):
            return f"{target} {op} {render(values[0])}"


def _milvus_filter(
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
            return _milvus_filter(operand, declared, negate=not negate)
        case FilterAnd(left, right) | FilterOr(left, right):
            operator = "&&" if isinstance(expr, FilterAnd) != negate else "||"
            return f" {operator} ".join(
                f"({_milvus_filter(operand, declared, negate=negate)})"
                for operand in (left, right)
            )
        case FilterIsNull(field):
            absent = _absent(field, declared)
            return f"not ({absent})" if negate else absent
        case FilterComparison(field, "!=", value):
            equal = FilterComparison(field=field, op="=", value=value)
            return _milvus_filter(equal, declared, negate=not negate)
        case FilterComparison() | FilterIn():
            condition = _condition(expr, declared)
            if negate:
                return f"(not ({condition})) || ({_absent(expr.field, declared)})"
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
    return f"{_PARTITION_KEY_FIELD} == {_expr_string(str(incarnation))}"


class MilvusVectorStorePartition(RegistryBackedVectorStorePartition):
    """A partition backed by Milvus: one partition-key value inside the store's collection."""

    def __init__(
        self,
        *,
        client: AsyncMilvusClient,
        collection_name: str,
        vector_store_name: str,
        registration: LiveRegistration,
        vector_dimensions: int,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
        request_timeout_seconds: int,
    ) -> None:
        """Initialize with a Milvus client and the live registration the handle is bound to."""
        super().__init__(
            vector_store_name=vector_store_name,
            registration=registration,
            vector_dimensions=vector_dimensions,
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
        for key, declared_type in declared.items():
            value = record.properties.get(key)
            if isinstance(value, datetime):
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = ensure_tz_aware(
                    value
                ).isoformat()
                entity[f"{_OFFSET_FIELD_PREFIX}{key}"] = utc_offset_seconds(value)
            elif declared_type is datetime:
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = None
                entity[f"{_OFFSET_FIELD_PREFIX}{key}"] = None
            else:
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = value
        return entity

    @override
    async def _upsert(self, records: list[Record]) -> None:
        await self._client.upsert(
            collection_name=self._collection_name,
            data=[self._build_entity(record) for record in records],
            timeout=self._request_timeout_seconds,
        )

    @override
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        min_cosine_similarity: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        filter_expr = _incarnation_filter(self._incarnation)
        if property_filter is not None:
            property_expr = _milvus_filter(property_filter, self.indexed_properties)
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

        results: list[QueryResult] = []
        for raw_matches in raw_results:
            matches: list[QueryMatch] = []
            for raw_match in raw_matches:
                entity = cast(Mapping[str, Any], raw_match["entity"])
                # Milvus returns the cosine similarity as a COSINE index's distance.
                cosine_similarity = raw_match["distance"]
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
    `vector_store_name`. Each declared property is a nullable typed field,
    `_p_<key>`, with a scalar index.

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

    Reads run at Milvus's default consistency level, Bounded: a query
    reflects every write that returned at least the server's
    `common.gracefulTime` before it began.
    """

    _MILVUS_METRIC_TYPE: ClassVar[str] = "COSINE"

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        message = str(error).lower()
        return "already exist" in message or "already exists" in message

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
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
            metric_type=MilvusVectorStore._MILVUS_METRIC_TYPE,
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
                if declared_type is datetime:
                    schema.add_field(
                        field_name=f"{_OFFSET_FIELD_PREFIX}{key}",
                        datatype=DataType.INT32,
                        nullable=True,
                    )

            # Without index_params, so the indexes and the load below are
            # steps of their own.
            await self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
                properties={"partitionkey.isolation": True},
                timeout=self._request_timeout_seconds,
            )

        if not await self._client.has_collection(
            self._collection_name,
            timeout=self._request_timeout_seconds,
        ):
            try:
                await _create_collection()
            except MilvusException as exc:
                if not MilvusVectorStore._is_already_exists_error(exc):
                    raise

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
        self, registration: LiveRegistration
    ) -> MilvusVectorStorePartition:
        return MilvusVectorStorePartition(
            client=self._client,
            collection_name=self._collection_name,
            vector_store_name=self.vector_store_name,
            registration=registration,
            vector_dimensions=self.vector_dimensions,
            indexed_properties=self.indexed_properties,
            tracker=self._tracker,
            request_timeout_seconds=self._request_timeout_seconds,
        )

    @override
    async def _purge_round(self, incarnation: UUID) -> bool:
        # Deletes the incarnation's entities by primary key, one listed batch
        # per round, keeping each delete short for every tenant of the native
        # collection.
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
