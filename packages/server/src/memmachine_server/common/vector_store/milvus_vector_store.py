"""Milvus-based vector store implementation."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

import grpc
import grpc.aio
from pydantic import BaseModel, ConfigDict, Field, InstanceOf, field_validator
from pymilvus import AsyncMilvusClient, DataType
from pymilvus.exceptions import MilvusException

from memmachine_server.common.data_types import PropertyType, PropertyValue
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
)
from memmachine_server.common.metrics_factory import OperationTracker
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


def _condition_expression(expr: Equals | Ordering | In) -> str:
    """A Milvus expression for one condition; false or null where the property has no value."""
    values = list(expr.values) if isinstance(expr, In) else [expr.value]
    if not values:
        return _FALSE_EXPR
    target = f"{_DECLARED_FIELD_PREFIX}{expr.field}"
    match expr:
        case In():
            return f"{target} in [{', '.join(_declared_property_value_literal(value) for value in values)}]"
        case Equals():
            return f"{target} == {_declared_property_value_literal(values[0])}"
        case Ordering(op=op):
            return f"{target} {op} {_declared_property_value_literal(values[0])}"


def _filter_expression(expr: FilterExpr, *, negate: bool = False) -> str:
    """Compile a filter, or with `negate` its complement, into a Milvus expression.

    A condition on a property with no value is false, and a negation is the
    complement, true wherever the negated expression is not, missing values
    included. Milvus evaluates a condition on a null the SQL way, so negation
    is pushed down to the conditions, each of which, negated, also holds
    where its property has no value.
    """
    match expr:
        case Not(operand):
            return _filter_expression(operand, negate=not negate)
        case And(operands) | Or(operands):
            operator = "&&" if isinstance(expr, And) != negate else "||"
            return f" {operator} ".join(
                f"({_filter_expression(operand, negate=negate)})"
                for operand in operands
            )
        case IsNull(field):
            absent = _property_absent_expression(field)
            return f"not ({absent})" if negate else absent
        case Equals() | Ordering() | In():
            condition = _condition_expression(expr)
            if negate:
                return f"(not ({condition})) || ({_property_absent_expression(expr.field)})"
            return condition


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

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {Equals, Ordering, In, IsNull, And, Or, Not}
    )

    def __init__(
        self,
        *,
        client: AsyncMilvusClient,
        collection_name: str,
        vector_store_name: str,
        registration: Registration,
        vector_dimensions: int,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
        request_timeout_seconds: int,
        search_params: dict[str, bool | int | float | str],
    ) -> None:
        """Initialize with a Milvus client and the registration the handle is bound to."""
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
        self._search_params = search_params

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return MilvusVectorStorePartition._SUPPORTED_FILTER_NODES

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
        min_cosine_similarity: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        filter_expr = _incarnation_filter(self._incarnation)
        if property_filter is not None:
            property_expr = _filter_expression(property_filter)
            filter_expr = f"({filter_expr}) && ({property_expr})"

        raw_results = await self._client.search(
            collection_name=self._collection_name,
            data=query_vectors,
            filter=filter_expr,
            limit=limit,
            search_params={"params": self._search_params},
            output_fields=[_RECORD_UUID_FIELD],
            anns_field=_VECTOR_FIELD,
            timeout=self._request_timeout_seconds,
        )

        # Milvus returns each query's hits best first.
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


class MilvusVectorIndex(BaseModel):
    """
    A vector index of the store's collection and the parameters its searches pass.

    Milvus checks the parameters the index type takes when it creates the
    index and when a search reaches an indexed segment, and ignores other
    keys. The collection isolates partitions by key, which Milvus supports
    for the HNSW family only, and the metric is cosine.

    Attributes:
        index_type (str):
            The Milvus index type.
        params (dict[str, bool | int | float | str]):
            The index's build parameters, other than its type and metric
            (default: empty).
        search_params (dict[str, bool | int | float | str]):
            The parameters every search of the index passes (default: empty).
    """

    model_config = ConfigDict(extra="forbid")

    index_type: str = Field(..., description="The Milvus index type")
    params: dict[str, bool | int | float | str] = Field(
        default_factory=dict,
        description="The index's build parameters, other than its type and metric",
    )
    search_params: dict[str, bool | int | float | str] = Field(
        default_factory=dict,
        description="The parameters every search of the index passes",
    )

    @field_validator("params")
    @classmethod
    def _reject_type_and_metric(
        cls, params: dict[str, bool | int | float | str]
    ) -> dict[str, bool | int | float | str]:
        reserved = sorted(params.keys() & {"index_type", "metric_type"})
        if reserved:
            raise ValueError(
                f"params sets {', '.join(reserved)}: the index type is the "
                "index_type field and the metric is the store's cosine"
            )
        return params


_DEFAULT_VECTOR_INDEX = MilvusVectorIndex(
    index_type="HNSW_SQ",
    params={
        "M": 18,
        "efConstruction": 240,
        "sq_type": "SQ4U",
        "refine": True,
        "refine_type": "FP16",
    },
    search_params={"refine_k": 8},
)
"""The vector index unless one is configured.

HNSW_SQ with explicit parameters, so every server builds the same index
whatever its version or AUTOINDEX configuration. A search rescores
refine_k candidates per result against the FP16 vectors refine keeps.
"""


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
        vector_index (MilvusVectorIndex | None):
            The vector index of the store's collection and the parameters
            every search passes. Startup creates the index unless the
            collection has one, in which case the collection keeps its
            index; the search parameters apply to every query from startup
            on (default: None, which selects HNSW_SQ).
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
    vector_index: MilvusVectorIndex | None = Field(
        None,
        description=(
            "The vector index of the store's collection and the parameters "
            "every search passes. Startup creates the index unless the "
            "collection has one, in which case the collection keeps its "
            "index; the search parameters apply to every query from startup on"
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

    _MILVUS_METRIC_TYPE: ClassVar[str] = "COSINE"

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__(params, metrics_prefix="vector_store_milvus")
        self._client = params.client
        self._collection_name = f"{_COLLECTION_NAME_PREFIX}{params.vector_store_name}"
        self._request_timeout_seconds = params.request_timeout_seconds
        self._max_varchar_length = params.max_varchar_length
        self._purge_batch_size = params.purge_batch_size
        self._vector_index = (
            params.vector_index
            if params.vector_index is not None
            else _DEFAULT_VECTOR_INDEX
        )

    @override
    async def _prepare_storage(self) -> None:
        # Each step runs when it is missing, and the collection is loaded, so
        # the next startup completes one that failed part way.
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_name=_VECTOR_FIELD,
            index_type=self._vector_index.index_type,
            metric_type=MilvusVectorStore._MILVUS_METRIC_TYPE,
            params=self._vector_index.params,
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
            indexed_properties=self.indexed_properties,
            tracker=self._tracker,
            request_timeout_seconds=self._request_timeout_seconds,
            search_params=self._vector_index.search_params,
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
