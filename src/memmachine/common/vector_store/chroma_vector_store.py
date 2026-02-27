"""ChromaDB-based vector store implementation."""

import logging
import time
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import (
    BoolInvertedIndexConfig,
    FloatInvertedIndexConfig,
    IntInvertedIndexConfig,
    Schema,
    StringInvertedIndexConfig,
)
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.metrics_factory import MetricsFactory

from .data_types import PropertyValue, QueryResult, Record
from .vector_store import Collection, VectorStore

logger = logging.getLogger(__name__)

_DATETIME_PREFIX = "__dt__:"
_NULL_SENTINEL = "__null__"
_ESCAPED_PREFIX = "__esc__:"

_SIMILARITY_METRIC_TO_CHROMA_DISTANCE: dict[SimilarityMetric, str] = {
    SimilarityMetric.COSINE: "cosine",
    SimilarityMetric.DOT: "ip",
    SimilarityMetric.EUCLIDEAN: "l2",
}

_CHROMA_DISTANCE_TO_SIMILARITY_METRIC: dict[str, SimilarityMetric] = {
    v: k for k, v in _SIMILARITY_METRIC_TO_CHROMA_DISTANCE.items()
}

_OP_MAP: dict[str, str] = {
    "=": "$eq",
    "!=": "$ne",
    ">": "$gt",
    ">=": "$gte",
    "<": "$lt",
    "<=": "$lte",
}

_INVERSE_OP_MAP: dict[str, str] = {
    "=": "$ne",
    "!=": "$eq",
    ">": "$lte",
    ">=": "$lt",
    "<": "$gte",
    "<=": "$gt",
}

_PROPERTY_TYPE_TO_INDEX_CONFIG: dict[
    type,
    BoolInvertedIndexConfig
    | IntInvertedIndexConfig
    | FloatInvertedIndexConfig
    | StringInvertedIndexConfig,
] = {
    bool: BoolInvertedIndexConfig(),
    int: IntInvertedIndexConfig(),
    float: FloatInvertedIndexConfig(),
    str: StringInvertedIndexConfig(),
    datetime: StringInvertedIndexConfig(),  # datetimes serialize to "__dt__:" prefixed strings
}


def _build_chroma_schema(
    properties_schema: Mapping[str, type[PropertyValue]],
) -> Schema:
    """Build a ChromaDB Schema with inverted indexes for the given properties."""
    schema = Schema()
    for field_name, field_type in properties_schema.items():
        index_config = _PROPERTY_TYPE_TO_INDEX_CONFIG.get(field_type)
        if index_config is None:
            msg = f"Unsupported property type for indexing: {field_type}"
            raise TypeError(msg)
        schema.create_index(config=index_config, key=field_name)
    return schema


def _serialize_property_value(value: PropertyValue) -> str | int | float | bool:
    """Serialize a property value for Chroma metadata storage."""
    if isinstance(value, datetime):
        return f"{_DATETIME_PREFIX}{value.isoformat()}"
    if isinstance(value, str) and (
        value == _NULL_SENTINEL or value.startswith(_ESCAPED_PREFIX)
    ):
        return f"{_ESCAPED_PREFIX}{value}"
    return value


def _deserialize_property_value(value: bool | float | str) -> PropertyValue:
    """Deserialize a Chroma metadata value back to a PropertyValue."""
    if isinstance(value, str):
        if value.startswith(_ESCAPED_PREFIX):
            return value[len(_ESCAPED_PREFIX) :]
        if value.startswith(_DATETIME_PREFIX):
            return datetime.fromisoformat(value[len(_DATETIME_PREFIX) :])
    return value


def _distance_to_similarity(distance: float, metric: SimilarityMetric) -> float:
    """Convert a Chroma distance value to a similarity score."""
    if metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
        return 1.0 - distance
    # Euclidean
    return 1.0 / (1.0 + distance)


def _build_chroma_where(expr: FilterExpr) -> dict[str, Any]:
    """Convert a FilterExpr tree to a Chroma ``where`` dictionary."""
    if isinstance(expr, FilterComparison):
        return _build_chroma_comparison(expr)
    if isinstance(expr, FilterIn):
        return _build_chroma_in(expr)
    if isinstance(expr, FilterIsNull):
        return {expr.field: {"$eq": _NULL_SENTINEL}}
    if isinstance(expr, FilterNot):
        return _build_chroma_not(expr)
    if isinstance(expr, FilterAnd):
        return {
            "$and": [_build_chroma_where(expr.left), _build_chroma_where(expr.right)]
        }
    if isinstance(expr, FilterOr):
        return {
            "$or": [_build_chroma_where(expr.left), _build_chroma_where(expr.right)]
        }
    msg = f"Unsupported filter expression type: {type(expr)}"
    raise NotImplementedError(msg)


def _build_chroma_comparison(comp: FilterComparison) -> dict[str, Any]:
    """Convert a single Comparison node into a Chroma where clause."""
    chroma_op = _OP_MAP.get(comp.op)
    if chroma_op is None:
        msg = f"Unsupported filter operator: {comp.op}"
        raise NotImplementedError(msg)
    serialized_value = _serialize_property_value(comp.value)
    return {comp.field: {chroma_op: serialized_value}}


def _build_chroma_in(expr: FilterIn) -> dict[str, Any]:
    """Convert an In node into a Chroma where clause."""
    serialized = [_serialize_property_value(v) for v in expr.values]
    return {expr.field: {"$in": serialized}}


def _build_chroma_not(expr: FilterNot) -> dict[str, Any]:
    """Convert a Not node into a Chroma where clause.

    ChromaDB has no generic ``$not`` operator, so we handle each case
    by algebraic inversion.
    """
    inner = expr.expr
    if isinstance(inner, FilterComparison):
        chroma_op = _INVERSE_OP_MAP.get(inner.op)
        if chroma_op is None:
            msg = f"Cannot negate operator: {inner.op}"
            raise NotImplementedError(msg)
        serialized_value = _serialize_property_value(inner.value)
        return {inner.field: {chroma_op: serialized_value}}
    if isinstance(inner, FilterIn):
        serialized = [_serialize_property_value(v) for v in inner.values]
        return {inner.field: {"$nin": serialized}}
    if isinstance(inner, FilterIsNull):
        return {inner.field: {"$ne": _NULL_SENTINEL}}
    if isinstance(inner, FilterNot):
        # Double negation elimination
        return _build_chroma_where(inner.expr)
    if isinstance(inner, FilterAnd):
        # De Morgan: NOT(A AND B) → (NOT A) OR (NOT B)
        return {
            "$or": [
                _build_chroma_not(FilterNot(expr=inner.left)),
                _build_chroma_not(FilterNot(expr=inner.right)),
            ]
        }
    if isinstance(inner, FilterOr):
        # De Morgan: NOT(A OR B) → (NOT A) AND (NOT B)
        return {
            "$and": [
                _build_chroma_not(FilterNot(expr=inner.left)),
                _build_chroma_not(FilterNot(expr=inner.right)),
            ]
        }
    msg = f"Unsupported filter expression type inside Not: {type(inner)}"
    raise NotImplementedError(msg)


class ChromaVectorStoreParams(BaseModel):
    """Parameters for ChromaVectorStore.

    Attributes:
        client (AsyncClientAPI):
            Pre-constructed async Chroma client.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        user_metrics_labels (dict[str, str]):
            Labels to attach to the collected metrics
            (default: {}).

    """

    client: InstanceOf[AsyncClientAPI] = Field(
        ...,
        description="Pre-constructed async Chroma client",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels to attach to the collected metrics",
    )


class ChromaVectorStore(VectorStore):
    """Asynchronous ChromaDB-based implementation of VectorStore."""

    def __init__(self, params: ChromaVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()

        self._client: AsyncClientAPI = params.client
        self._collection_metrics: dict[str, SimilarityMetric] = {}
        self._collection_schemas: dict[str, Mapping[str, type[PropertyValue]]] = {}
        self._collection_handles: dict[str, ChromaCollection] = {}

        # Metrics initialization
        self._create_collection_calls_counter = None
        self._create_collection_latency_summary = None
        self._get_collection_calls_counter = None
        self._get_collection_latency_summary = None
        self._delete_collection_calls_counter = None
        self._delete_collection_latency_summary = None
        self._upsert_calls_counter = None
        self._upsert_latency_summary = None
        self._query_calls_counter = None
        self._query_latency_summary = None
        self._get_calls_counter = None
        self._get_latency_summary = None
        self._delete_calls_counter = None
        self._delete_latency_summary = None

        self._should_collect_metrics = False
        metrics_factory = params.metrics_factory
        if metrics_factory is not None:
            self._should_collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
            label_names = self._user_metrics_labels.keys()

            self._create_collection_calls_counter = metrics_factory.get_counter(
                "vector_store_chroma_create_collection_calls",
                "Number of calls to create_collection in ChromaVectorStore",
                label_names=label_names,
            )
            self._create_collection_latency_summary = metrics_factory.get_summary(
                "vector_store_chroma_create_collection_latency_seconds",
                "Latency in seconds for create_collection in ChromaVectorStore",
                label_names=label_names,
            )
            self._get_collection_calls_counter = metrics_factory.get_counter(
                "vector_store_chroma_get_collection_calls",
                "Number of calls to get_collection in ChromaVectorStore",
                label_names=label_names,
            )
            self._get_collection_latency_summary = metrics_factory.get_summary(
                "vector_store_chroma_get_collection_latency_seconds",
                "Latency in seconds for get_collection in ChromaVectorStore",
                label_names=label_names,
            )
            self._delete_collection_calls_counter = metrics_factory.get_counter(
                "vector_store_chroma_delete_collection_calls",
                "Number of calls to delete_collection in ChromaVectorStore",
                label_names=label_names,
            )
            self._delete_collection_latency_summary = metrics_factory.get_summary(
                "vector_store_chroma_delete_collection_latency_seconds",
                "Latency in seconds for delete_collection in ChromaVectorStore",
                label_names=label_names,
            )
            self._upsert_calls_counter = metrics_factory.get_counter(
                "vector_store_chroma_upsert_calls",
                "Number of calls to upsert in ChromaVectorStore",
                label_names=label_names,
            )
            self._upsert_latency_summary = metrics_factory.get_summary(
                "vector_store_chroma_upsert_latency_seconds",
                "Latency in seconds for upsert in ChromaVectorStore",
                label_names=label_names,
            )
            self._query_calls_counter = metrics_factory.get_counter(
                "vector_store_chroma_query_calls",
                "Number of calls to query in ChromaVectorStore",
                label_names=label_names,
            )
            self._query_latency_summary = metrics_factory.get_summary(
                "vector_store_chroma_query_latency_seconds",
                "Latency in seconds for query in ChromaVectorStore",
                label_names=label_names,
            )
            self._get_calls_counter = metrics_factory.get_counter(
                "vector_store_chroma_get_calls",
                "Number of calls to get in ChromaVectorStore",
                label_names=label_names,
            )
            self._get_latency_summary = metrics_factory.get_summary(
                "vector_store_chroma_get_latency_seconds",
                "Latency in seconds for get in ChromaVectorStore",
                label_names=label_names,
            )
            self._delete_calls_counter = metrics_factory.get_counter(
                "vector_store_chroma_delete_calls",
                "Number of calls to delete in ChromaVectorStore",
                label_names=label_names,
            )
            self._delete_latency_summary = metrics_factory.get_summary(
                "vector_store_chroma_delete_latency_seconds",
                "Latency in seconds for delete in ChromaVectorStore",
                label_names=label_names,
            )

    async def startup(self) -> None:
        """No-op: client is ready on construction."""

    async def shutdown(self) -> None:
        """No-op: caller manages client lifecycle."""

    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,  # noqa: ARG002
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """Create a collection in the vector store."""
        start_time = time.monotonic()

        if similarity_metric == SimilarityMetric.MANHATTAN:
            msg = "ChromaDB does not support the Manhattan similarity metric"
            raise ValueError(msg)

        distance_metric = _SIMILARITY_METRIC_TO_CHROMA_DISTANCE[similarity_metric]

        chroma_schema: Schema | None = None
        if properties_schema is not None:
            chroma_schema = _build_chroma_schema(properties_schema)

        await self._client.create_collection(
            name=collection_name,
            metadata={"distance_metric": distance_metric},
            schema=chroma_schema,
        )
        self._collection_metrics[collection_name] = similarity_metric
        if properties_schema is not None:
            self._collection_schemas[collection_name] = properties_schema

        end_time = time.monotonic()
        self._collect_metrics(
            self._create_collection_calls_counter,
            self._create_collection_latency_summary,
            start_time,
            end_time,
        )

    async def get_collection(self, collection_name: str) -> Collection:
        """Get a collection proxy. Handle persists across deletion and creation."""
        start_time = time.monotonic()

        if collection_name not in self._collection_handles:
            self._collection_handles[collection_name] = ChromaCollection(
                store=self,
                collection_name=collection_name,
            )

        end_time = time.monotonic()
        self._collect_metrics(
            self._get_collection_calls_counter,
            self._get_collection_latency_summary,
            start_time,
            end_time,
        )
        return self._collection_handles[collection_name]

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector store."""
        start_time = time.monotonic()

        await self._client.delete_collection(name=collection_name)
        self._collection_metrics.pop(collection_name, None)
        self._collection_schemas.pop(collection_name, None)

        end_time = time.monotonic()
        self._collect_metrics(
            self._delete_collection_calls_counter,
            self._delete_collection_latency_summary,
            start_time,
            end_time,
        )

    # -- Internal helpers used by ChromaCollection ----------------------------

    def resolve_chroma_collection(
        self, name: str
    ) -> Coroutine[None, None, AsyncCollection]:
        """Return coroutine that resolves a Chroma collection by name."""
        return self._client.get_collection(name)

    def get_cached_similarity_metric(self, name: str) -> SimilarityMetric | None:
        """Return cached similarity metric for a collection, if available."""
        return self._collection_metrics.get(name)

    def cache_similarity_metric(self, name: str, metric: SimilarityMetric) -> None:
        """Cache a similarity metric for a collection."""
        self._collection_metrics[name] = metric

    def get_cached_properties_schema(
        self, name: str
    ) -> Mapping[str, type[PropertyValue]] | None:
        """Return cached properties schema for a collection, if available."""
        return self._collection_schemas.get(name)

    def _collect_metrics(
        self,
        calls_counter: MetricsFactory.Counter | None,
        latency_summary: MetricsFactory.Summary | None,
        start_time: float,
        end_time: float,
    ) -> None:
        """Increment calls and observe latency."""
        if self._should_collect_metrics:
            cast(MetricsFactory.Counter, calls_counter).increment(
                labels=self._user_metrics_labels,
            )
            cast(MetricsFactory.Summary, latency_summary).observe(
                value=end_time - start_time,
                labels=self._user_metrics_labels,
            )

    def collect_operation_metrics(
        self, operation: str, start_time: float, end_time: float
    ) -> None:
        """Collect metrics for a collection-level operation."""
        counter_map = {
            "upsert": self._upsert_calls_counter,
            "query": self._query_calls_counter,
            "get": self._get_calls_counter,
            "delete": self._delete_calls_counter,
        }
        summary_map = {
            "upsert": self._upsert_latency_summary,
            "query": self._query_latency_summary,
            "get": self._get_latency_summary,
            "delete": self._delete_latency_summary,
        }
        self._collect_metrics(
            counter_map.get(operation),
            summary_map.get(operation),
            start_time,
            end_time,
        )


class ChromaCollection(Collection):
    """Proxy around a Chroma collection that lazily resolves on each call."""

    def __init__(self, *, store: ChromaVectorStore, collection_name: str) -> None:
        """Initialize the collection proxy."""
        self._store = store
        self._collection_name = collection_name

    async def _get_chroma_collection(self) -> AsyncCollection:
        """Resolve the underlying Chroma collection."""
        return await self._store.resolve_chroma_collection(self._collection_name)

    async def _get_similarity_metric(self) -> SimilarityMetric:
        """Get the similarity metric for this collection."""
        cached = self._store.get_cached_similarity_metric(self._collection_name)
        if cached is not None:
            return cached

        collection = await self._get_chroma_collection()
        metadata = collection.metadata or {}
        distance = metadata.get("distance_metric", "cosine")
        metric = _CHROMA_DISTANCE_TO_SIMILARITY_METRIC.get(
            str(distance), SimilarityMetric.COSINE
        )
        self._store.cache_similarity_metric(self._collection_name, metric)
        return metric

    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records in the collection."""
        start_time = time.monotonic()

        records_list = list(records)
        if not records_list:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        schema = self._store.get_cached_properties_schema(self._collection_name)

        for record in records_list:
            ids.append(str(record.uuid))
            embeddings.append(record.vector if record.vector is not None else [])
            metadata: dict[str, str | int | float | bool] = {}
            if record.properties:
                for key, value in record.properties.items():
                    metadata[key] = _serialize_property_value(value)
            if schema is not None:
                for key in schema:
                    if key not in metadata:
                        metadata[key] = _NULL_SENTINEL
            metadatas.append(metadata)

        collection = await self._get_chroma_collection()
        await collection.upsert(
            ids=ids,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
        )

        end_time = time.monotonic()
        self._store.collect_operation_metrics("upsert", start_time, end_time)

    async def query(
        self,
        *,
        query_vector: Sequence[float],
        similarity_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[QueryResult]:
        """Query for records matching the criteria by vector similarity."""
        start_time = time.monotonic()

        collection = await self._get_chroma_collection()

        if limit is None:
            n_results = await collection.count()
            if n_results == 0:
                end_time = time.monotonic()
                self._store.collect_operation_metrics("query", start_time, end_time)
                return []
        else:
            n_results = limit

        include: list[str] = ["distances"]
        if return_properties:
            include.append("metadatas")
        if return_vector:
            include.append("embeddings")

        where = (
            _build_chroma_where(property_filter)
            if property_filter is not None
            else None
        )

        raw = await collection.query(
            query_embeddings=[list(query_vector)],
            n_results=n_results,
            where=where,
            include=include,  # type: ignore[arg-type]
        )

        similarity_metric = await self._get_similarity_metric()

        results: list[QueryResult] = []
        raw_ids = raw.get("ids", [[]])[0]
        raw_distances = (raw.get("distances") or [[]])[0]
        raw_metadatas = (raw.get("metadatas") or [[]])[0]
        raw_embeddings = (raw.get("embeddings") or [[]])[0]

        for i, record_id in enumerate(raw_ids):
            distance = raw_distances[i] if i < len(raw_distances) else 0.0
            score = _distance_to_similarity(distance, similarity_metric)

            if similarity_threshold is not None and score < similarity_threshold:
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties and i < len(raw_metadatas) and raw_metadatas[i]:
                properties = {
                    k: _deserialize_property_value(v)
                    for k, v in raw_metadatas[i].items()
                    if isinstance(v, (str, int, float, bool)) and v != _NULL_SENTINEL
                }

            vector: list[float] | None = None
            if return_vector and i < len(raw_embeddings) and raw_embeddings[i]:
                vector = list(raw_embeddings[i])

            results.append(
                QueryResult(
                    score=score,
                    record=Record(
                        uuid=UUID(record_id),
                        vector=vector,
                        properties=properties,
                    ),
                )
            )

        end_time = time.monotonic()
        self._store.collect_operation_metrics("query", start_time, end_time)
        return results

    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records from the collection by their UUIDs."""
        start_time = time.monotonic()

        uuid_list = list(record_uuids)
        if not uuid_list:
            return []

        ids = [str(u) for u in uuid_list]

        include: list[str] = []
        if return_properties:
            include.append("metadatas")
        if return_vector:
            include.append("embeddings")

        collection = await self._get_chroma_collection()
        raw = await collection.get(
            ids=ids,
            include=include,  # type: ignore[arg-type]
        )

        raw_ids = raw.get("ids", [])
        raw_metadatas = raw.get("metadatas") or []
        raw_embeddings = raw.get("embeddings") or []

        # Build a lookup by id for ordering
        record_map: dict[str, Record] = {}
        for i, record_id in enumerate(raw_ids):
            properties: dict[str, PropertyValue] | None = None
            if return_properties and i < len(raw_metadatas) and raw_metadatas[i]:
                properties = {
                    k: _deserialize_property_value(v)
                    for k, v in raw_metadatas[i].items()
                    if isinstance(v, (str, int, float, bool)) and v != _NULL_SENTINEL
                }

            vector: list[float] | None = None
            if return_vector and i < len(raw_embeddings) and raw_embeddings[i]:
                vector = list(raw_embeddings[i])

            record_map[record_id] = Record(
                uuid=UUID(record_id),
                vector=vector,
                properties=properties,
            )

        # Return in input order
        results = [record_map[str(u)] for u in uuid_list if str(u) in record_map]

        end_time = time.monotonic()
        self._store.collect_operation_metrics("get", start_time, end_time)
        return results

    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """Delete records from the collection by their UUIDs."""
        start_time = time.monotonic()

        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        ids = [str(u) for u in uuid_list]
        collection = await self._get_chroma_collection()
        await collection.delete(ids=ids)

        end_time = time.monotonic()
        self._store.collect_operation_metrics("delete", start_time, end_time)
