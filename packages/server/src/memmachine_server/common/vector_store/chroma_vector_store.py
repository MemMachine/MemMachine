"""Chroma-based vector store implementation."""

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, cast, override
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
from chromadb.api.types import (
    QueryResult as ChromaQueryResult,
)
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
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

from .data_types import QueryResult, Record
from .vector_store import Collection, VectorStore

logger = logging.getLogger(__name__)

_SCHEMA_METADATA_KEY_PREFIX = "schema."
_SIMILARITY_METRIC_METADATA_KEY = "similarity_metric"

_ChromaPropertyValue = bool | int | float | str


def _serialize_schema(
    schema: Mapping[str, type[PropertyValue]],
) -> dict[str, str]:
    """Serialize a properties schema to collection metadata entries.

    Returns one metadata key per type, e.g.
    ``{"schema.str": '["name","email"]', "schema.int": '["age"]'}``.
    """
    type_properties_map: dict[str, list[str]] = {}
    for property_name, property_type in schema.items():
        property_type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]
        type_properties_map.setdefault(property_type_name, []).append(property_name)
    return {
        f"{_SCHEMA_METADATA_KEY_PREFIX}{property_type_name}": json.dumps(property_names)
        for property_type_name, property_names in type_properties_map.items()
    }


def _deserialize_schema(
    metadata: Mapping[str, Any],
) -> dict[str, type[PropertyValue]]:
    """Deserialize collection metadata entries back to a properties schema."""
    schema: dict[str, type[PropertyValue]] = {}
    for key, value in metadata.items():
        if not key.startswith(_SCHEMA_METADATA_KEY_PREFIX):
            continue
        property_type_name = key.removeprefix(_SCHEMA_METADATA_KEY_PREFIX)
        property_type = PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE.get(property_type_name)
        if property_type is None:
            continue
        property_names = json.loads(value) if isinstance(value, str) else value
        for property_name in property_names:
            schema[property_name] = property_type
    return schema


class ChromaVectorStoreParams(BaseModel):
    """Parameters for ChromaVectorStore.

    Attributes:
        client (AsyncClientAPI):
            Pre-constructed async Chroma client.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    client: InstanceOf[AsyncClientAPI] = Field(
        ...,
        description="Pre-constructed async Chroma client",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class ChromaVectorStore(VectorStore):
    """Asynchronous Chroma-based implementation of VectorStore."""

    _SIMILARITY_METRIC_TO_CHROMA_DISTANCE: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.DOT: "ip",
        SimilarityMetric.EUCLIDEAN: "l2",
    }

    _PROPERTY_TYPE_TO_INDEX_CONFIG: ClassVar[
        dict[
            type,
            BoolInvertedIndexConfig
            | IntInvertedIndexConfig
            | FloatInvertedIndexConfig
            | StringInvertedIndexConfig,
        ]
    ] = {
        bool: BoolInvertedIndexConfig(),
        int: IntInvertedIndexConfig(),
        float: FloatInvertedIndexConfig(),
        str: StringInvertedIndexConfig(),
        datetime: IntInvertedIndexConfig(),
    }

    def __init__(self, params: ChromaVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncClientAPI = params.client
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_chroma",
        )

    @override
    async def startup(self) -> None:
        """No-op: client is ready on construction."""

    @override
    async def shutdown(self) -> None:
        """No-op: caller manages client lifecycle."""

    @override
    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """Create a collection in the vector store."""
        async with self._tracker("create_collection"):
            distance_metric = (
                ChromaVectorStore._SIMILARITY_METRIC_TO_CHROMA_DISTANCE.get(
                    similarity_metric
                )
            )
            if distance_metric is None:
                msg = f"Chroma does not support the {similarity_metric.value} similarity metric"
                raise ValueError(msg)

            chroma_schema: Schema | None = None
            if properties_schema is not None:
                chroma_schema = ChromaVectorStore._build_chroma_schema(
                    properties_schema
                )

            metadata: dict[str, str] = {
                _SIMILARITY_METRIC_METADATA_KEY: similarity_metric.value,
                **_serialize_schema(properties_schema or {}),
            }

            await self._client.create_collection(
                name=collection_name,
                metadata=metadata,
                schema=chroma_schema,
            )

    @override
    async def get_collection(self, collection_name: str) -> Collection:
        """Get a collection handle."""
        async with self._tracker("get_collection"):
            return ChromaCollection(
                store=self,
                collection_name=collection_name,
                tracker=self._tracker,
            )

    @override
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector store."""
        async with self._tracker("delete_collection"):
            await self._client.delete_collection(name=collection_name)

    async def resolve_chroma_collection(self, name: str) -> AsyncCollection:
        """Return coroutine that resolves a Chroma collection by name."""
        return await self._client.get_collection(name)

    @staticmethod
    def _build_chroma_schema(
        properties_schema: Mapping[str, type[PropertyValue]],
    ) -> Schema:
        """Build a Chroma Schema with inverted indexes for the given properties."""
        schema = Schema()
        for field_name, field_type in properties_schema.items():
            index_config = ChromaVectorStore._PROPERTY_TYPE_TO_INDEX_CONFIG.get(
                field_type
            )
            if index_config is None:
                msg = f"Unsupported property type for indexing: {field_type}"
                raise TypeError(msg)
            schema.create_index(config=index_config, key=field_name)
        return schema


class ChromaCollection(Collection):
    """Proxy around a Chroma collection that caches the resolved handle."""

    _EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
    _MICROSECOND = timedelta(microseconds=1)
    _NULL_SENTINEL = "__null__"
    _ESCAPED_PREFIX = "__esc__:"

    _CHROMA_DISTANCE_TO_SIMILARITY_METRIC: ClassVar[dict[str, SimilarityMetric]] = {
        "cosine": SimilarityMetric.COSINE,
        "ip": SimilarityMetric.DOT,
        "l2": SimilarityMetric.EUCLIDEAN,
    }

    _OP_MAP: ClassVar[dict[str, str]] = {
        "=": "$eq",
        "!=": "$ne",
        ">": "$gt",
        ">=": "$gte",
        "<": "$lt",
        "<=": "$lte",
    }

    _INVERSE_OP_MAP: ClassVar[dict[str, str]] = {
        "=": "$ne",
        "!=": "$eq",
        ">": "$lte",
        ">=": "$lt",
        "<": "$gte",
        "<=": "$gt",
    }

    def __init__(
        self,
        *,
        store: ChromaVectorStore,
        collection_name: str,
        tracker: OperationTracker,
    ) -> None:
        """Initialize the collection proxy."""
        self._store = store
        self._tracker = tracker
        self._collection_name = collection_name
        self._cached_collection: AsyncCollection | None = None
        self._properties_schema: Mapping[str, type[PropertyValue]] | None = None
        self._similarity_metric: SimilarityMetric | None = None

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records in the collection."""
        async with self._tracker("upsert"):
            records_list = list(records)
            if not records_list:
                return

            collection = await self._get_chroma_collection()

            if any(r.vector is None for r in records_list):
                msg = "All records must have vectors"
                raise ValueError(msg)

            ids: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict[str, _ChromaPropertyValue] | None] = []

            for record in records_list:
                ids.append(str(record.uuid))
                assert record.vector is not None  # Validated above.
                embeddings.append(record.vector)
                metadata: dict[str, _ChromaPropertyValue] = {}
                if record.properties:
                    for key, value in record.properties.items():
                        metadata[key] = ChromaCollection._serialize_property_value(
                            value
                        )
                if self._properties_schema is not None:
                    for key in self._properties_schema:
                        if key not in metadata:
                            metadata[key] = ChromaCollection._NULL_SENTINEL
                metadatas.append(metadata or None)

            await collection.upsert(
                ids=ids,
                embeddings=cast(Any, embeddings),
                metadatas=cast(Any, metadatas),
            )

    @override
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
        async with self._tracker("query"):
            collection = await self._get_chroma_collection()
            raw = await self._execute_query(
                collection,
                query_vector,
                limit,
                property_filter,
                return_vector,
                return_properties,
            )

            if raw is None:
                return []

            similarity_metric = await self._get_similarity_metric()
            return self._build_query_results(
                raw,
                similarity_metric,
                similarity_threshold,
                return_vector,
                return_properties,
                self._properties_schema,
            )

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records from the collection by their UUIDs."""
        async with self._tracker("get"):
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
                include=cast(Any, include),
            )

            raw_ids = raw.get("ids", [])
            raw_metadatas_val = raw.get("metadatas")
            raw_metadatas = raw_metadatas_val if raw_metadatas_val is not None else []
            raw_embeddings_val = raw.get("embeddings")
            raw_embeddings = (
                raw_embeddings_val if raw_embeddings_val is not None else []
            )

            schema = self._properties_schema

            # Build a lookup by id for ordering
            record_map: dict[str, Record] = {}
            for i, record_id in enumerate(raw_ids):
                properties: dict[str, PropertyValue] | None = None
                if return_properties and i < len(raw_metadatas) and raw_metadatas[i]:
                    properties = {
                        k: ChromaCollection._deserialize_property_value(
                            v,
                            as_datetime=schema is not None
                            and schema.get(k) is datetime,
                        )
                        for k, v in raw_metadatas[i].items()
                        if isinstance(v, _ChromaPropertyValue)
                        and v != ChromaCollection._NULL_SENTINEL
                    }

                vector: list[float] | None = None
                if (
                    return_vector
                    and i < len(raw_embeddings)
                    and len(raw_embeddings[i]) > 0
                ):
                    vector = list(raw_embeddings[i])

                record_map[record_id] = Record(
                    uuid=UUID(record_id),
                    vector=vector,
                    properties=properties,
                )

            # Return in input order
            return [record_map[str(u)] for u in uuid_list if str(u) in record_map]

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

            ids = [str(u) for u in uuid_list]

            collection = await self._get_chroma_collection()
            await collection.delete(ids=ids)

    async def _get_chroma_collection(self) -> AsyncCollection:
        """Return cached Chroma collection handle, resolving on first call."""
        if self._cached_collection is not None:
            return self._cached_collection
        self._cached_collection = await self._store.resolve_chroma_collection(
            self._collection_name
        )
        if self._properties_schema is None:
            metadata = self._cached_collection.metadata or {}
            schema = _deserialize_schema(metadata)
            if schema:
                self._properties_schema = schema
        return self._cached_collection

    async def _get_similarity_metric(self) -> SimilarityMetric:
        """Get the similarity metric for this collection."""
        if self._similarity_metric is not None:
            return self._similarity_metric

        collection = await self._get_chroma_collection()
        metadata = collection.metadata or {}
        distance = metadata.get(_SIMILARITY_METRIC_METADATA_KEY, "cosine")
        metric = ChromaCollection._CHROMA_DISTANCE_TO_SIMILARITY_METRIC.get(
            str(distance), SimilarityMetric.COSINE
        )
        self._similarity_metric = metric
        return metric

    async def _execute_query(
        self,
        collection: AsyncCollection,
        query_vector: Sequence[float],
        limit: int | None,
        property_filter: FilterExpr | None,
        return_vector: bool,
        return_properties: bool,
    ) -> ChromaQueryResult | None:
        """Run the underlying Chroma query, returning None for empty collections."""
        if limit is None:
            n_results = await collection.count()
            if n_results == 0:
                return None
        else:
            n_results = limit

        include: list[str] = ["distances"]
        if return_properties:
            include.append("metadatas")
        if return_vector:
            include.append("embeddings")

        where = (
            ChromaCollection._build_chroma_filter(property_filter)
            if property_filter is not None
            else None
        )

        return await collection.query(
            query_embeddings=[list(query_vector)],
            n_results=n_results,
            where=where,
            include=cast(Any, include),
        )

    @staticmethod
    def _build_query_results(
        raw: ChromaQueryResult,
        similarity_metric: SimilarityMetric,
        similarity_threshold: float | None,
        return_vector: bool,
        return_properties: bool,
        schema: Mapping[str, type[PropertyValue]] | None,
    ) -> list[QueryResult]:
        """Parse a raw Chroma query response into QueryResult objects."""
        results: list[QueryResult] = []
        raw_ids = raw.get("ids", [[]])[0]
        distances_val = raw.get("distances")
        raw_distances = (distances_val if distances_val is not None else [[]])[0]
        metadatas_val = raw.get("metadatas")
        raw_metadatas = (metadatas_val if metadatas_val is not None else [[]])[0]
        embeddings_val = raw.get("embeddings")
        raw_embeddings = (embeddings_val if embeddings_val is not None else [[]])[0]

        for i, record_id in enumerate(raw_ids):
            distance = raw_distances[i] if i < len(raw_distances) else 0.0
            score = ChromaCollection._distance_to_similarity(
                distance, similarity_metric
            )

            if similarity_threshold is not None and score < similarity_threshold:
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties and i < len(raw_metadatas) and raw_metadatas[i]:
                properties = {
                    k: ChromaCollection._deserialize_property_value(
                        v,
                        as_datetime=schema is not None and schema.get(k) is datetime,
                    )
                    for k, v in raw_metadatas[i].items()
                    if isinstance(v, _ChromaPropertyValue)
                    and v != ChromaCollection._NULL_SENTINEL
                }

            vector: list[float] | None = None
            if return_vector and i < len(raw_embeddings) and len(raw_embeddings[i]) > 0:
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
        return results

    @staticmethod
    def _serialize_property_value(value: PropertyValue) -> _ChromaPropertyValue:
        """Serialize a property value for Chroma metadata storage."""
        if isinstance(value, datetime):
            return (value - ChromaCollection._EPOCH) // ChromaCollection._MICROSECOND
        if isinstance(value, str) and (
            value == ChromaCollection._NULL_SENTINEL
            or value.startswith(ChromaCollection._ESCAPED_PREFIX)
        ):
            return f"{ChromaCollection._ESCAPED_PREFIX}{value}"
        return value

    @staticmethod
    def _deserialize_property_value(
        value: _ChromaPropertyValue,
        *,
        as_datetime: bool = False,
    ) -> PropertyValue:
        """Deserialize a Chroma metadata value back to a PropertyValue.

        Args:
            value: The raw Chroma metadata value.
            as_datetime: If True and value is an int, interpret as
                microseconds since epoch and convert to datetime.

        """
        if as_datetime and isinstance(value, int):
            return ChromaCollection._EPOCH + timedelta(microseconds=value)
        if isinstance(value, str) and value.startswith(
            ChromaCollection._ESCAPED_PREFIX
        ):
            return value.removeprefix(ChromaCollection._ESCAPED_PREFIX)
        return value

    @staticmethod
    def _distance_to_similarity(distance: float, metric: SimilarityMetric) -> float:
        """Convert a Chroma distance value to a similarity score."""
        if metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            return 1.0 - distance
        # Euclidean
        return 1.0 / (1.0 + distance)

    @staticmethod
    def _build_chroma_filter(expr: FilterExpr) -> dict[str, Any]:
        """Convert a FilterExpr tree to a Chroma ``where`` dictionary."""
        if isinstance(expr, FilterComparison):
            return ChromaCollection._build_chroma_comparison(expr)
        if isinstance(expr, FilterIn):
            return ChromaCollection._build_chroma_in(expr)
        if isinstance(expr, FilterIsNull):
            return {expr.field: {"$eq": ChromaCollection._NULL_SENTINEL}}
        if isinstance(expr, FilterNot):
            return ChromaCollection._build_chroma_not(expr)
        if isinstance(expr, FilterAnd):
            return {
                "$and": [
                    ChromaCollection._build_chroma_filter(expr.left),
                    ChromaCollection._build_chroma_filter(expr.right),
                ]
            }
        if isinstance(expr, FilterOr):
            return {
                "$or": [
                    ChromaCollection._build_chroma_filter(expr.left),
                    ChromaCollection._build_chroma_filter(expr.right),
                ]
            }
        msg = f"Unsupported filter expression type: {type(expr)}"
        raise NotImplementedError(msg)

    @staticmethod
    def _build_chroma_comparison(comp: FilterComparison) -> dict[str, Any]:
        """Convert a single Comparison node into a Chroma where clause."""
        chroma_op = ChromaCollection._OP_MAP.get(comp.op)
        if chroma_op is None:
            msg = f"Unsupported filter operator: {comp.op}"
            raise NotImplementedError(msg)
        serialized_value = ChromaCollection._serialize_property_value(comp.value)
        return {comp.field: {chroma_op: serialized_value}}

    @staticmethod
    def _build_chroma_in(expr: FilterIn) -> dict[str, Any]:
        """Convert an In node into a Chroma where clause."""
        serialized = [
            ChromaCollection._serialize_property_value(v) for v in expr.values
        ]
        return {expr.field: {"$in": serialized}}

    @staticmethod
    def _build_chroma_not(expr: FilterNot) -> dict[str, Any]:
        """Convert a Not node into a Chroma where clause.

        Chroma has no generic ``$not`` operator, so we handle each case
        by algebraic inversion.
        """
        inner = expr.expr
        if isinstance(inner, FilterComparison):
            chroma_op = ChromaCollection._INVERSE_OP_MAP.get(inner.op)
            if chroma_op is None:
                msg = f"Cannot negate operator: {inner.op}"
                raise NotImplementedError(msg)
            serialized_value = ChromaCollection._serialize_property_value(inner.value)
            return {inner.field: {chroma_op: serialized_value}}
        if isinstance(inner, FilterIn):
            serialized = [
                ChromaCollection._serialize_property_value(v) for v in inner.values
            ]
            return {inner.field: {"$nin": serialized}}
        if isinstance(inner, FilterIsNull):
            return {inner.field: {"$ne": ChromaCollection._NULL_SENTINEL}}
        if isinstance(inner, FilterNot):
            # Double negation elimination
            return ChromaCollection._build_chroma_filter(inner.expr)
        if isinstance(inner, FilterAnd):
            # De Morgan: NOT(A AND B) -> (NOT A) OR (NOT B)
            return {
                "$or": [
                    ChromaCollection._build_chroma_not(FilterNot(expr=inner.left)),
                    ChromaCollection._build_chroma_not(FilterNot(expr=inner.right)),
                ]
            }
        if isinstance(inner, FilterOr):
            # De Morgan: NOT(A OR B) -> (NOT A) AND (NOT B)
            return {
                "$and": [
                    ChromaCollection._build_chroma_not(FilterNot(expr=inner.left)),
                    ChromaCollection._build_chroma_not(FilterNot(expr=inner.right)),
                ]
            }
        msg = f"Unsupported filter expression type inside Not: {type(inner)}"
        raise NotImplementedError(msg)
