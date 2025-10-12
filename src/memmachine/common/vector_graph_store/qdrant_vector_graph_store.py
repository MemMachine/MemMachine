"""
This module provides an implementation of a vector graph store
using Qdrant as the backend database.
"""

from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from memmachine.common.embedder import SimilarityMetric

from .data_types import Edge, Node, Property
from .vector_graph_store import VectorGraphStore


class QdrantVectorGraphStoreConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str | None = Field(None, description="Qdrant connection URL")
    api_key: SecretStr | None = Field(None, description="Qdrant API key")
    timeout: float = Field(60.0, description="Request timeout in seconds", gt=0)
    vector_dimension: int = Field(
        384, description="Vector dimension for embeddings", gt=0
    )
    client: AsyncQdrantClient | None = Field(None, description="Custom Qdrant client")

    @model_validator(mode="after")
    def check_client_or_url(self):
        if self.client is None and self.url is None:
            raise ValueError("Either Qdrant client or URL must be provided")
        return self


class QdrantVectorGraphStore(VectorGraphStore):
    def __init__(self, config: QdrantVectorGraphStoreConfig):
        super().__init__()

        if config.client is None:
            url = cast(str, config.url)
            api_key = config.api_key.get_secret_value() if config.api_key else None
            self._client = AsyncQdrantClient(
                url=url, api_key=api_key, timeout=config.timeout
            )
        else:
            self._client = config.client

        self._vector_dimension = config.vector_dimension

        self._nodes_collection = "nodes"
        self._edges_collection = "edges"

    async def add_nodes(self, nodes: Collection[Node]):
        if not nodes:
            return

        await self._ensure_collection_exists(
            self._nodes_collection, self._vector_dimension
        )

        points = []
        for node in nodes:
            embedding = self._extract_embedding(node.properties)

            point = PointStruct(
                id=str(node.uuid),
                vector=embedding,
                payload={
                    "uuid": str(node.uuid),
                    "labels": list(node.labels),
                    "properties": self._serialize_properties(node.properties),
                },
            )
            points.append(point)

        await self._client.upsert(collection_name=self._nodes_collection, points=points)

    async def add_edges(self, edges: Collection[Edge]):
        if not edges:
            return

        await self._ensure_collection_exists(self._edges_collection, 1)

        points = []
        for edge in edges:
            point = PointStruct(
                id=str(edge.uuid),
                vector={},
                payload={
                    "uuid": str(edge.uuid),
                    "source_uuid": str(edge.source_uuid),
                    "target_uuid": str(edge.target_uuid),
                    "relation": edge.relation,
                    "properties": self._serialize_properties(edge.properties),
                },
            )
            points.append(point)

        await self._client.upsert(collection_name=self._edges_collection, points=points)

    async def search_similar_nodes(
        self,
        query_embedding: list[float],
        embedding_property_name: str,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        if not await self._client.collection_exists(self._nodes_collection):
            return []

        filter_conditions = []

        if required_labels:
            for label in required_labels:
                filter_conditions.append(
                    FieldCondition(key="labels", match=MatchValue(value=label))
                )

        for prop_name, prop_value in required_properties.items():
            if prop_value is not None:
                filter_conditions.append(
                    FieldCondition(
                        key=f"properties.{prop_name}",
                        match=MatchValue(value=prop_value),
                    )
                )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        search_response = await self._client.query_points(
            collection_name=self._nodes_collection,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit or 100,
            with_payload=True,
        )

        nodes = []
        for point in search_response.points:
            node = self._point_to_node(point)
            if node:
                node.score = point.score
                nodes.append(node)

        return nodes

    async def search_related_nodes(
        self,
        node_uuid: UUID,
        allowed_relations: Collection[str] | None = None,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        if not await self._client.collection_exists(
            self._edges_collection
        ) or not await self._client.collection_exists(self._nodes_collection):
            return []

        edge_filter_conditions = []

        if find_sources:
            edge_filter_conditions.append(
                FieldCondition(
                    key="target_uuid", match=MatchValue(value=str(node_uuid))
                )
            )

        if find_targets:
            edge_filter_conditions.append(
                FieldCondition(
                    key="source_uuid", match=MatchValue(value=str(node_uuid))
                )
            )

        if allowed_relations:
            for relation in allowed_relations:
                edge_filter_conditions.append(
                    FieldCondition(key="relation", match=MatchValue(value=relation))
                )

        edge_filter = (
            Filter(should=edge_filter_conditions) if edge_filter_conditions else None
        )

        edge_results, _offset = await self._client.scroll(
            collection_name=self._edges_collection,
            scroll_filter=edge_filter,
            # TODO (Anush008): Paginate with _offset to retrieve all points
            limit=1000,
            with_payload=True,
        )

        related_uuids = set()
        for point in edge_results:
            payload = point.payload
            if payload:
                if find_sources and payload.get("target_uuid") == str(node_uuid):
                    related_uuids.add(payload.get("source_uuid"))
                if find_targets and payload.get("source_uuid") == str(node_uuid):
                    related_uuids.add(payload.get("target_uuid"))

        if not related_uuids:
            return []

        retrieved_points = await self._client.retrieve(
            collection_name=self._nodes_collection,
            ids=list(related_uuids),
            with_payload=True,
        )

        nodes = []
        for point in retrieved_points:
            node = self._point_to_node(point)
            if node:
                if required_labels and not required_labels.issubset(node.labels):
                    continue
                if required_properties:
                    if not all(
                        node.properties.get(k) == v
                        for k, v in required_properties.items()
                    ):
                        continue
                nodes.append(node)

        if limit:
            nodes = nodes[:limit]

        return nodes

    async def search_directional_nodes(
        self,
        by_property: str,
        start_at_value: Any | None = None,
        include_equal_start_at_value: bool = False,
        order_ascending: bool = True,
        limit: int | None = 1,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        all_nodes = await self.search_matching_nodes(
            limit=None,
            required_labels=required_labels,
            required_properties=required_properties,
            include_missing_properties=include_missing_properties,
        )

        filtered_nodes = []
        for node in all_nodes:
            if by_property in node.properties:
                prop_value = node.properties[by_property]
                if start_at_value is not None:
                    if include_equal_start_at_value:
                        if order_ascending and prop_value >= start_at_value:
                            filtered_nodes.append(node)
                        elif not order_ascending and prop_value <= start_at_value:
                            filtered_nodes.append(node)
                    else:
                        if order_ascending and prop_value > start_at_value:
                            filtered_nodes.append(node)
                        elif not order_ascending and prop_value < start_at_value:
                            filtered_nodes.append(node)
                else:
                    filtered_nodes.append(node)

        filtered_nodes.sort(
            key=lambda x: x.properties.get(by_property, 0), reverse=not order_ascending
        )

        return filtered_nodes[:limit] if limit else filtered_nodes

    async def search_matching_nodes(
        self,
        limit: int | None = None,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        if not await self._client.collection_exists(self._nodes_collection):
            return []

        filter_conditions = []

        if required_labels:
            for label in required_labels:
                filter_conditions.append(
                    FieldCondition(key="labels", match=MatchValue(value=label))
                )

        for prop_name, prop_value in required_properties.items():
            if prop_value is not None:
                filter_conditions.append(
                    FieldCondition(
                        key=f"properties.{prop_name}",
                        match=MatchValue(value=prop_value),
                    )
                )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        scroll_result, _offset = await self._client.scroll(
            collection_name=self._nodes_collection,
            scroll_filter=query_filter,
            limit=limit or 100,
            with_payload=True,
        )

        nodes = []
        for point in scroll_result:
            node = self._point_to_node(point)
            if node:
                nodes.append(node)

        return nodes

    async def delete_nodes(self, node_uuids: Collection[UUID]):
        if not node_uuids:
            return

        if not await self._client.collection_exists(self._nodes_collection):
            return

        await self._client.delete(
            collection_name=self._nodes_collection,
            points_selector=PointIdsList(points=[str(uuid) for uuid in node_uuids]),
        )

    async def clear_data(self):
        if await self._client.collection_exists(self._nodes_collection):
            await self._client.delete_collection(self._nodes_collection)
        if await self._client.collection_exists(self._edges_collection):
            await self._client.delete_collection(self._edges_collection)

    async def close(self):
        if hasattr(self._client, "close"):
            await self._client.close()

    async def _ensure_collection_exists(self, collection_name: str, vector_dim: int):
        if not await self._client.collection_exists(collection_name):
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

    def _extract_embedding(self, properties: dict[str, Property]) -> list[float]:
        for prop_name, prop_value in properties.items():
            if isinstance(prop_value, list) and all(
                isinstance(x, (int, float)) for x in prop_value
            ):
                if len(prop_value) == self._vector_dimension:
                    return prop_value
        return [0.0] * self._vector_dimension

    def _serialize_properties(self, properties: dict[str, Property]) -> dict[str, Any]:
        serialized = {}
        for key, value in properties.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, UUID):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    def _deserialize_properties(
        self, properties: dict[str, Any]
    ) -> dict[str, Property]:
        deserialized = {}
        for key, value in properties.items():
            if isinstance(value, str):
                try:
                    deserialized[key] = datetime.fromisoformat(value)
                    continue
                except (ValueError, TypeError):
                    pass

                try:
                    deserialized[key] = UUID(value)
                    continue
                except (ValueError, TypeError):
                    pass

            deserialized[key] = value
        return deserialized

    def _point_to_node(self, point) -> Node | None:
        if not point.payload:
            return None

        uuid_str = point.payload.get("uuid")
        if not uuid_str:
            return None
        labels = set(point.payload.get("labels", []))
        properties = self._deserialize_properties(point.payload.get("properties", {}))
        return Node(uuid=UUID(uuid_str), labels=labels, properties=properties)
