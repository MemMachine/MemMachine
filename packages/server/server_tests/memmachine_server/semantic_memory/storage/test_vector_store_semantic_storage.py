from __future__ import annotations

import numpy as np
import pytest

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.common.vector_store import VectorStoreCollectionConfig
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorStoreSemanticStorage,
    feature_vector_uuid,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_collection import (
    InMemoryVectorStoreCollection,
)


@pytest.fixture
def vector_collection() -> InMemoryVectorStoreCollection:
    return InMemoryVectorStoreCollection(
        VectorStoreCollectionConfig(
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties_schema={
                "set_id": str,
                "category": str,
                "tag": str,
                "feature_name": str,
            },
        )
    )


@pytest.mark.asyncio
async def test_add_update_delete_feature_keeps_vector_collection_in_sync(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, vector_collection)
    await storage.startup()
    try:
        feature_id = await storage.add_feature(
            set_id="user",
            category_name="default",
            feature="likes",
            value="pizza",
            tag="food",
            embedding=np.array([1.0, 0.0], dtype=float),
        )

        record_uuid = feature_vector_uuid(feature_id)
        assert record_uuid in vector_collection.records
        assert vector_collection.records[record_uuid].vector == [1.0, 0.0]

        await storage.update_feature(
            feature_id,
            value="sushi",
            embedding=np.array([0.0, 1.0], dtype=float),
        )
        assert vector_collection.records[record_uuid].vector == [0.0, 1.0]

        feature = await storage.get_feature(feature_id)
        assert feature is not None
        assert feature.value == "sushi"

        await storage.delete_features([feature_id])
        assert record_uuid not in vector_collection.records
        assert await storage.get_feature(feature_id) is None
    finally:
        await storage.delete_all()
        await storage.cleanup()


@pytest.mark.asyncio
async def test_vector_search_returns_relational_features_in_similarity_order(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, vector_collection)
    await storage.startup()
    try:
        await storage.add_feature(
            set_id="user",
            category_name="default",
            feature="likes",
            value="pizza",
            tag="food",
            embedding=np.array([1.0, 0.0], dtype=float),
        )
        await storage.add_feature(
            set_id="user",
            category_name="default",
            feature="likes",
            value="sushi",
            tag="food",
            embedding=np.array([0.0, 1.0], dtype=float),
        )

        results = [
            feature
            async for feature in storage.get_feature_set(
                filter_expr=parse_filter("set_id IN (user)"),
                vector_search_opts=SemanticStorage.VectorSearchOpts(
                    query_embedding=np.array([0.9, 0.1], dtype=float),
                ),
            )
        ]

        assert [feature.value for feature in results] == ["pizza", "sushi"]
    finally:
        await storage.delete_all()
        await storage.cleanup()
