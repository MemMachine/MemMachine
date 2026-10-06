from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import numpy as np
import pytest
from sqlalchemy import text

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
@pytest.mark.parametrize("legacy_column", ["batch_position", "sequence_num"])
async def test_startup_removes_existing_history_position(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
    legacy_column: str,
):
    history_id = UUID("550e8400-e29b-41d4-a716-446655440019")
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE vector_semantic_set_ingested_history ("
                "set_id VARCHAR NOT NULL, history_id VARCHAR NOT NULL, "
                "created_at DATETIME, episode_created_at DATETIME, "
                f"{legacy_column} INTEGER NOT NULL DEFAULT 0, "
                "ingested BOOLEAN NOT NULL, "
                "PRIMARY KEY (set_id, history_id))"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO vector_semantic_set_ingested_history "
                f"(set_id, history_id, {legacy_column}, ingested) "
                "VALUES ('user', :history_id, 42, 0)"
            ),
            {"history_id": str(history_id)},
        )

    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, vector_collection)
    await storage.startup()
    assert [item async for item in storage.get_history_messages(set_ids=["user"])] == [
        history_id
    ]
    async with sqlalchemy_sqlite_engine.begin() as conn:
        result = await conn.execute(
            text("PRAGMA table_info(vector_semantic_set_ingested_history)"),
        )
        assert legacy_column not in {row[1] for row in result}


@pytest.mark.asyncio
async def test_older_than_compares_instants_not_wall_clocks(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    """A non-UTC-offset bound names an instant on SQLite, not a wall clock.

    created_at is server-generated UTC; the bound below is an instant
    BEFORE the row's creation whose +08:00 wall clock lies far after it,
    so comparing wall clocks wrongly selects the row.
    """
    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, vector_collection)
    await storage.startup()
    try:
        await storage.add_history_to_set(
            set_id="user", history_id=UUID("550e8400-e29b-41d4-a716-446655440001")
        )

        cutoff = (datetime.now(UTC) - timedelta(minutes=5)).astimezone(
            timezone(timedelta(hours=8))
        )
        set_ids = {sid async for sid in storage.get_history_set_ids(older_than=cutoff)}
        assert set_ids == set()
    finally:
        await storage.delete_all()
        await storage.cleanup()


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
