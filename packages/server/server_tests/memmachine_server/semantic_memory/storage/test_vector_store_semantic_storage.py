from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import numpy as np
import pytest

from memmachine_server.common.filter.filter_parser import (
    parse_filter,
)
from memmachine_server.common.vector_store import VectorStoreCollectionConfig
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorSemanticFeature,
    VectorStoreSemanticStorage,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_collection import (
    InMemoryVectorStoreCollection,
)


@pytest.fixture
def vector_collection() -> InMemoryVectorStoreCollection:
    return InMemoryVectorStoreCollection(
        VectorStoreCollectionConfig(vector_dimensions=2),
        {"set_id": str, "category": str, "tag": str, "feature_name": str},
    )


async def _vector_uuid(storage: VectorStoreSemanticStorage, feature_id) -> UUID:
    """The uuid of the vector record a feature owns, read off its own row."""
    async with storage._create_session() as session:
        row = await session.get(VectorSemanticFeature, int(feature_id))
    assert row is not None
    return row.vector_uuid


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
        await storage.add_history_to_set(set_id="user", history_id="ep-1")

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

        # The feature row owns the mapping to its vector record.
        async with storage._create_session() as session:
            row = await session.get(VectorSemanticFeature, int(feature_id))
        assert row is not None
        record_uuid = row.vector_uuid
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
async def test_each_feature_owns_a_distinct_vector_record(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    """`vector_uuid` is minted per feature, so no two features can collide."""
    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, vector_collection)
    await storage.startup()
    try:
        feature_ids = [
            await storage.add_feature(
                set_id="user",
                category_name="default",
                feature="likes",
                value=value,
                tag="food",
                embedding=np.array([1.0, 0.0], dtype=float),
            )
            # Same set, category, feature name, tag and embedding: only the
            # value differs, so anything derived from the others would collide.
            for value in ("pizza", "sushi")
        ]

        vector_uuids = [
            await _vector_uuid(storage, feature_id) for feature_id in feature_ids
        ]

        assert len(set(vector_uuids)) == 2
        assert all(
            vector_uuid in vector_collection.records for vector_uuid in vector_uuids
        )
    finally:
        await storage.delete_all()
        await storage.cleanup()


@pytest.mark.asyncio
async def test_vector_records_carry_no_properties(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    """The feature row is the authority; the vector record holds only a vector."""
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
        vector_uuid = await _vector_uuid(storage, feature_id)
        assert not vector_collection.records[vector_uuid].properties

        # An update rewrites the vector and still writes no properties.
        await storage.update_feature(
            feature_id,
            value="sushi",
            embedding=np.array([0.0, 1.0], dtype=float),
        )
        assert not vector_collection.records[vector_uuid].properties
    finally:
        await storage.delete_all()
        await storage.cleanup()


@pytest.mark.asyncio
async def test_delete_feature_set_removes_the_vector_records(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    """The uuids must be read before the rows go, or the vectors are orphaned."""
    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, vector_collection)
    await storage.startup()
    try:
        dropped_id = await storage.add_feature(
            set_id="dropped",
            category_name="default",
            feature="likes",
            value="pizza",
            tag="food",
            embedding=np.array([1.0, 0.0], dtype=float),
        )
        kept_id = await storage.add_feature(
            set_id="kept",
            category_name="default",
            feature="likes",
            value="sushi",
            tag="food",
            embedding=np.array([0.0, 1.0], dtype=float),
        )
        dropped_uuid = await _vector_uuid(storage, dropped_id)
        kept_uuid = await _vector_uuid(storage, kept_id)

        await storage.delete_feature_set(
            filter_expr=parse_filter("set_id IN (dropped)")
        )

        assert dropped_uuid not in vector_collection.records
        assert kept_uuid in vector_collection.records
    finally:
        await storage.delete_all()
        await storage.cleanup()


@pytest.mark.asyncio
async def test_delete_all_removes_the_vector_records(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    storage = VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, vector_collection)
    await storage.startup()
    try:
        for value in ("pizza", "sushi"):
            await storage.add_feature(
                set_id="user",
                category_name="default",
                feature="likes",
                value=value,
                tag="food",
                embedding=np.array([1.0, 0.0], dtype=float),
            )
        assert len(vector_collection.records) == 2

        await storage.delete_all()

        assert vector_collection.records == {}
    finally:
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
