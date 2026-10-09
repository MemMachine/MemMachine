from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest
from sqlalchemy import insert

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.common.vector_store import Record, VectorStoreCollectionConfig
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

        record_uuid = await _vector_uuid(storage, feature_id)
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


@pytest.mark.asyncio
async def test_vector_records_carry_no_properties(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    """The feature row is the authority; the vector record holds a vector."""
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
            metadata={"source": "chat"},
        )
        record_uuid = await _vector_uuid(storage, feature_id)
        assert not vector_collection.records[record_uuid].properties

        await storage.update_feature(
            feature_id,
            value="sushi",
            embedding=np.array([0.0, 1.0], dtype=float),
        )
        assert not vector_collection.records[record_uuid].properties
    finally:
        await storage.delete_all()
        await storage.cleanup()


@pytest.mark.asyncio
async def test_an_update_without_an_embedding_leaves_the_vector_record_alone(
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
        record_uuid = await _vector_uuid(storage, feature_id)
        before = vector_collection.records[record_uuid]

        await storage.update_feature(feature_id, value="sushi", tag="meal")

        assert vector_collection.records[record_uuid] == before
        feature = await storage.get_feature(feature_id)
        assert feature is not None
        assert (feature.value, feature.tag) == ("sushi", "meal")
    finally:
        await storage.delete_all()
        await storage.cleanup()


@pytest.mark.asyncio
async def test_an_update_reads_nothing_back_from_the_vector_store(
    sqlalchemy_sqlite_engine,
    vector_collection: InMemoryVectorStoreCollection,
):
    """An update succeeds with the vector record absent, as a backend may not show it yet."""
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
        record_uuid = await _vector_uuid(storage, feature_id)
        del vector_collection.records[record_uuid]

        await storage.update_feature(feature_id, value="sushi")
        feature = await storage.get_feature(feature_id)
        assert feature is not None
        assert feature.value == "sushi"

        await storage.update_feature(
            feature_id, embedding=np.array([0.0, 1.0], dtype=float)
        )
        assert vector_collection.records[record_uuid].vector == [0.0, 1.0]
    finally:
        await storage.delete_all()
        await storage.cleanup()


# More features than SQLite binds in one statement (32,766 parameters): the
# feature store's deletions must not bind every feature they delete.
_MANY_FEATURES = 40_000


@pytest.mark.asyncio
@pytest.mark.parametrize("deletion", ["delete_all", "delete_feature_set"])
async def test_deleting_more_features_than_sqlite_binds_removes_every_vector_record(
    sqlalchemy_engine,
    vector_collection: InMemoryVectorStoreCollection,
    deletion: str,
):
    storage = VectorStoreSemanticStorage(sqlalchemy_engine, vector_collection)
    await storage.startup()
    try:
        vector_uuids = [uuid4() for _ in range(_MANY_FEATURES)]
        async with storage._create_session() as session:
            await session.execute(
                insert(VectorSemanticFeature),
                [
                    {
                        "vector_uuid": vector_uuid,
                        "set_id": "user",
                        "semantic_category_id": "profile",
                        "tag_id": "tag",
                        "feature": "feature",
                        "value": "value",
                    }
                    for vector_uuid in vector_uuids
                ],
            )
            await session.commit()
        await vector_collection.upsert(
            records=[
                Record(uuid=vector_uuid, vector=[1.0, 0.0])
                for vector_uuid in vector_uuids
            ]
        )

        if deletion == "delete_all":
            await storage.delete_all()
        else:
            await storage.delete_feature_set(
                filter_expr=parse_filter("set_id IN (user)")
            )

        assert vector_collection.records == {}
    finally:
        await storage.delete_all()
        await storage.cleanup()
