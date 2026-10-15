from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.configuration import (
    PromptConf,
    SemanticMemoryConf,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.errors import ResourceNotReadyError
from memmachine_server.common.filter.filter_parser import In
from memmachine_server.common.resource_manager.semantic_manager import (
    _VECTOR_STORE_COLLECTION_NAME,
    _VECTOR_STORE_NAMESPACE,
    SemanticResourceManager,
    build_semantic_deletion_storage,
)
from memmachine_server.common.vector_store import VectorStoreCollectionPendingError
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreParams,
)
from memmachine_server.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
)
from memmachine_server.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorStoreSemanticStorage,
)

SET_ID = "mem_project_set_org_o_project_p_0_x__"


requires_sqlite_extensions = pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason="sqlite3 built without extension loading support",
)


def _vector_store_conf(**overrides) -> SemanticMemoryConf:
    return SemanticMemoryConf(
        enabled=False,
        storage_backend=SemanticMemoryStorageBackend.VECTOR_STORE,
        feature_store="semantic_db",
        vector_collection="semantic_vectors",
        **overrides,
    )


def _resources(engine: AsyncEngine, vector_store) -> MagicMock:
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(return_value=engine)
    resources.get_vector_store = AsyncMock(return_value=vector_store)
    resources.get_neo4j_driver = AsyncMock(
        side_effect=AssertionError("vector_store deletion must not open neo4j")
    )
    resources.get_embedder = AsyncMock(
        side_effect=AssertionError("deletion must not resolve an embedder")
    )
    resources.get_language_model = AsyncMock(
        side_effect=AssertionError("deletion must not resolve a language model")
    )
    return resources


async def _has_table(engine: AsyncEngine, name: str) -> bool:
    async with engine.connect() as conn:
        return await conn.run_sync(lambda sync: inspect(sync).has_table(name))


@pytest_asyncio.fixture
async def sqlite_vec_store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'vectors.db'}")
    store = SQLiteVecVectorStore(SQLiteVecVectorStoreParams(engine=engine))
    await store.startup()
    yield store
    await store.shutdown()
    await engine.dispose()


async def _initialize_serving_storage(engine, vector_store):
    manager = SemanticResourceManager(
        semantic_conf=_vector_store_conf(vector_dimensions=2),
        prompt_conf=PromptConf(),
        resource_manager=_resources(engine, vector_store),
        episode_storage=MagicMock(),
    )
    return await manager.get_semantic_storage()


@requires_sqlite_extensions
@pytest.mark.asyncio
async def test_never_initialized_vector_store_returns_none_and_creates_nothing(
    sqlalchemy_sqlite_engine, sqlite_vec_store
):
    storage = await build_semantic_deletion_storage(
        _vector_store_conf(), _resources(sqlalchemy_sqlite_engine, sqlite_vec_store)
    )

    assert storage is None
    assert not await _has_table(sqlalchemy_sqlite_engine, "vector_semantic_feature")
    assert (
        await sqlite_vec_store.open_collection(
            namespace=_VECTOR_STORE_NAMESPACE, name=_VECTOR_STORE_COLLECTION_NAME
        )
        is None
    )


@requires_sqlite_extensions
@pytest.mark.asyncio
async def test_deletes_rows_and_vectors_the_serving_storage_wrote(
    sqlalchemy_sqlite_engine, sqlite_vec_store
):
    serving = await _initialize_serving_storage(
        sqlalchemy_sqlite_engine, sqlite_vec_store
    )
    feature_id = await serving.add_feature(
        set_id=SET_ID,
        category_name="profile",
        feature="topic",
        value="pizza",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )

    storage = await build_semantic_deletion_storage(
        _vector_store_conf(), _resources(sqlalchemy_sqlite_engine, sqlite_vec_store)
    )
    assert storage is not None
    await storage.delete_feature_set(filter_expr=In(field="set_id", values=[SET_ID]))

    assert await serving.get_feature(feature_id) is None
    collection = await sqlite_vec_store.open_collection(
        namespace=_VECTOR_STORE_NAMESPACE, name=_VECTOR_STORE_COLLECTION_NAME
    )
    assert collection is not None
    [result] = await collection.query(query_vectors=[[1.0, 0.0]], limit=10)
    assert result.matches == []


@requires_sqlite_extensions
@pytest.mark.asyncio
async def test_feature_table_without_collection_raises(
    sqlalchemy_sqlite_engine, sqlite_vec_store
):
    await _initialize_serving_storage(sqlalchemy_sqlite_engine, sqlite_vec_store)
    await sqlite_vec_store.delete_collection(
        namespace=_VECTOR_STORE_NAMESPACE, name=_VECTOR_STORE_COLLECTION_NAME
    )

    with pytest.raises(ResourceNotReadyError, match="collection"):
        await build_semantic_deletion_storage(
            _vector_store_conf(),
            _resources(sqlalchemy_sqlite_engine, sqlite_vec_store),
        )


@pytest.mark.asyncio
async def test_pending_collection_error_propagates(sqlalchemy_sqlite_engine):
    await VectorStoreSemanticStorage(sqlalchemy_sqlite_engine, MagicMock()).startup()
    pending_store = MagicMock()
    pending_store.open_collection = AsyncMock(
        side_effect=VectorStoreCollectionPendingError(
            _VECTOR_STORE_NAMESPACE,
            _VECTOR_STORE_COLLECTION_NAME,
            datetime.now(UTC),
            VectorStoreCollectionConfig(vector_dimensions=2),
        )
    )

    with pytest.raises(VectorStoreCollectionPendingError):
        await build_semantic_deletion_storage(
            _vector_store_conf(),
            _resources(sqlalchemy_sqlite_engine, pending_store),
        )


@pytest.mark.asyncio
async def test_auto_backend_falls_back_to_neo4j_without_startup():
    driver = MagicMock()
    driver.execute_query = AsyncMock()
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(
        side_effect=ValueError("SQL config 'graph' not found.")
    )
    resources.get_neo4j_driver = AsyncMock(return_value=driver)

    storage = await build_semantic_deletion_storage(
        SemanticMemoryConf(enabled=False, database="graph"), resources
    )

    assert isinstance(storage, Neo4jSemanticStorage)
    driver.execute_query.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_never_migrated_returns_none(sqlalchemy_pg_engine):
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(return_value=sqlalchemy_pg_engine)

    storage = await build_semantic_deletion_storage(
        SemanticMemoryConf(
            enabled=False,
            storage_backend=SemanticMemoryStorageBackend.PGVECTOR,
            database="pg",
        ),
        resources,
    )

    assert storage is None
    assert not await _has_table(sqlalchemy_pg_engine, "feature")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_deletes_features(sqlalchemy_pg_engine):
    serving = SqlAlchemyPgVectorSemanticStorage(sqlalchemy_pg_engine)
    await serving.startup()
    feature_id = await serving.add_feature(
        set_id=SET_ID,
        category_name="profile",
        feature="topic",
        value="pizza",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(return_value=sqlalchemy_pg_engine)

    storage = await build_semantic_deletion_storage(
        SemanticMemoryConf(
            enabled=False,
            storage_backend=SemanticMemoryStorageBackend.PGVECTOR,
            database="pg",
        ),
        resources,
    )
    assert storage is not None
    await storage.delete_feature_set(filter_expr=In(field="set_id", values=[SET_ID]))

    assert await serving.get_feature(feature_id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_neo4j_deletes_features(neo4j_semantic_storage, neo4j_driver):
    feature_id = await neo4j_semantic_storage.add_feature(
        set_id=SET_ID,
        category_name="profile",
        feature="topic",
        value="pizza",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )
    resources = MagicMock()
    resources.get_neo4j_driver = AsyncMock(return_value=neo4j_driver)

    storage = await build_semantic_deletion_storage(
        SemanticMemoryConf(
            enabled=False,
            storage_backend=SemanticMemoryStorageBackend.NEO4J,
            database="graph",
        ),
        resources,
    )
    assert storage is not None
    await storage.delete_feature_set(filter_expr=In(field="set_id", values=[SET_ID]))

    assert await neo4j_semantic_storage.get_feature(feature_id) is None


def test_built_semantic_storage_returns_the_service_storage_without_building():
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(
        side_effect=AssertionError("must not build storage")
    )
    manager = SemanticResourceManager(
        semantic_conf=_vector_store_conf(vector_dimensions=2),
        prompt_conf=PromptConf(),
        resource_manager=resources,
        episode_storage=MagicMock(),
    )

    assert manager.built_semantic_storage() is None
    resources.get_sql_engine.assert_not_called()

    service_storage = MagicMock()
    manager._semantic_service = MagicMock(semantic_storage=service_storage)
    assert manager.built_semantic_storage() is service_storage
