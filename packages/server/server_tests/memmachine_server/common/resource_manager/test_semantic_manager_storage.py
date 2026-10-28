from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.configuration import (
    PromptConf,
    SemanticMemoryConf,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.filter.filter_parser import In
from memmachine_server.common.resource_manager.semantic_manager import (
    SemanticResourceManager,
)
from memmachine_server.semantic_memory.storage import sqlalchemy_pgvector_semantic
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorStoreSemanticStorage,
)

SET_ID = "mem_project_set_org_o_project_p_0_x__"


def _manager(conf: SemanticMemoryConf, resources: MagicMock) -> SemanticResourceManager:
    return SemanticResourceManager(
        semantic_conf=conf,
        prompt_conf=PromptConf(),
        resource_manager=resources,
        episode_storage=MagicMock(),
    )


def _resources_without_models(engine: AsyncEngine) -> MagicMock:
    vector_store = MagicMock()
    vector_store.open_or_create_partition = AsyncMock(return_value=MagicMock())
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(return_value=engine)
    resources.get_vector_store = AsyncMock(return_value=vector_store)
    resources.get_embedder = AsyncMock(
        side_effect=AssertionError("storage must not resolve an embedder")
    )
    resources.get_language_model = AsyncMock(
        side_effect=AssertionError("storage must not resolve a language model")
    )
    return resources


_DISABLED_VECTOR_STORE_CONF = SemanticMemoryConf(
    enabled=False,
    storage_backend=SemanticMemoryStorageBackend.VECTOR_STORE,
    feature_store="semantic_db",
    vector_collection="semantic_vectors",
    vector_dimensions=2,
)


@pytest.mark.asyncio
async def test_storage_builds_while_disabled_without_models(sqlalchemy_sqlite_engine):
    resources = _resources_without_models(sqlalchemy_sqlite_engine)
    manager = _manager(_DISABLED_VECTOR_STORE_CONF, resources)

    storage = await manager.get_semantic_storage()

    assert isinstance(storage, VectorStoreSemanticStorage)
    resources.get_embedder.assert_not_awaited()
    resources.get_language_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_is_built_once(sqlalchemy_sqlite_engine):
    resources = _resources_without_models(sqlalchemy_sqlite_engine)
    manager = _manager(_DISABLED_VECTOR_STORE_CONF, resources)

    first, second = await asyncio.gather(
        manager.get_semantic_storage(), manager.get_semantic_storage()
    )

    assert first is second
    assert await manager.get_semantic_storage() is first
    resources.get_vector_store.assert_awaited_once()


def _alembic_config() -> Config:
    script_location = Path(sqlalchemy_pgvector_semantic.__file__).parent / "alembic_pg"
    config = Config()
    config.set_main_option("script_location", str(script_location))
    config.set_main_option("version_locations", str(script_location / "versions"))
    config.set_main_option("path_separator", "os")
    return config


async def _migrate_pgvector_to(engine: AsyncEngine, revision: str) -> None:
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")

        def upgrade(sync_conn):
            config = _alembic_config()
            config.set_main_option("sqlalchemy.url", str(sync_conn.engine.url))
            config.attributes["connection"] = sync_conn
            command.upgrade(config, revision)

        await conn.run_sync(upgrade)


async def _current_revision(engine: AsyncEngine) -> str | None:
    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda sync_conn: MigrationContext.configure(
                sync_conn
            ).get_current_revision()
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "revision",
    [
        pytest.param("001", id="legacy-prof-table"),
        pytest.param("d1a9df11343b", id="legacy-set-ids"),
    ],
)
async def test_pgvector_store_before_data_migrations_is_migrated_and_deleted_from(
    sqlalchemy_pg_engine, revision
):
    await _migrate_pgvector_to(sqlalchemy_pg_engine, revision)
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(return_value=sqlalchemy_pg_engine)
    manager = _manager(
        SemanticMemoryConf(
            enabled=False,
            storage_backend=SemanticMemoryStorageBackend.PGVECTOR,
            database="pg",
        ),
        resources,
    )

    storage = await manager.get_semantic_storage()

    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    assert await _current_revision(sqlalchemy_pg_engine) == head
    feature_id = await storage.add_feature(
        set_id=SET_ID,
        category_name="profile",
        feature="topic",
        value="pizza",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
    )
    await storage.delete_feature_set(filter_expr=In(field="set_id", values=[SET_ID]))
    assert await storage.get_feature(feature_id) is None
