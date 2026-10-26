"""`memmachine-server migrate`: the setup step that creates every configured store's durable resources."""

import sqlite3
import sys
from pathlib import Path

import pytest

from memmachine_server.common.configuration.log_conf import LogConf
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    EVENT_MEMORY_VECTOR_STORE_NAME,
    event_backend_store_schema,
)
from memmachine_server.server import app
from memmachine_server.server.api_v2.mcp import load_configuration

_CONFIGURATION = """
logging:
  level: info
  path: {directory}/memmachine.log
episode_store:
  database: local_sql
episodic_memory:
  long_term_memory:
    backend: event
    embedder: openai_embedder
    vector_store: local_vectors
    segment_store: local_sql
session_manager:
  database: local_sql
semantic_memory:
  llm_model: openai_model
  embedding_model: openai_embedder
  storage_backend: vector_store
  feature_store: local_sql
  vector_collection: local_vectors
  config_database: local_sql
prompt:
  default_project_categories:
  - profile_prompt
resources:
  databases:
    local_sql:
      provider: sqlite
      config:
        path: {directory}/memmachine.db
    local_vectors:
      provider: sqlite_vector_store
      config:
        path: {directory}/vectors.db
  embedders:
    openai_embedder:
      provider: openai
      config:
        model: text-embedding-3-small
        api_key: not-used
        dimensions: 1536
  language_models:
    openai_model:
      provider: openai-responses
      config:
        model: gpt-4o-mini
        api_key: not-used
"""


@pytest.fixture
def configuration_path(tmp_path, monkeypatch) -> Path:
    """A configuration of SQLite stores only, the shape the installer writes."""
    path = tmp_path / "cfg.yml"
    path.write_text(_CONFIGURATION.format(directory=tmp_path))
    monkeypatch.setenv("MEMORY_CONFIG", str(path))
    # Loading a configuration applies its logging to the whole process.
    monkeypatch.setattr(LogConf, "apply", lambda _self: None)
    return path


def _tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return {name for (name,) in rows}


def test_migrate_command_creates_every_selectable_store(
    configuration_path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["memmachine-server", "migrate"])

    app.main()

    directory = configuration_path.parent
    relational = _tables(directory / "memmachine.db")
    assert {
        "alembic_version_session",
        "alembic_version_lease_lock",
        "alembic_version_episode_store",
        "alembic_version_segment_store",
        "sessions",
        "lease_lock",
        "episodestore",
        "segment_store_pt",
        "vector_semantic_feature",
        "semantic_config_category",
    } <= relational
    assert not {table for table in relational if table.startswith("partition_registry")}
    vectors = _tables(directory / "vectors.db")
    assert {"alembic_version_sqlite_vector_store", "vector_store_sqlite_pt"} <= vectors


@pytest.mark.asyncio
async def test_migrate_covers_disabled_semantic_memory_storage(
    configuration_path,
):
    configuration_path.write_text(
        configuration_path.read_text().replace(
            "semantic_memory:\n", "semantic_memory:\n  enabled: false\n"
        )
    )

    await app.migrate()

    assert {"vector_semantic_feature", "semantic_config_category"} <= _tables(
        configuration_path.parent / "memmachine.db"
    )


@pytest.mark.asyncio
async def test_a_rerun_changes_nothing(configuration_path):
    await app.migrate()
    before = _tables(configuration_path.parent / "memmachine.db")

    await app.migrate()

    assert _tables(configuration_path.parent / "memmachine.db") == before


@pytest.mark.asyncio
async def test_stores_start_on_what_migrate_created(configuration_path):
    await app.migrate()

    resource_manager = ResourceManagerImpl(load_configuration())
    try:
        await resource_manager.get_session_data_manager()
        await resource_manager.get_sql_lock_service()
        await resource_manager.get_episode_storage()
        await resource_manager.get_segment_store("local_sql")
        embedder = await resource_manager.get_embedder("openai_embedder")
        await resource_manager.get_vector_store(
            "local_vectors",
            vector_store_name=EVENT_MEMORY_VECTOR_STORE_NAME,
            store_schema=event_backend_store_schema(embedder),
        )
        semantic_manager = await resource_manager.get_semantic_manager()
        await semantic_manager.get_semantic_storage()
        await semantic_manager.get_semantic_config_storage()
    finally:
        await resource_manager.close()
