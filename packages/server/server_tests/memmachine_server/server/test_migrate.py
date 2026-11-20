"""`memmachine-server migrate`: the setup step that creates every configured store's durable resources."""

import re
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import Engine, event

from memmachine_server.common.configuration.log_conf import LogConf
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
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
    partition_registry: local_sql
session_manager:
  database: local_sql
semantic_memory:
  llm_model: openai_model
  embedding_model: openai_embedder
  storage_backend: vector_store
  feature_store: local_sql
  vector_collection: local_vectors
  vector_dimensions: 1536
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
        "sessions",
        "lease_lock",
        "episodestore",
        "segment_store_sg",
        "partition_registry_pt",
        "vector_semantic_feature",
        "semantic_config_category",
    } <= relational
    vectors = _tables(directory / "vectors.db")
    assert {"vector_store_sqlite_pt", "vector_store_sqlite_pd_op"} <= vectors


@pytest.mark.asyncio
async def test_a_rerun_changes_nothing(configuration_path):
    await app.migrate()
    before = _tables(configuration_path.parent / "memmachine.db")

    await app.migrate()

    assert _tables(configuration_path.parent / "memmachine.db") == before


@pytest.mark.asyncio
async def test_stores_start_on_what_migrate_created_without_ddl(configuration_path):
    await app.migrate()
    statements: list[str] = []

    def record(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        resource_manager = ResourceManagerImpl(load_configuration())
        try:
            await resource_manager.get_session_data_manager()
            await resource_manager.get_sql_lock_service()
            await resource_manager.get_episode_storage()
            await resource_manager.get_segment_store("local_sql")
            semantic_manager = await resource_manager.get_semantic_manager()
            await semantic_manager.get_semantic_storage()
            await semantic_manager.get_semantic_config_storage()
        finally:
            await resource_manager.close()
    finally:
        event.remove(Engine, "before_cursor_execute", record)

    assert statements
    assert [
        statement
        for statement in statements
        if re.match(r"\s*(CREATE|ALTER|DROP)\b", statement, re.IGNORECASE)
    ] == []
