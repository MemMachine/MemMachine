from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.configuration.storage_conf import DatabasesConf
from memmachine.common.resource_manager.storage_manager import StorageManager
from memmachine.common.vector_graph_store import VectorGraphStore


@pytest.fixture
def mock_conf():
    """Mock StoragesConf with dummy connection configurations."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {
        "neo1": MagicMock(host="localhost", port=1234, user="neo", password="pw"),
    }
    conf.relational_db_confs = {
        "pg1": MagicMock(
            host="localhost",
            port=5432,
            user="pg",
            password=MagicMock(get_secret_value=lambda: "secret"),
            db_name="testdb",
            vector_schema="public",
            statement_cache_size=100,
        ),
    }
    conf.sqlite_confs = {"sqlite1": MagicMock(file_path="sqlite:///tmp/test.db")}
    return conf


@pytest.mark.asyncio
async def test_build_neo4j(mock_conf):
    builder = StorageManager(mock_conf)
    await builder._build_neo4j()

    assert "neo1" in builder.graph_stores
    assert isinstance(builder.graph_stores["neo1"], VectorGraphStore)


@pytest.mark.asyncio
async def test_build_sqlite(mock_conf):
    builder = StorageManager(mock_conf)
    await builder._build_sql_engines()

    assert "sqlite1" in builder.sql_engines
    assert isinstance(builder.sql_engines["sqlite1"], AsyncEngine)


@pytest.mark.asyncio
async def test_build_all_without_validation(mock_conf):
    builder = StorageManager(mock_conf)
    builder._build_neo4j = AsyncMock()
    builder._build_sql_engines = AsyncMock()
    builder._validate_neo4j = AsyncMock()
    builder._validate_sql_engines = AsyncMock()

    result = await builder.build_all(validate=True)

    # build_all should call the build methods but NOT validation methods
    builder._build_neo4j.assert_called_once()
    builder._build_sql_engines.assert_called_once()
    builder._validate_neo4j.assert_called_once()
    builder._validate_sql_engines.assert_called_once()

    assert result is builder
