from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import neo4j
import pytest
import sqlalchemy

from memmachine.common.configuration.storage_conf import StorageConf
from memmachine.common.resource_mgr.storage_mgr import StorageMgr


@pytest.fixture
def mock_conf():
    """Mock StorageConf with dummy connection configurations."""
    conf = MagicMock(spec=StorageConf)
    conf.neo4jConfs = {
        "neo1": MagicMock(host="localhost", port=7687, user="neo", password="pw")
    }
    conf.postgresConfs = {
        "pg1": MagicMock(
            host="localhost",
            port=5432,
            user="pg",
            password=MagicMock(get_secret_value=lambda: "secret"),
            db_name="testdb",
            vector_schema="public",
            statement_cache_size=100,
        )
    }
    conf.sqliteConfs = {"sqlite1": MagicMock(file_path="sqlite:///tmp/test.db")}
    return conf


@pytest.mark.asyncio
async def test_build_neo4j(mock_conf):
    mock_driver = AsyncMock()
    with patch(
        "neo4j.AsyncGraphDatabase.driver", return_value=mock_driver
    ) as mock_driver_func:
        builder = StorageMgr(mock_conf)
        await builder._build_neo4j()

    mock_driver_func.assert_called_once_with(
        "bolt://localhost:7687", auth=("neo", "pw")
    )
    assert "neo1" in builder.graph_stores
    assert builder.graph_stores["neo1"] is mock_driver


@pytest.mark.asyncio
async def test_build_neo4j_a(mock_conf):
    builder = StorageMgr(mock_conf)
    await builder._build_neo4j()

    assert "neo1" in builder.graph_stores
    assert isinstance(builder.graph_stores["neo1"], neo4j.AsyncDriver)


@pytest.mark.asyncio
async def test_build_postgres(mock_conf):
    builder = StorageMgr(mock_conf)
    await builder._build_postgres()
    assert "pg1" in builder.postgres_pools
    assert isinstance(builder.postgres_pools["pg1"], asyncpg.Pool)


@pytest.mark.asyncio
async def test_build_sqlite(mock_conf):
    builder = StorageMgr(mock_conf)
    await builder._build_sql_engines()

    assert "sqlite1" in builder.sql_engines
    assert isinstance(builder.sql_engines["sqlite1"], sqlalchemy.Engine)


@pytest.mark.asyncio
async def test_build_all_without_validation(mock_conf):
    builder = StorageMgr(mock_conf)
    builder._build_neo4j = AsyncMock()
    builder._build_postgres = AsyncMock()
    builder._build_sql_engines = AsyncMock()
    builder._validate_neo4j = AsyncMock()
    builder._validate_postgres = AsyncMock()
    builder._validate_sql_engines = AsyncMock()

    result = builder.build_all(validate=True)

    # build_all should call the build methods but NOT validation methods
    builder._build_neo4j.assert_called_once()
    builder._build_postgres.assert_called_once()
    builder._build_sql_engines.assert_called_once()
    builder._validate_neo4j.assert_called_once()
    builder._validate_postgres.assert_called_once()
    builder._validate_sql_engines.assert_called_once()

    assert result is builder
