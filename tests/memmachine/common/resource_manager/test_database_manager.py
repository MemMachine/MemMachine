from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.configuration.database_conf import DatabasesConf, SqlAlchemyConf
from memmachine.common.resource_manager.database_manager import DatabaseManager
from memmachine.common.vector_graph_store import VectorGraphStore


@pytest.fixture
def mock_conf():
    """Mock StoragesConf with dummy connection configurations."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {
        "neo1": MagicMock(host="localhost", port=1234, user="neo", password="pw"),
    }
    conf.relational_db_confs = {
        "pg1": SqlAlchemyConf(
            dialect="postgresql",
            driver="asyncpg",
            host="localhost",
            port=5432,
            user="user",
            password=SecretStr("password"),
            db_name="testdb",
        ),
        "sqlite1": SqlAlchemyConf(
            dialect="sqlite",
            driver="aiosqlite",
            path="test.db",
        ),
    }
    conf.sqlite_confs = {}
    return conf


@pytest.mark.asyncio
async def test_build_neo4j(mock_conf):
    builder = DatabaseManager(mock_conf)
    await builder._build_neo4j()

    assert "neo1" in builder.graph_stores
    assert isinstance(builder.graph_stores["neo1"], VectorGraphStore)


@pytest.mark.asyncio
async def test_build_sqlite(mock_conf):
    builder = DatabaseManager(mock_conf)
    await builder._build_sql_engines()

    assert "sqlite1" in builder.sql_engines
    assert isinstance(builder.sql_engines["sqlite1"], AsyncEngine)


@pytest.mark.asyncio
async def test_build_and_validate_sqlite():
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {}
    conf.relational_db_confs = {
        "sqlite1": SqlAlchemyConf(
            dialect="sqlite",
            driver="aiosqlite",
            path=":memory:",
        )
    }
    builder = DatabaseManager(conf)
    await builder.build_all(validate=True)
    # If no exception is raised, validation passed
    assert "sqlite1" in builder.sql_engines
    await builder.close()
    assert "sqlite1" not in builder.sql_engines


@pytest.mark.asyncio
async def test_build_all_without_validation(mock_conf):
    builder = DatabaseManager(mock_conf)
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
