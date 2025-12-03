from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.configuration.database_conf import (
    DatabasesConf,
    Neo4jConf,
    SqlAlchemyConf,
)
from memmachine.common.resource_manager.database_manager import DatabaseManager
from memmachine.common.vector_graph_store import VectorGraphStore


@pytest.fixture
def mock_conf():
    """Mock StoragesConf with dummy connection configurations."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {
        "neo1": Neo4jConf(
            host="localhost", port=1234, user="neo", password=SecretStr("pw")
        ),
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
    driver = builder.graph_stores["neo1"]
    assert isinstance(driver, VectorGraphStore)


@pytest.mark.asyncio
async def test_validate_neo4j(mock_conf):
    builder = DatabaseManager(mock_conf)

    mock_driver = MagicMock()
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_record = {"ok": 1}

    mock_driver.close = AsyncMock()
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

    builder.neo4j_drivers = {"neo1": mock_driver}

    await builder._validate_neo4j_drivers()
    mock_session.run.assert_awaited_once_with("RETURN 1 AS ok")


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
    builder._validate_neo4j_drivers = AsyncMock()
    builder._validate_sql_engines = AsyncMock()

    await builder.build_all(validate=False)

    assert "sqlite1" in builder.sql_engines
    assert "pg1" in builder.sql_engines
    assert "neo1" in builder.graph_stores
