from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.configuration.database_conf import (
    AgeConf,
    DatabasesConf,
    Neo4jConf,
    SqlAlchemyConf,
)
from memmachine_server.common.resource_manager.database_manager import DatabaseManager
from memmachine_server.common.vector_graph_store import VectorGraphStore


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
    conf.nebula_graph_confs = {}
    conf.age_confs = {}
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
    conf.nebula_graph_confs = {}
    conf.age_confs = {}
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
    builder_any = cast(Any, builder)
    builder_any._build_neo4j = AsyncMock()
    builder_any._build_sql_engines = AsyncMock()
    builder_any._validate_neo4j_drivers = AsyncMock()
    builder_any._validate_sql_engines = AsyncMock()

    await builder.build_all(validate=False)

    assert "sqlite1" in builder.sql_engines
    assert "pg1" in builder.sql_engines
    assert "neo1" in builder.graph_stores


@pytest.mark.asyncio
async def test_neo4j_pool_lifecycle_kwargs():
    """max_connection_lifetime and liveness_check_timeout are forwarded to the driver."""
    conf = MagicMock(spec=DatabasesConf)
    neo4j_conf = Neo4jConf(
        host="localhost",
        port=7687,
        user="neo4j",
        password=SecretStr("pw"),
        max_connection_lifetime=3000.0,
        liveness_check_timeout=300.0,
    )
    conf.neo4j_confs = {"neo1": neo4j_conf}
    conf.relational_db_confs = {}
    conf.nebula_graph_confs = {}
    conf.age_confs = {}

    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.AsyncGraphDatabase.driver",
            return_value=mock_driver,
        ) as mock_driver_cls,
        patch(
            "memmachine_server.common.resource_manager.database_manager.Neo4jVectorGraphStoreParams",
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager.Neo4jVectorGraphStore",
        ),
    ):
        builder = DatabaseManager(conf)
        await builder.async_get_neo4j_driver("neo1")

    call_kwargs = mock_driver_cls.call_args.kwargs
    assert call_kwargs["max_connection_lifetime"] == 3000.0
    assert call_kwargs["liveness_check_timeout"] == 300.0


@pytest.mark.asyncio
async def test_sqlalchemy_pool_lifecycle_kwargs():
    """pool_timeout, pool_recycle, and pool_pre_ping are forwarded to create_async_engine."""
    conf = MagicMock(spec=DatabasesConf)
    sql_conf = SqlAlchemyConf(
        dialect="sqlite",
        driver="aiosqlite",
        path=":memory:",
        pool_timeout=30,
        pool_recycle=3000,
        pool_pre_ping=True,
    )
    conf.neo4j_confs = {}
    conf.nebula_graph_confs = {}
    conf.relational_db_confs = {"db1": sql_conf}

    mock_engine = MagicMock(spec=AsyncEngine)

    with patch(
        "memmachine_server.common.resource_manager.database_manager.create_async_engine",
        return_value=mock_engine,
    ) as mock_create:
        builder = DatabaseManager(conf)
        await builder.async_get_sql_engine("db1")

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["pool_timeout"] == 30
    assert call_kwargs["pool_recycle"] == 3000
    assert call_kwargs["pool_pre_ping"] is True


@pytest.mark.asyncio
async def test_neo4j_pool_lifecycle_kwargs_none_omitted():
    """When pool lifecycle fields are None, they are not forwarded to the driver."""
    conf = MagicMock(spec=DatabasesConf)
    neo4j_conf = Neo4jConf(
        host="localhost",
        port=7687,
        user="neo4j",
        password=SecretStr("pw"),
    )
    conf.neo4j_confs = {"neo1": neo4j_conf}
    conf.relational_db_confs = {}
    conf.nebula_graph_confs = {}
    conf.age_confs = {}

    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.AsyncGraphDatabase.driver",
            return_value=mock_driver,
        ) as mock_driver_cls,
        patch(
            "memmachine_server.common.resource_manager.database_manager.Neo4jVectorGraphStoreParams",
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager.Neo4jVectorGraphStore",
        ),
    ):
        builder = DatabaseManager(conf)
        await builder.async_get_neo4j_driver("neo1")

    call_kwargs = mock_driver_cls.call_args.kwargs
    assert "max_connection_lifetime" not in call_kwargs
    assert "liveness_check_timeout" not in call_kwargs


@pytest.mark.asyncio
async def test_sqlalchemy_pool_lifecycle_kwargs_none_omitted():
    """When pool lifecycle fields are None, they are not forwarded to create_async_engine."""
    conf = MagicMock(spec=DatabasesConf)
    sql_conf = SqlAlchemyConf(
        dialect="sqlite",
        driver="aiosqlite",
        path=":memory:",
    )
    conf.neo4j_confs = {}
    conf.nebula_graph_confs = {}
    conf.relational_db_confs = {"db1": sql_conf}

    mock_engine = MagicMock(spec=AsyncEngine)

    with patch(
        "memmachine_server.common.resource_manager.database_manager.create_async_engine",
        return_value=mock_engine,
    ) as mock_create:
        builder = DatabaseManager(conf)
        await builder.async_get_sql_engine("db1")

    call_kwargs = mock_create.call_args.kwargs
    assert "pool_timeout" not in call_kwargs
    assert "pool_recycle" not in call_kwargs
    assert "pool_pre_ping" not in call_kwargs


def _age_conf(**overrides: Any) -> AgeConf:
    defaults: dict[str, Any] = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": SecretStr("pw"),
        "db_name": "memmachine",
        "graph_name": "mem_graph",
    }
    defaults.update(overrides)
    return AgeConf(**defaults)


@pytest.mark.asyncio
async def test_build_age(mock_conf):
    """AGE engine + graph store are both created and registered under the conf name."""
    mock_conf.age_confs = {"age1": _age_conf()}
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.sync_engine = MagicMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager._register_age_connect_hook",
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager.AgeVectorGraphStore",
        ) as mock_store_cls,
    ):
        builder = DatabaseManager(mock_conf)
        await builder.async_get_age_engine("age1")

    assert builder.age_engines["age1"] is mock_engine
    assert builder.graph_stores["age1"] is mock_store_cls.return_value


@pytest.mark.asyncio
async def test_age_pool_lifecycle_kwargs_forwarded():
    """pool_size, max_overflow, pool_timeout, pool_recycle, pool_pre_ping are forwarded."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {}
    conf.relational_db_confs = {}
    conf.nebula_graph_confs = {}
    conf.age_confs = {
        "age1": _age_conf(
            pool_size=7,
            max_overflow=3,
            pool_timeout=45,
            pool_recycle=1800,
            pool_pre_ping=True,
        )
    }

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.sync_engine = MagicMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ) as mock_create,
        patch(
            "memmachine_server.common.resource_manager.database_manager._register_age_connect_hook",
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager.AgeVectorGraphStore",
        ),
    ):
        builder = DatabaseManager(conf)
        await builder.async_get_age_engine("age1")

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["pool_size"] == 7
    assert call_kwargs["max_overflow"] == 3
    assert call_kwargs["pool_timeout"] == 45
    assert call_kwargs["pool_recycle"] == 1800
    assert call_kwargs["pool_pre_ping"] is True


@pytest.mark.asyncio
async def test_age_pool_lifecycle_kwargs_none_omitted():
    """When pool knobs are None, they are not forwarded to create_async_engine."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {}
    conf.relational_db_confs = {}
    conf.nebula_graph_confs = {}
    conf.age_confs = {"age1": _age_conf()}

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.sync_engine = MagicMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ) as mock_create,
        patch(
            "memmachine_server.common.resource_manager.database_manager._register_age_connect_hook",
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager.AgeVectorGraphStore",
        ),
    ):
        builder = DatabaseManager(conf)
        await builder.async_get_age_engine("age1")

    call_kwargs = mock_create.call_args.kwargs
    for key in (
        "pool_size",
        "max_overflow",
        "pool_timeout",
        "pool_recycle",
        "pool_pre_ping",
    ):
        assert key not in call_kwargs


@pytest.mark.asyncio
async def test_age_graph_store_params_forwarded():
    """Index thresholds and HNSW knobs from AgeConf reach AgeVectorGraphStoreParams."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {}
    conf.relational_db_confs = {}
    conf.nebula_graph_confs = {}
    conf.age_confs = {
        "age1": _age_conf(
            range_index_creation_threshold=500,
            vector_index_creation_threshold=1000,
            hnsw_m=24,
            hnsw_ef_construction=96,
        )
    }

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.sync_engine = MagicMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager._register_age_connect_hook",
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager.AgeVectorGraphStore",
        ),
        patch(
            "memmachine_server.common.resource_manager.database_manager.AgeVectorGraphStoreParams",
        ) as mock_params,
    ):
        builder = DatabaseManager(conf)
        await builder.async_get_age_engine("age1")

    params_kwargs = mock_params.call_args.kwargs
    assert params_kwargs["graph_name"] == "mem_graph"
    assert params_kwargs["range_index_creation_threshold"] == 500
    assert params_kwargs["vector_index_creation_threshold"] == 1000
    assert params_kwargs["hnsw_m"] == 24
    assert params_kwargs["hnsw_ef_construction"] == 96


@pytest.mark.asyncio
async def test_validate_age_engine_passes():
    """validate_age_engine completes silently when SELECT 1 returns the expected row."""
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    mock_result = MagicMock()
    mock_result.fetchone.return_value = (1,)

    mock_conn = AsyncMock()
    mock_conn.execute.return_value = mock_result

    mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)

    await DatabaseManager.validate_age_engine("age1", cast(AsyncEngine, mock_engine))
    mock_engine.dispose.assert_not_awaited()
