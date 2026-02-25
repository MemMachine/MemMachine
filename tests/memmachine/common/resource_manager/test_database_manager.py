from typing import Any, cast
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
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)


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
    builder_any = cast(Any, builder)
    builder_any._build_neo4j = AsyncMock()
    builder_any._build_sql_engines = AsyncMock()
    builder_any._validate_neo4j_drivers = AsyncMock()
    builder_any._validate_sql_engines = AsyncMock()

    await builder.build_all(validate=False)

    assert "sqlite1" in builder.sql_engines
    assert "pg1" in builder.sql_engines
    assert "neo1" in builder.graph_stores


# ---------------------------------------------------------------------------
# Config forwarding tests (dedup + GDS)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_config_forwarded_to_store_params():
    """Verify dedup fields from Neo4jConf flow to Neo4jVectorGraphStoreParams."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {
        "neo_dedup": Neo4jConf(
            host="localhost",
            port=7687,
            user="neo4j",
            password=SecretStr("pw"),
            dedup_trigger_threshold=500,
            dedup_embedding_threshold=0.90,
            dedup_property_threshold=0.7,
            dedup_auto_merge=True,
        ),
    }
    conf.relational_db_confs = {}

    builder = DatabaseManager(conf)
    await builder._build_neo4j()

    store = cast(Neo4jVectorGraphStore, builder.graph_stores["neo_dedup"])
    assert store._dedup_trigger_threshold == 500
    assert store._dedup_embedding_threshold == 0.90
    assert store._dedup_property_threshold == 0.7
    assert store._dedup_auto_merge is True


@pytest.mark.asyncio
async def test_gds_config_forwarded_to_store_params():
    """Verify GDS fields from Neo4jConf flow to Neo4jVectorGraphStoreParams."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {
        "neo_gds": Neo4jConf(
            host="localhost",
            port=7687,
            user="neo4j",
            password=SecretStr("pw"),
            gds_enabled=True,
            gds_default_damping_factor=0.9,
            gds_default_max_iterations=30,
        ),
    }
    conf.relational_db_confs = {}

    builder = DatabaseManager(conf)
    await builder._build_neo4j()

    store = cast(Neo4jVectorGraphStore, builder.graph_stores["neo_gds"])
    assert store._gds_enabled is True
    assert store._gds_default_damping_factor == 0.9
    assert store._gds_default_max_iterations == 30


@pytest.mark.asyncio
async def test_default_config_values_forwarded():
    """Verify defaults from Neo4jConf flow through when not explicitly set."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {
        "neo_defaults": Neo4jConf(
            host="localhost",
            port=7687,
            user="neo4j",
            password=SecretStr("pw"),
        ),
    }
    conf.relational_db_confs = {}

    builder = DatabaseManager(conf)
    await builder._build_neo4j()

    store = cast(Neo4jVectorGraphStore, builder.graph_stores["neo_defaults"])
    # Dedup defaults
    assert store._dedup_trigger_threshold == 1000
    assert store._dedup_embedding_threshold == 0.95
    assert store._dedup_property_threshold == 0.8
    assert store._dedup_auto_merge is False
    # PageRank auto defaults
    assert store._pagerank_auto_enabled is True
    assert store._pagerank_trigger_threshold == 50
    # GDS defaults
    assert store._gds_enabled is False
    assert store._gds_default_damping_factor == 0.85
    assert store._gds_default_max_iterations == 20


# ---------------------------------------------------------------------------
# GDS availability respects gds_enabled config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pagerank_auto_config_forwarded_to_store_params():
    """Verify PageRank auto fields from Neo4jConf flow to the store."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {
        "neo_pr": Neo4jConf(
            host="localhost",
            port=7687,
            user="neo4j",
            password=SecretStr("pw"),
            pagerank_auto_enabled=False,
            pagerank_trigger_threshold=200,
        ),
    }
    conf.relational_db_confs = {}

    builder = DatabaseManager(conf)
    await builder._build_neo4j()

    store = cast(Neo4jVectorGraphStore, builder.graph_stores["neo_pr"])
    assert store._pagerank_auto_enabled is False
    assert store._pagerank_trigger_threshold == 200


@pytest.mark.asyncio
async def test_is_gds_available_returns_false_when_disabled():
    """is_gds_available() returns False immediately when gds_enabled=False."""
    from neo4j import AsyncDriver

    driver = AsyncMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock()

    store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            gds_enabled=False,
        )
    )
    result = await store.is_gds_available()
    assert result is False
    # Should NOT have queried Neo4j
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_is_gds_available_queries_when_enabled():
    """is_gds_available() queries Neo4j when gds_enabled=True."""
    from neo4j import AsyncDriver

    driver = AsyncMock(spec=AsyncDriver)
    # Simulate GDS not being installed
    driver.execute_query = AsyncMock(side_effect=Exception("Unknown function"))

    store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            gds_enabled=True,
        )
    )
    result = await store.is_gds_available()
    assert result is False
    driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_is_gds_available_returns_true_when_gds_installed():
    """is_gds_available() returns True when GDS plugin responds."""
    from neo4j import AsyncDriver

    driver = AsyncMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock(return_value=([], MagicMock(), MagicMock()))

    store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            gds_enabled=True,
        )
    )
    result = await store.is_gds_available()
    assert result is True
