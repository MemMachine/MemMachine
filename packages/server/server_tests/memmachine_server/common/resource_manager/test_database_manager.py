from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.configuration.database_conf import (
    DatabasesConf,
    Neo4jConf,
    QdrantConf,
    SqlAlchemyConf,
    SQLiteVectorStoreConf,
    SQLiteVectorStoreEngine,
    SQLiteVecVectorStoreConf,
)
from memmachine_server.common.errors import (
    QdrantConfigurationError,
    VectorStoreConfigurationError,
)
from memmachine_server.common.resource_manager.database_manager import DatabaseManager
from memmachine_server.common.vector_graph_store import VectorGraphStore


def _empty_vector_store_confs(conf: MagicMock) -> None:
    """Set vector-store-related conf dicts to empty on a DatabasesConf mock."""
    conf.qdrant_confs = {}
    conf.sqlite_vector_store_confs = {}
    conf.sqlite_vec_vector_store_confs = {}


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
    _empty_vector_store_confs(conf)
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
    _empty_vector_store_confs(conf)
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
    _empty_vector_store_confs(conf)

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
    _empty_vector_store_confs(conf)
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
    _empty_vector_store_confs(conf)

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
    _empty_vector_store_confs(conf)
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


# --- Qdrant ---


def _qdrant_only_conf() -> MagicMock:
    """Build a DatabasesConf mock with only a Qdrant entry."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {}
    conf.relational_db_confs = {}
    conf.nebula_graph_confs = {}
    conf.qdrant_confs = {
        "qdrant1": QdrantConf(host="localhost", port=6333),
    }
    conf.sqlite_vector_store_confs = {}
    conf.sqlite_vec_vector_store_confs = {}
    return conf


@pytest.mark.asyncio
async def test_qdrant_client_kwargs_forwarded():
    """host, port, grpc_port, prefer_grpc, and https are forwarded to AsyncQdrantClient."""
    conf = _qdrant_only_conf()
    conf.qdrant_confs["qdrant1"] = QdrantConf(
        host="qdrant.example.com",
        port=7333,
        grpc_port=7334,
        prefer_grpc=True,
        https=True,
        api_key=SecretStr("secret-key"),
    )

    mock_client = AsyncMock()
    mock_client.get_collections = AsyncMock(return_value=[])
    mock_client.close = AsyncMock()

    with (
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStore",
        ) as mock_store_cls,
        patch(
            "qdrant_client.AsyncQdrantClient",
            return_value=mock_client,
        ) as mock_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        builder = DatabaseManager(conf)
        await builder.async_get_qdrant_client("qdrant1")

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["host"] == "qdrant.example.com"
    assert call_kwargs["port"] == 7333
    assert call_kwargs["grpc_port"] == 7334
    assert call_kwargs["prefer_grpc"] is True
    assert call_kwargs["https"] is True
    assert call_kwargs["api_key"] == "secret-key"


@pytest.mark.asyncio
async def test_qdrant_api_key_omitted_when_empty():
    """api_key is not forwarded when it is the empty default."""
    conf = _qdrant_only_conf()

    mock_client = AsyncMock()
    mock_client.get_collections = AsyncMock(return_value=[])
    mock_client.close = AsyncMock()

    with (
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStore",
        ) as mock_store_cls,
        patch(
            "qdrant_client.AsyncQdrantClient",
            return_value=mock_client,
        ) as mock_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        builder = DatabaseManager(conf)
        await builder.async_get_qdrant_client("qdrant1")

    assert "api_key" not in mock_cls.call_args.kwargs


@pytest.mark.asyncio
async def test_qdrant_creates_vector_store():
    """async_get_qdrant_client creates a QdrantVectorStore and stores it."""
    conf = _qdrant_only_conf()
    conf.qdrant_confs["qdrant1"] = QdrantConf(
        is_distributed=True,
        registry_replication_factor=3,
    )

    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    with (
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStoreParams",
        ) as mock_params_cls,
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStore",
        ) as mock_store_cls,
        patch(
            "qdrant_client.AsyncQdrantClient",
            return_value=mock_client,
        ),
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        builder = DatabaseManager(conf)
        await builder.async_get_qdrant_client("qdrant1")

    mock_params_cls.assert_called_once_with(
        client=mock_client,
        is_distributed=True,
        registry_replication_factor=3,
    )
    mock_store_cls.assert_called_once_with(mock_params_cls.return_value)
    mock_store_cls.return_value.startup.assert_awaited_once()
    assert "qdrant1" in builder.vector_stores


@pytest.mark.asyncio
async def test_get_vector_store_qdrant():
    """get_vector_store returns the VectorStore for a Qdrant config."""
    conf = _qdrant_only_conf()

    mock_client = AsyncMock()
    mock_client.get_collections = AsyncMock(return_value=[])
    mock_client.close = AsyncMock()

    with (
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStore",
        ) as mock_store_cls,
        patch(
            "qdrant_client.AsyncQdrantClient",
            return_value=mock_client,
        ),
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        builder = DatabaseManager(conf)
        store = await builder.get_vector_store("qdrant1")

    assert store is mock_store_cls.return_value


@pytest.mark.asyncio
async def test_qdrant_config_not_found():
    """async_get_qdrant_client raises ValueError for unknown names."""
    conf = _qdrant_only_conf()
    conf.qdrant_confs = {}
    builder = DatabaseManager(conf)
    with pytest.raises(ValueError, match="Qdrant config 'missing' not found"):
        await builder.async_get_qdrant_client("missing")


@pytest.mark.asyncio
async def test_qdrant_validation_failure():
    """validate_qdrant_client raises QdrantConfigurationError on failure."""
    mock_client = AsyncMock()
    mock_client.get_collections = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.close = AsyncMock()

    with pytest.raises(QdrantConfigurationError, match="failed verification"):
        await DatabaseManager.validate_qdrant_client("qdrant1", mock_client)

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_qdrant_close():
    """close() cleans up Qdrant clients and vector stores."""
    conf = _qdrant_only_conf()

    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    with (
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.qdrant_vector_store.QdrantVectorStore",
        ) as mock_store_cls,
        patch(
            "qdrant_client.AsyncQdrantClient",
            return_value=mock_client,
        ),
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        mock_store_cls.return_value.shutdown = AsyncMock()
        builder = DatabaseManager(conf)
        await builder.async_get_qdrant_client("qdrant1")

    assert "qdrant1" in builder.qdrant_clients
    assert "qdrant1" in builder.vector_stores

    await builder.close()

    assert builder.qdrant_clients == {}
    assert builder.vector_stores == {}
    mock_client.close.assert_awaited()


# --- SQLite-backed VectorStores ---


def _sqlite_vector_store_only_conf(**vs_overrides) -> MagicMock:
    """Build a DatabasesConf mock with only SQLite-based vector store entries."""
    conf = MagicMock(spec=DatabasesConf)
    conf.neo4j_confs = {}
    conf.relational_db_confs = {}
    conf.nebula_graph_confs = {}
    conf.qdrant_confs = {}
    conf.sqlite_vector_store_confs = vs_overrides.get("sqlite_vector_store_confs", {})
    conf.sqlite_vec_vector_store_confs = vs_overrides.get(
        "sqlite_vec_vector_store_confs", {}
    )
    return conf


@pytest.mark.asyncio
async def test_sqlite_vector_store_creates_store_and_engine():
    """async_get_sqlite_vector_store builds an engine, store, and starts it up."""
    conf = _sqlite_vector_store_only_conf(
        sqlite_vector_store_confs={
            "vs1": SQLiteVectorStoreConf(
                path="vs.db",
                vector_search_engine=SQLiteVectorStoreEngine.USEARCH,
                index_directory="/tmp/vs",
                save_threshold=200,
            )
        }
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ) as mock_create,
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStoreParams",
        ) as mock_params_cls,
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStore",
        ) as mock_store_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        mock_store_cls.return_value.shutdown = AsyncMock()
        builder = DatabaseManager(conf)
        store = await builder.async_get_sqlite_vector_store("vs1")

    assert store is mock_store_cls.return_value
    mock_create.assert_called_once()
    create_args = mock_create.call_args
    assert create_args.args[0] == "sqlite+aiosqlite:///vs.db"

    params_kwargs = mock_params_cls.call_args.kwargs
    assert params_kwargs["sqlalchemy_engine"] is mock_engine
    assert params_kwargs["index_directory"] == "/tmp/vs"
    assert params_kwargs["save_threshold"] == 200
    assert callable(params_kwargs["vector_search_engine_factory"])

    mock_store_cls.return_value.startup.assert_awaited_once()
    assert "vs1" in builder.vector_stores
    assert "vs1" in builder.vector_store_sql_engines


@pytest.mark.asyncio
async def test_sqlite_vector_store_default_engine_is_usearch():
    """When no engine is configured, the factory uses USearch (the default).

    Avoids touching the hnswlib import path so the test runs on CI, where
    hnswlib build caches mismatch CPU across GitHub-hosted runners.
    """
    conf = _sqlite_vector_store_only_conf(
        sqlite_vector_store_confs={"vs1": SQLiteVectorStoreConf(path="vs.db")}
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStoreParams",
        ) as mock_params_cls,
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStore",
        ) as mock_store_cls,
        patch(
            "memmachine_server.common.vector_store.vector_search_engine.usearch_engine.USearchVectorSearchEngine",
        ) as mock_usearch_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        mock_store_cls.return_value.shutdown = AsyncMock()
        builder = DatabaseManager(conf)
        await builder.async_get_sqlite_vector_store("vs1")

        # Invoke the factory the manager passed into params and confirm it
        # routes to the USearch engine.
        from memmachine_server.common.data_types import SimilarityMetric

        factory = mock_params_cls.call_args.kwargs["vector_search_engine_factory"]
        factory(8, SimilarityMetric.COSINE)

    mock_usearch_cls.assert_called_once()


@pytest.mark.asyncio
async def test_sqlite_vector_store_caches_after_first_call():
    """Repeat calls return the same store without rebuilding."""
    conf = _sqlite_vector_store_only_conf(
        sqlite_vector_store_confs={"vs1": SQLiteVectorStoreConf(path="vs.db")}
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ) as mock_create,
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStore",
        ) as mock_store_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        mock_store_cls.return_value.shutdown = AsyncMock()
        builder = DatabaseManager(conf)
        first = await builder.async_get_sqlite_vector_store("vs1")
        second = await builder.async_get_sqlite_vector_store("vs1")

    assert first is second
    assert mock_create.call_count == 1
    mock_store_cls.return_value.startup.assert_awaited_once()


@pytest.mark.asyncio
async def test_sqlite_vector_store_startup_failure_disposes_engine():
    """If startup() fails, the engine is disposed and no state is retained."""
    conf = _sqlite_vector_store_only_conf(
        sqlite_vector_store_confs={"vs1": SQLiteVectorStoreConf(path="vs.db")}
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStore",
        ) as mock_store_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        builder = DatabaseManager(conf)
        with pytest.raises(VectorStoreConfigurationError, match="failed to start"):
            await builder.async_get_sqlite_vector_store("vs1")

    mock_engine.dispose.assert_awaited_once()
    assert "vs1" not in builder.vector_stores
    assert "vs1" not in builder.vector_store_sql_engines


@pytest.mark.asyncio
async def test_sqlite_vector_store_unknown_name_raises():
    conf = _sqlite_vector_store_only_conf()
    builder = DatabaseManager(conf)
    with pytest.raises(
        ValueError, match="SQLiteVectorStore config 'missing' not found"
    ):
        await builder.async_get_sqlite_vector_store("missing")


@pytest.mark.asyncio
async def test_sqlite_vec_vector_store_creates_store_and_engine():
    """async_get_sqlite_vec_vector_store builds an engine, store, and starts it up."""
    conf = _sqlite_vector_store_only_conf(
        sqlite_vec_vector_store_confs={"vec1": SQLiteVecVectorStoreConf(path="vec.db")}
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ) as mock_create,
        patch(
            "memmachine_server.common.vector_store.sqlite_vec_vector_store.SQLiteVecVectorStoreParams",
        ) as mock_params_cls,
        patch(
            "memmachine_server.common.vector_store.sqlite_vec_vector_store.SQLiteVecVectorStore",
        ) as mock_store_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        mock_store_cls.return_value.shutdown = AsyncMock()
        builder = DatabaseManager(conf)
        store = await builder.async_get_sqlite_vec_vector_store("vec1")

    assert store is mock_store_cls.return_value
    create_args = mock_create.call_args
    assert create_args.args[0] == "sqlite+aiosqlite:///vec.db"

    params_kwargs = mock_params_cls.call_args.kwargs
    assert params_kwargs["engine"] is mock_engine

    mock_store_cls.return_value.startup.assert_awaited_once()
    assert "vec1" in builder.vector_stores
    assert "vec1" in builder.vector_store_sql_engines


@pytest.mark.asyncio
async def test_sqlite_vec_vector_store_unknown_name_raises():
    conf = _sqlite_vector_store_only_conf()
    builder = DatabaseManager(conf)
    with pytest.raises(
        ValueError, match="SQLiteVecVectorStore config 'missing' not found"
    ):
        await builder.async_get_sqlite_vec_vector_store("missing")


@pytest.mark.asyncio
async def test_get_vector_store_dispatches_to_sqlite_backends():
    """get_vector_store routes to the right SQLite-backed builder."""
    conf = _sqlite_vector_store_only_conf(
        sqlite_vector_store_confs={"vs1": SQLiteVectorStoreConf(path="vs.db")},
        sqlite_vec_vector_store_confs={"vec1": SQLiteVecVectorStoreConf(path="vec.db")},
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStore",
        ) as mock_vs_cls,
        patch(
            "memmachine_server.common.vector_store.sqlite_vec_vector_store.SQLiteVecVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vec_vector_store.SQLiteVecVectorStore",
        ) as mock_vec_cls,
    ):
        mock_vs_cls.return_value.startup = AsyncMock()
        mock_vs_cls.return_value.shutdown = AsyncMock()
        mock_vec_cls.return_value.startup = AsyncMock()
        mock_vec_cls.return_value.shutdown = AsyncMock()
        builder = DatabaseManager(conf)
        vs = await builder.get_vector_store("vs1")
        vec = await builder.get_vector_store("vec1")

    assert vs is mock_vs_cls.return_value
    assert vec is mock_vec_cls.return_value


@pytest.mark.asyncio
async def test_get_vector_store_unknown_name_raises():
    conf = _sqlite_vector_store_only_conf()
    builder = DatabaseManager(conf)
    with pytest.raises(ValueError, match="VectorStore 'missing' not found"):
        await builder.get_vector_store("missing")


@pytest.mark.asyncio
async def test_close_disposes_vector_store_engines_and_shuts_down_stores():
    """close() shuts down vector stores and disposes their engines."""
    conf = _sqlite_vector_store_only_conf(
        sqlite_vector_store_confs={"vs1": SQLiteVectorStoreConf(path="vs.db")}
    )

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    with (
        patch(
            "memmachine_server.common.resource_manager.database_manager.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStoreParams",
        ),
        patch(
            "memmachine_server.common.vector_store.sqlite_vector_store.SQLiteVectorStore",
        ) as mock_store_cls,
    ):
        mock_store_cls.return_value.startup = AsyncMock()
        mock_store_cls.return_value.shutdown = AsyncMock()
        builder = DatabaseManager(conf)
        await builder.async_get_sqlite_vector_store("vs1")

        assert "vs1" in builder.vector_stores
        assert "vs1" in builder.vector_store_sql_engines

        await builder.close()

    assert builder.vector_stores == {}
    assert builder.vector_store_sql_engines == {}
    mock_store_cls.return_value.shutdown.assert_awaited_once()
    mock_engine.dispose.assert_awaited()
