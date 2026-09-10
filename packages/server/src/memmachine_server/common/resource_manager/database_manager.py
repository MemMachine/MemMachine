"""Manage database engines for SQL, Neo4j, NebulaGraph, and VectorStore backends."""

import asyncio
import logging
from asyncio import Lock
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Self

from neo4j import AsyncDriver, AsyncGraphDatabase
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import event, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

from memmachine_server.common.configuration.database_conf import (
    DatabasesConf,
    MilvusConf,
    Neo4jConf,
    QdrantConf,
    SqlAlchemyConf,
    SQLiteVectorStoreConf,
    SQLiteVectorStoreEngine,
    SQLiteVecVectorStoreConf,
)
from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
)
from memmachine_server.common.errors import (
    MilvusConfigurationError,
    Neo4JConfigurationError,
    QdrantConfigurationError,
    SQLConfigurationError,
    VectorStoreConfigurationError,
)
from memmachine_server.common.vector_graph_store import VectorGraphStore
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.common.vector_store import IndexedProperties, VectorStore
from memmachine_server.common.vector_store.vector_search_engine import (
    VectorSearchEngine,
)

# TYPE_CHECKING is True only when type checkers (mypy, pyright) run, False at runtime.
# This allows type hints without requiring nebulagraph_python or qdrant_client to be
# installed unless those backends are actually used. Imports happen at use sites.
if TYPE_CHECKING:
    from nebulagraph_python.client import NebulaAsyncClient
    from pymilvus import MilvusClient
    from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

_INDEXED_PROPERTIES = TypeAdapter(IndexedProperties)


def enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """Enforce foreign keys on every connection of a SQLite engine.

    SQLite defaults foreign_keys to OFF, per connection. Registering the
    pragma at engine creation, before any connection is pooled, is the
    only placement that covers every connection: a listener added later
    misses already-pooled connections, and mutating a live pool's
    listeners races its event dispatch. No-op for other dialects.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(
        dbapi_connection: DBAPIConnection,
        _connection_record: ConnectionPoolEntry,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _sql_engine_kwargs(conf: SqlAlchemyConf) -> dict[str, Any]:
    """Build create_async_engine keywords, omitting anything left unset."""
    kwargs: dict[str, Any] = {"echo": False, "future": True}
    for attr in (
        "pool_size",
        "max_overflow",
        "pool_timeout",
        "pool_recycle",
        "pool_pre_ping",
    ):
        value = getattr(conf, attr)
        if value is not None:
            kwargs[attr] = value

    # Gate on the driver, not the dialect: these reach asyncpg.connect() and
    # mean nothing to aiosqlite or aiomysql.
    if conf.driver == "asyncpg":
        connect_args: dict[str, float] = {}
        if conf.command_timeout is not None:
            connect_args["command_timeout"] = conf.command_timeout
        if conf.connect_timeout is not None:
            connect_args["timeout"] = conf.connect_timeout
        if connect_args:
            kwargs["connect_args"] = connect_args
    return kwargs


class DatabaseManager:
    """Create and manage database backends with lazy initialization."""

    def __init__(self, conf: DatabasesConf) -> None:
        """Initialize with database configuration."""
        self.conf = conf
        self.graph_stores: dict[str, VectorGraphStore] = {}
        self.vector_stores: dict[str, VectorStore] = {}
        self.sql_engines: dict[str, AsyncEngine] = {}
        self.neo4j_drivers: dict[str, AsyncDriver] = {}
        # String annotation "NebulaAsyncClient" (forward reference) because the type
        # is only imported under TYPE_CHECKING and doesn't exist at runtime.
        # Type checkers see it, but runtime treats it as a string literal.
        self.nebula_clients: dict[str, NebulaAsyncClient] = {}
        self.qdrant_clients: dict[str, AsyncQdrantClient] = {}
        self.milvus_clients: dict[str, MilvusClient] = {}
        # SQLAlchemy engines that back SQLite-based vector stores. Tracked
        # separately from `sql_engines` (which holds user-facing relational
        # databases) so vector store internals don't collide with caller names.
        self.vector_store_sql_engines: dict[str, AsyncEngine] = {}

        self._lock = Lock()
        self._neo4j_locks: dict[str, Lock] = {}
        self._sql_locks: dict[str, Lock] = {}
        self._nebula_locks: dict[str, Lock] = {}
        self._qdrant_locks: dict[str, Lock] = {}
        self._milvus_locks: dict[str, Lock] = {}
        self._vector_store_locks: dict[str, Lock] = {}

    async def build_all(self, validate: bool = False) -> Self:
        """Optionally eagerly initialize all backends."""
        neo4j_tasks = [
            self.async_get_neo4j_driver(name, validate=validate)
            for name in self.conf.neo4j_confs
        ]
        relation_db_tasks = [
            self.async_get_sql_engine(name, validate=validate)
            for name in self.conf.relational_db_confs
        ]
        nebula_tasks = [
            self.async_get_nebula_client(name, validate=validate)
            for name in self.conf.nebula_graph_confs
        ]
        qdrant_tasks = [
            self.async_get_qdrant_client(name, validate=validate)
            for name in self.conf.qdrant_confs
        ]
        milvus_tasks = [
            self.async_get_milvus_client(name, validate=validate)
            for name in self.conf.milvus_confs
        ]
        # Lazy build will occur in get_* calls, but build_all can trigger them
        tasks = (
            neo4j_tasks + relation_db_tasks + nebula_tasks + qdrant_tasks + milvus_tasks
        )
        await asyncio.gather(*tasks)

        if validate:
            await asyncio.gather(
                self._validate_neo4j_drivers(),
                self._validate_sql_engines(),
                self._validate_nebula_clients(),
                self._validate_qdrant_clients(),
                self._validate_milvus_clients(),
            )

        return self

    async def close(self) -> None:
        """Close all database connections."""
        async with self._lock:
            tasks = []
            for name, driver in self.neo4j_drivers.items():
                tasks.append(self._close_async_driver(name, driver))
            for name, engine in self.sql_engines.items():
                tasks.append(self._close_async_engine(name, engine))
            for name, engine in self.vector_store_sql_engines.items():
                tasks.append(self._close_async_engine(name, engine))
            for name, client in self.nebula_clients.items():
                tasks.append(self._close_nebula_client(name, client))
            for name, client in self.qdrant_clients.items():
                tasks.append(self._close_qdrant_client(name, client))
            for name, client in self.milvus_clients.items():
                tasks.append(self._close_milvus_client(name, client))
            for name, vector_store in self.vector_stores.items():
                tasks.append(self._shutdown_vector_store(name, vector_store))
            await asyncio.gather(*tasks)
            self.graph_stores.clear()
            self.vector_stores.clear()
            self.neo4j_drivers.clear()
            self.sql_engines.clear()
            self.vector_store_sql_engines.clear()
            self.nebula_clients.clear()
            self.qdrant_clients.clear()
            self.milvus_clients.clear()
            self._neo4j_locks.clear()
            self._sql_locks.clear()
            self._nebula_locks.clear()
            self._qdrant_locks.clear()
            self._milvus_locks.clear()
            self._vector_store_locks.clear()

    @staticmethod
    async def _close_async_driver(name: str, driver: AsyncDriver) -> None:
        try:
            await driver.close()
        except Exception as ex:
            logger.warning("Error closing Neo4j driver '%s': %s", name, ex)

    @staticmethod
    async def _close_async_engine(name: str, engine: AsyncEngine) -> None:
        try:
            await engine.dispose()
        except Exception as ex:
            logger.warning("Error disposing SQL engine '%s': %s", name, ex)

    # --- Neo4j ---

    async def _build_neo4j(self) -> None:
        """
        Eagerly build all Neo4j drivers and graph stores.

        This simply calls the lazy initializer for each configured Neo4j instance.
        """
        tasks = [self.async_get_neo4j_driver(name) for name in self.conf.neo4j_confs]
        if tasks:
            await asyncio.gather(*tasks)

    @staticmethod
    def _build_neo4j_driver_kwargs(conf: Neo4jConf) -> dict[str, Any]:
        """Build keyword arguments for AsyncGraphDatabase.driver from config."""
        kwargs: dict[str, Any] = {
            "uri": conf.get_uri(),
            "auth": (conf.user, conf.password.get_secret_value()),
        }
        optional_fields = (
            "max_connection_pool_size",
            "connection_acquisition_timeout",
            "max_connection_lifetime",
            "liveness_check_timeout",
        )
        for field in optional_fields:
            value = getattr(conf, field)
            if value is not None:
                kwargs[field] = value
        return kwargs

    async def async_get_neo4j_driver(
        self, name: str, validate: bool = False
    ) -> AsyncDriver:
        """Return a Neo4j driver, creating it if necessary (lazy)."""
        if name not in self._neo4j_locks:
            async with self._lock:
                self._neo4j_locks.setdefault(name, Lock())

        async with self._neo4j_locks[name]:
            if name in self.neo4j_drivers:
                return self.neo4j_drivers[name]

            conf = self.conf.neo4j_confs.get(name)
            if not conf:
                raise ValueError(f"Neo4j config '{name}' not found.")

            driver = AsyncGraphDatabase.driver(**self._build_neo4j_driver_kwargs(conf))
            if validate:
                await self.validate_neo4j_driver(name, driver)
            self.neo4j_drivers[name] = driver
            params_kwargs: dict[str, Any] = {
                "driver": driver,
                "force_exact_similarity_search": conf.force_exact_similarity_search,
                "range_index_hierarchies": [["uid"], ["timestamp", "uid"]],
                # Without this the store's OperationTracker receives no factory and
                # every one of its timed operations is silently a no-op, which is
                # why no database latency was observable.
                "metrics_factory": conf.get_metrics_factory(),
            }
            if conf.range_index_creation_threshold is not None:
                params_kwargs["range_index_creation_threshold"] = (
                    conf.range_index_creation_threshold
                )
            if conf.vector_index_creation_threshold is not None:
                params_kwargs["vector_index_creation_threshold"] = (
                    conf.vector_index_creation_threshold
                )

            params = Neo4jVectorGraphStoreParams(**params_kwargs)
            self.graph_stores[name] = Neo4jVectorGraphStore(params)
            return driver

    def get_neo4j_driver(self, name: str) -> AsyncDriver:
        """Sync wrapper to get Neo4j driver lazily."""
        return asyncio.run(self.async_get_neo4j_driver(name, validate=True))

    async def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        """Return a vector graph store, auto-detecting Neo4j or NebulaGraph backend."""
        # Check if it's a Neo4j configuration
        if name in self.conf.neo4j_confs:
            await self.async_get_neo4j_driver(name, validate=True)
            return self.graph_stores[name]

        # Check if it's a NebulaGraph configuration
        if name in self.conf.nebula_graph_confs:
            await self.async_get_nebula_client(name, validate=True)
            return self.graph_stores[name]

        # Not found in either
        raise ValueError(
            f"VectorGraphStore '{name}' not found in neo4j_confs or nebula_graph_confs"
        )

    @staticmethod
    async def validate_neo4j_driver(name: str, driver: AsyncDriver) -> None:
        """Validate connectivity to a Neo4j instance."""
        try:
            logger.info("Validating Neo4j driver '%s'", name)
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
            logger.info("Neo4j driver '%s' validated successfully", name)
        except Exception as e:
            await driver.close()
            raise Neo4JConfigurationError(
                f"Neo4j config '{name}' failed verification: {e}",
            ) from e

        if not record or record["ok"] != 1:
            await driver.close()
            raise Neo4JConfigurationError(
                f"Verification failed for Neo4j config '{name}'",
            )

    async def _validate_neo4j_drivers(self) -> None:
        """Validate connectivity to each Neo4j instance."""
        for name, driver in self.neo4j_drivers.items():
            await self.validate_neo4j_driver(name, driver)

    # --- SQL ---

    async def _build_sql_engines(self) -> None:
        """
        Eagerly build all SQL engines.

        This simply calls the lazy initializer for each configured relational DB.
        """
        tasks = [
            self.async_get_sql_engine(name) for name in self.conf.relational_db_confs
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def async_get_sql_engine(
        self, name: str, validate: bool = False
    ) -> AsyncEngine:
        """Return a SQL engine, creating it if necessary (lazy)."""
        if name not in self._sql_locks:
            async with self._lock:
                self._sql_locks.setdefault(name, Lock())

        async with self._sql_locks[name]:
            if name in self.sql_engines:
                return self.sql_engines[name]

            conf = self.conf.relational_db_confs.get(name)
            if not conf:
                raise ValueError(f"SQL config '{name}' not found.")

            engine = create_async_engine(conf.uri, **_sql_engine_kwargs(conf))
            enable_sqlite_foreign_keys(engine)
            if validate:
                await self.validate_sql_engine(name, engine)
            self.sql_engines[name] = engine
            return engine

    def get_sql_engine(self, name: str) -> AsyncEngine:
        """Sync wrapper to get SQL engine lazily."""
        return asyncio.run(self.async_get_sql_engine(name, validate=True))

    @staticmethod
    async def validate_sql_engine(name: str, engine: AsyncEngine) -> None:
        """Validate connectivity for a single SQL engine."""
        try:
            logger.info("Validating SQL engine '%s'", name)
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1;"))
                row = result.fetchone()
            logger.info("SQL engine '%s' validated successfully", name)
        except Exception as e:
            raise SQLConfigurationError(
                f"SQL config '{name}' failed verification: {e}",
            ) from e

        if not row or row[0] != 1:
            raise SQLConfigurationError(
                f"Verification failed for SQL config '{name}'",
            )

    async def _validate_sql_engines(self) -> None:
        """Validate connectivity for each SQL engine."""
        for name, engine in self.sql_engines.items():
            await self.validate_sql_engine(name, engine)

    # --- NebulaGraph ---

    @staticmethod
    async def _close_nebula_client(name: str, client: "NebulaAsyncClient") -> None:
        try:
            await client.close()
        except Exception as ex:
            logger.warning("Error closing NebulaGraph client '%s': %s", name, ex)

    async def async_get_nebula_client(
        self, name: str, validate: bool = False
    ) -> "NebulaAsyncClient":
        """Return a NebulaGraph async client, creating it if necessary (lazy)."""
        if name not in self._nebula_locks:
            async with self._lock:
                self._nebula_locks.setdefault(name, Lock())

        async with self._nebula_locks[name]:
            if name in self.nebula_clients:
                return self.nebula_clients[name]

            conf = self.conf.nebula_graph_confs.get(name)
            if not conf:
                raise ValueError(f"NebulaGraph config '{name}' not found.")

            # Import at use site (not at module level) to make nebulagraph_python
            # an optional dependency - only required if NebulaGraph is actually used.
            # This avoids ImportError for users who only use Neo4j/PostgreSQL.
            from nebulagraph_python.client import (
                NebulaAsyncClient,
                SessionConfig,
                SessionPoolConfig,
            )

            # Create session config
            session_config = SessionConfig(
                schema=conf.schema_name,
                graph=conf.graph_name,
            )

            # Create session pool config
            session_pool_config = SessionPoolConfig(
                size=conf.session_pool_size,
                wait_timeout=conf.session_pool_wait_timeout
                if conf.session_pool_wait_timeout > 0
                else None,
            )

            # Connect to NebulaGraph
            client = await NebulaAsyncClient.connect(
                hosts=conf.get_hosts(),
                username=conf.username,
                password=conf.password.get_secret_value(),
                session_config=session_config,
                session_pool_config=session_pool_config,
            )

            # Initialize schema, graph type, and graph
            try:
                # Create schema if not exists
                await client.execute(f"CREATE SCHEMA IF NOT EXISTS {conf.schema_name}")
                logger.info("Ensured schema exists: %s", conf.schema_name)

                # Set session to the schema
                await client.execute(f"SESSION SET SCHEMA {conf.schema_name}")

                # Create empty graph type if not exists
                await client.execute(
                    f"CREATE GRAPH TYPE IF NOT EXISTS {conf.graph_type_name} AS {{}}"
                )
                logger.info("Ensured graph type exists: %s", conf.graph_type_name)

                # Create graph based on the graph type
                await client.execute(
                    f"CREATE GRAPH IF NOT EXISTS {conf.graph_name} TYPED {conf.graph_type_name}"
                )
                logger.info("Ensured graph exists: %s", conf.graph_name)

                # Set session to the graph
                await client.execute(f"SESSION SET GRAPH {conf.graph_name}")

            except Exception as e:
                await client.close()
                raise ValueError(
                    f"Failed to initialize NebulaGraph schema/graph for '{name}': {e}"
                ) from e

            if validate:
                await self.validate_nebula_client(name, client)

            self.nebula_clients[name] = client

            # Create and store VectorGraphStore
            # Import here to avoid circular dependency
            from memmachine_server.common.vector_graph_store.nebula_graph_vector_graph_store import (
                NebulaGraphVectorGraphStore,
                NebulaGraphVectorGraphStoreParams,
            )

            params_kwargs: dict[str, Any] = {
                "client": client,
                "schema_name": conf.schema_name,
                "graph_type_name": conf.graph_type_name,
                "graph_name": conf.graph_name,
                "force_exact_similarity_search": conf.force_exact_similarity_search,
                "ann_index_type": conf.ann_index_type,
                "ivf_nlist": conf.ivf_nlist,
                "ivf_nprobe": conf.ivf_nprobe,
                "hnsw_max_degree": conf.hnsw_max_degree,
                "hnsw_ef_construction": conf.hnsw_ef_construction,
                "hnsw_ef_search": conf.hnsw_ef_search,
            }
            if conf.range_index_creation_threshold is not None:
                params_kwargs["range_index_creation_threshold"] = (
                    conf.range_index_creation_threshold
                )
            if conf.vector_index_creation_threshold is not None:
                params_kwargs["vector_index_creation_threshold"] = (
                    conf.vector_index_creation_threshold
                )

            params = NebulaGraphVectorGraphStoreParams(**params_kwargs)
            self.graph_stores[name] = NebulaGraphVectorGraphStore(params)

            return client

    @staticmethod
    async def validate_nebula_client(name: str, client: "NebulaAsyncClient") -> None:
        """Validate connectivity to a NebulaGraph instance."""

        def _check_query_results(rows: list) -> None:
            """Check if query returned results."""
            if not rows:
                raise ValueError("Query returned no results")

        try:
            logger.info("Validating NebulaGraph client '%s'", name)
            result = await client.execute("RETURN 1 AS ok")

            # Extract first row using iteration (consistent with vector_graph_store usage)
            rows = list(result)
            _check_query_results(rows)

            row = rows[0]
            ok = row["ok"]
            # Normalize wrapped values: some versions return ValueWrapper, others return primitives
            ok_value = ok.cast_primitive() if hasattr(ok, "cast_primitive") else ok
        except Exception as e:
            await client.close()
            raise ValueError(
                f"NebulaGraph config '{name}' failed verification: {e}",
            ) from e

        if ok_value != 1:
            await client.close()
            raise ValueError(
                f"Verification failed for NebulaGraph config '{name}'",
            )
        logger.info("NebulaGraph client '%s' validated successfully", name)

    async def _validate_nebula_clients(self) -> None:
        """Validate connectivity to each NebulaGraph instance."""
        for name, client in self.nebula_clients.items():
            await self.validate_nebula_client(name, client)

    # --- Vector stores ---

    async def get_vector_store(
        self, name: str, *, indexed_properties: Mapping[str, PropertyType]
    ) -> VectorStore:
        """Return a vector store by name, built for the service declaring these keys.

        The schema a store is built with is its configured user keys plus the
        system keys of the one service that uses it, merged here on first
        use. Services do not share a vector store: a later request whose keys
        the store does not declare raises `VectorStoreConfigurationError`.
        """
        if name not in self._vector_store_locks:
            async with self._lock:
                self._vector_store_locks.setdefault(name, Lock())

        async with self._vector_store_locks[name]:
            store = self.vector_stores.get(name)
            if store is None:
                store = await self._build_vector_store(name, indexed_properties)
                self.vector_stores[name] = store
                return store
            for key, property_type in indexed_properties.items():
                if store.indexed_properties.get(key) is not property_type:
                    raise VectorStoreConfigurationError(
                        f"VectorStore '{name}' was built for another service and does "
                        f"not declare {key!r} as "
                        f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]}; "
                        "services do not share a vector store, so give each its own."
                    )
            return store

    async def _build_vector_store(
        self, name: str, system_properties: Mapping[str, PropertyType]
    ) -> VectorStore:
        if name in self.conf.qdrant_confs:
            conf = self.conf.qdrant_confs[name]
            return await self._build_qdrant_store(
                name, conf, self._declared_properties(name, conf, system_properties)
            )
        if name in self.conf.milvus_confs:
            conf = self.conf.milvus_confs[name]
            return await self._build_milvus_store(
                name, conf, self._declared_properties(name, conf, system_properties)
            )
        if name in self.conf.sqlite_vector_store_confs:
            conf = self.conf.sqlite_vector_store_confs[name]
            return await self._build_sqlite_vector_store(
                name, conf, self._declared_properties(name, conf, system_properties)
            )
        if name in self.conf.sqlite_vec_vector_store_confs:
            conf = self.conf.sqlite_vec_vector_store_confs[name]
            return await self._build_sqlite_vec_vector_store(
                name, conf, self._declared_properties(name, conf, system_properties)
            )
        raise ValueError(f"VectorStore '{name}' not found")

    @staticmethod
    def _declared_properties(
        name: str,
        conf: QdrantConf
        | MilvusConf
        | SQLiteVectorStoreConf
        | SQLiteVecVectorStoreConf,
        system_properties: Mapping[str, PropertyType],
    ) -> dict[str, PropertyType]:
        """The configured user keys plus the service's system keys, typed."""
        try:
            declared = _INDEXED_PROPERTIES.validate_python(conf.indexed_properties)
        except ValidationError as e:
            raise VectorStoreConfigurationError(
                f"VectorStore '{name}' has invalid indexed_properties: {e}"
            ) from e
        for key, property_type in system_properties.items():
            configured = declared.get(key)
            if configured is not None and configured is not property_type:
                raise VectorStoreConfigurationError(
                    f"VectorStore '{name}' configures {key!r} as "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[configured]}, but the "
                    f"service writes it as "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]}."
                )
            declared[key] = property_type
        return declared

    @staticmethod
    async def _shutdown_vector_store(name: str, vector_store: VectorStore) -> None:
        try:
            await vector_store.shutdown()
        except Exception as ex:
            logger.warning("Error shutting down VectorStore '%s': %s", name, ex)

    # --- Qdrant ---

    @staticmethod
    async def _close_qdrant_client(name: str, client: "AsyncQdrantClient") -> None:
        try:
            await client.close()
        except Exception as ex:
            logger.warning("Error closing Qdrant client '%s': %s", name, ex)

    async def async_get_qdrant_client(
        self, name: str, validate: bool = False
    ) -> "AsyncQdrantClient":
        """Return a Qdrant async client, creating it if necessary (lazy)."""
        if name not in self._qdrant_locks:
            async with self._lock:
                self._qdrant_locks.setdefault(name, Lock())

        async with self._qdrant_locks[name]:
            if name in self.qdrant_clients:
                return self.qdrant_clients[name]

            conf = self.conf.qdrant_confs.get(name)
            if not conf:
                raise ValueError(f"Qdrant config '{name}' not found.")

            # Import at use site (not at module level) to make qdrant-client
            # an optional dependency — only required if Qdrant is actually used.
            from qdrant_client import AsyncQdrantClient

            client_kwargs: dict[str, Any] = {
                "host": conf.host,
                "port": conf.port,
                "grpc_port": conf.grpc_port,
                "prefer_grpc": conf.prefer_grpc,
                "https": conf.https,
                "timeout": conf.request_timeout,
            }
            if conf.api_key.get_secret_value():
                client_kwargs["api_key"] = conf.api_key.get_secret_value()

            client = AsyncQdrantClient(**client_kwargs)

            if validate:
                await self.validate_qdrant_client(name, client)

            self.qdrant_clients[name] = client
            return client

    async def _build_qdrant_store(
        self,
        name: str,
        conf: QdrantConf,
        indexed_properties: Mapping[str, PropertyType],
    ) -> VectorStore:
        client = await self.async_get_qdrant_client(name, validate=True)

        from memmachine_server.common.vector_store.qdrant_vector_store import (
            QdrantVectorStore,
            QdrantVectorStoreParams,
        )

        # QdrantConf carries the native index and quantization settings as
        # plain mappings, so qdrant-client stays optional for config parsing;
        # the params model validates them against qdrant's own models.
        try:
            params = QdrantVectorStoreParams.model_validate(
                {
                    "client": client,
                    "is_distributed": conf.is_distributed,
                    "registry_replication_factor": conf.registry_replication_factor,
                    "indexed_properties": indexed_properties,
                    "hnsw_config": conf.hnsw_config,
                    "optimizers_config": conf.optimizers_config,
                    "quantization_config": conf.quantization_config,
                    "metrics_factory": conf.get_metrics_factory(),
                }
            )
        except ValidationError as e:
            raise QdrantConfigurationError(
                f"Qdrant config '{name}' is invalid: {e}"
            ) from e
        store = QdrantVectorStore(params)
        await store.startup()
        return store

    @staticmethod
    async def validate_qdrant_client(name: str, client: "AsyncQdrantClient") -> None:
        """Validate connectivity to a Qdrant instance."""
        try:
            logger.info("Validating Qdrant client '%s'", name)
            await client.get_collections()
            logger.info("Qdrant client '%s' validated successfully", name)
        except Exception as e:
            await client.close()
            raise QdrantConfigurationError(
                f"Qdrant config '{name}' failed verification: {e}",
            ) from e

    async def _validate_qdrant_clients(self) -> None:
        """Validate connectivity to each Qdrant instance."""
        for name, client in self.qdrant_clients.items():
            await self.validate_qdrant_client(name, client)

    # --- Milvus ---

    @staticmethod
    async def _close_milvus_client(name: str, client: "MilvusClient") -> None:
        try:
            await asyncio.to_thread(client.close)
        except Exception as ex:
            logger.warning("Error closing Milvus client '%s': %s", name, ex)

    async def async_get_milvus_client(
        self, name: str, validate: bool = False
    ) -> "MilvusClient":
        """Return a Milvus client, creating it if necessary (lazy)."""
        if name not in self._milvus_locks:
            async with self._lock:
                self._milvus_locks.setdefault(name, Lock())

        async with self._milvus_locks[name]:
            if name in self.milvus_clients:
                return self.milvus_clients[name]

            conf = self.conf.milvus_confs.get(name)
            if not conf:
                raise ValueError(f"Milvus config '{name}' not found.")

            from pymilvus import MilvusClient

            client_kwargs: dict[str, Any] = {
                "uri": conf.uri,
                "timeout": conf.request_timeout,
            }
            token = conf.token.get_secret_value()
            if token:
                client_kwargs["token"] = token
            if conf.db_name:
                client_kwargs["db_name"] = conf.db_name

            client = MilvusClient(**client_kwargs)

            if validate:
                await self.validate_milvus_client(name, client)

            self.milvus_clients[name] = client
            return client

    async def _build_milvus_store(
        self,
        name: str,
        conf: MilvusConf,
        indexed_properties: Mapping[str, PropertyType],
    ) -> VectorStore:
        client = await self.async_get_milvus_client(name, validate=True)

        from memmachine_server.common.vector_store.milvus_vector_store import (
            MilvusVectorStore,
            MilvusVectorStoreParams,
        )

        params = MilvusVectorStoreParams(
            client=client,
            consistency_level=conf.consistency_level,
            indexed_properties=indexed_properties,
        )
        store = MilvusVectorStore(params)
        await store.startup()
        return store

    @staticmethod
    async def validate_milvus_client(name: str, client: "MilvusClient") -> None:
        """Validate connectivity to a Milvus instance."""
        try:
            logger.info("Validating Milvus client '%s'", name)
            await asyncio.to_thread(client.list_collections)
            logger.info("Milvus client '%s' validated successfully", name)
        except Exception as e:
            await asyncio.to_thread(client.close)
            raise MilvusConfigurationError(
                f"Milvus config '{name}' failed verification: {e}",
            ) from e

    async def _validate_milvus_clients(self) -> None:
        """Validate connectivity to each Milvus instance."""
        for name, client in self.milvus_clients.items():
            await self.validate_milvus_client(name, client)

    # --- SQLite-backed VectorStores ---

    @staticmethod
    def _make_sqlite_search_engine_factory(
        conf: SQLiteVectorStoreConf,
    ) -> Callable[[int], VectorSearchEngine]:
        """Build a ndim -> VectorSearchEngine factory from config.

        Imports are deferred so the engine packages remain optional unless
        their backend is actually used.
        """
        match conf.vector_search_engine:
            case SQLiteVectorStoreEngine.USEARCH:
                from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
                    USearchVectorSearchEngine,
                )

                def usearch_factory(num_dimensions: int) -> VectorSearchEngine:
                    return USearchVectorSearchEngine(num_dimensions=num_dimensions)

                return usearch_factory
            case SQLiteVectorStoreEngine.HNSWLIB:
                from memmachine_server.common.vector_store.vector_search_engine.hnswlib_engine import (
                    HnswlibVectorSearchEngine,
                )

                def hnswlib_factory(num_dimensions: int) -> VectorSearchEngine:
                    return HnswlibVectorSearchEngine(num_dimensions=num_dimensions)

                return hnswlib_factory

    async def _build_sqlite_vector_store(
        self,
        name: str,
        conf: SQLiteVectorStoreConf,
        indexed_properties: Mapping[str, PropertyType],
    ) -> VectorStore:
        from memmachine_server.common.vector_store.sqlite_vector_store import (
            SQLiteVectorStore,
            SQLiteVectorStoreParams,
        )

        engine = create_async_engine(f"sqlite+aiosqlite:///{conf.path}")
        self.vector_store_sql_engines[name] = engine

        try:
            store = SQLiteVectorStore(
                SQLiteVectorStoreParams(
                    sqlalchemy_engine=engine,
                    vector_search_engine_factory=self._make_sqlite_search_engine_factory(
                        conf
                    ),
                    index_directory=conf.index_directory,
                    save_threshold=conf.save_threshold,
                    indexed_properties=indexed_properties,
                )
            )
            await store.startup()
        except Exception as e:
            await engine.dispose()
            self.vector_store_sql_engines.pop(name, None)
            raise VectorStoreConfigurationError(
                f"SQLiteVectorStore '{name}' failed to start: {e}",
            ) from e
        return store

    async def _build_sqlite_vec_vector_store(
        self,
        name: str,
        conf: SQLiteVecVectorStoreConf,
        indexed_properties: Mapping[str, PropertyType],
    ) -> VectorStore:
        from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
            SQLiteVecVectorStore,
            SQLiteVecVectorStoreParams,
        )

        engine = create_async_engine(f"sqlite+aiosqlite:///{conf.path}")
        self.vector_store_sql_engines[name] = engine

        try:
            store = SQLiteVecVectorStore(
                SQLiteVecVectorStoreParams(
                    engine=engine, indexed_properties=indexed_properties
                )
            )
            await store.startup()
        except Exception as e:
            await engine.dispose()
            self.vector_store_sql_engines.pop(name, None)
            raise VectorStoreConfigurationError(
                f"SQLiteVecVectorStore '{name}' failed to start: {e}",
            ) from e
        return store
