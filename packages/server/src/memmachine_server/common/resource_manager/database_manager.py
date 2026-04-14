"""Manage database engines for SQL, Neo4j, NebulaGraph, and Apache AGE backends."""

import asyncio
import logging
from asyncio import Lock
from typing import TYPE_CHECKING, Any, Self

from neo4j import AsyncDriver, AsyncGraphDatabase
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.age_utils import setup_age_sync_connection
from memmachine_server.common.configuration.database_conf import (
    AgeConf,
    DatabasesConf,
    Neo4jConf,
)
from memmachine_server.common.errors import (
    Neo4JConfigurationError,
    SQLConfigurationError,
)
from memmachine_server.common.vector_graph_store import VectorGraphStore
from memmachine_server.common.vector_graph_store.age_vector_graph_store import (
    AgeVectorGraphStore,
    AgeVectorGraphStoreParams,
)
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)

# TYPE_CHECKING is True only when type checkers (mypy, pyright) run, False at runtime.
# This allows type hints without requiring nebulagraph_python to be installed
# unless NebulaGraph is actually used. The actual import happens at use site.
if TYPE_CHECKING:
    from nebulagraph_python.client import NebulaAsyncClient

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Create and manage database backends with lazy initialization."""

    def __init__(self, conf: DatabasesConf) -> None:
        """Initialize with database configuration."""
        self.conf = conf
        self.graph_stores: dict[str, VectorGraphStore] = {}
        self.sql_engines: dict[str, AsyncEngine] = {}
        self.neo4j_drivers: dict[str, AsyncDriver] = {}
        # String annotation "NebulaAsyncClient" (forward reference) because the type
        # is only imported under TYPE_CHECKING and doesn't exist at runtime.
        # Type checkers see it, but runtime treats it as a string literal.
        self.nebula_clients: dict[str, NebulaAsyncClient] = {}
        self.age_engines: dict[str, AsyncEngine] = {}

        self._lock = Lock()
        self._neo4j_locks: dict[str, Lock] = {}
        self._sql_locks: dict[str, Lock] = {}
        self._nebula_locks: dict[str, Lock] = {}
        self._age_locks: dict[str, Lock] = {}

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
        age_tasks = [
            self.async_get_age_engine(name, validate=validate)
            for name in self.conf.age_confs
        ]
        # Lazy build will occur in get_* calls, but build_all can trigger them
        tasks = neo4j_tasks + relation_db_tasks + nebula_tasks + age_tasks
        await asyncio.gather(*tasks)

        if validate:
            await asyncio.gather(
                self._validate_neo4j_drivers(),
                self._validate_sql_engines(),
                self._validate_nebula_clients(),
                self._validate_age_engines(),
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
            for name, client in self.nebula_clients.items():
                tasks.append(self._close_nebula_client(name, client))
            for name, engine in self.age_engines.items():
                tasks.append(self._close_async_engine(name, engine))
            await asyncio.gather(*tasks)
            self.graph_stores.clear()
            self.neo4j_drivers.clear()
            self.sql_engines.clear()
            self.nebula_clients.clear()
            self.age_engines.clear()
            self._neo4j_locks.clear()
            self._sql_locks.clear()
            self._nebula_locks.clear()
            self._age_locks.clear()

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
        """Return a vector graph store, auto-detecting the configured backend."""
        # Check if it's a Neo4j configuration
        if name in self.conf.neo4j_confs:
            await self.async_get_neo4j_driver(name, validate=True)
            return self.graph_stores[name]

        # Check if it's a NebulaGraph configuration
        if name in self.conf.nebula_graph_confs:
            await self.async_get_nebula_client(name, validate=True)
            return self.graph_stores[name]

        # Check if it's an Apache AGE configuration
        if name in self.conf.age_confs:
            await self.async_get_age_engine(name, validate=True)
            return self.graph_stores[name]

        # Not found in any
        raise ValueError(
            f"VectorGraphStore '{name}' not found in neo4j_confs, "
            "nebula_graph_confs, or age_confs"
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

            engine_kwargs: dict[str, bool | int] = {
                "echo": False,
                "future": True,
            }
            if conf.pool_size is not None:
                engine_kwargs["pool_size"] = conf.pool_size
            if conf.max_overflow is not None:
                engine_kwargs["max_overflow"] = conf.max_overflow
            if conf.pool_timeout is not None:
                engine_kwargs["pool_timeout"] = conf.pool_timeout
            if conf.pool_recycle is not None:
                engine_kwargs["pool_recycle"] = conf.pool_recycle
            if conf.pool_pre_ping is not None:
                engine_kwargs["pool_pre_ping"] = conf.pool_pre_ping

            engine = create_async_engine(conf.uri, **engine_kwargs)
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

    # --- Apache AGE ---

    async def async_get_age_engine(
        self, name: str, validate: bool = False
    ) -> AsyncEngine:
        """Return an AGE-enabled SQLAlchemy engine, creating it if necessary."""
        if name not in self._age_locks:
            async with self._lock:
                self._age_locks.setdefault(name, Lock())

        async with self._age_locks[name]:
            if name in self.age_engines:
                return self.age_engines[name]

            conf = self.conf.age_confs.get(name)
            if not conf:
                raise ValueError(f"AGE config '{name}' not found.")

            engine = self._build_age_engine(conf)

            if validate:
                await self.validate_age_engine(name, engine)

            self.age_engines[name] = engine

            params_kwargs: dict[str, Any] = {
                "engine": engine,
                "graph_name": conf.graph_name,
                "force_exact_similarity_search": conf.force_exact_similarity_search,
                "range_index_hierarchies": [["uid"], ["timestamp", "uid"]],
                "hnsw_m": conf.hnsw_m,
                "hnsw_ef_construction": conf.hnsw_ef_construction,
            }
            if conf.range_index_creation_threshold is not None:
                params_kwargs["range_index_creation_threshold"] = (
                    conf.range_index_creation_threshold
                )
            if conf.vector_index_creation_threshold is not None:
                params_kwargs["vector_index_creation_threshold"] = (
                    conf.vector_index_creation_threshold
                )

            params = AgeVectorGraphStoreParams(**params_kwargs)
            self.graph_stores[name] = AgeVectorGraphStore(params)
            return engine

    @staticmethod
    def _build_age_engine(conf: AgeConf) -> AsyncEngine:
        """Build an async SQLAlchemy engine and wire the per-connection setup."""
        engine_kwargs: dict[str, bool | int] = {
            "echo": False,
            "future": True,
        }
        if conf.pool_size is not None:
            engine_kwargs["pool_size"] = conf.pool_size
        if conf.max_overflow is not None:
            engine_kwargs["max_overflow"] = conf.max_overflow
        if conf.pool_timeout is not None:
            engine_kwargs["pool_timeout"] = conf.pool_timeout
        if conf.pool_recycle is not None:
            engine_kwargs["pool_recycle"] = conf.pool_recycle
        if conf.pool_pre_ping is not None:
            engine_kwargs["pool_pre_ping"] = conf.pool_pre_ping

        engine = create_async_engine(conf.uri, **engine_kwargs)
        _register_age_connect_hook(engine)
        return engine

    @staticmethod
    async def validate_age_engine(name: str, engine: AsyncEngine) -> None:
        """Validate connectivity for an AGE engine."""
        try:
            logger.info("Validating AGE engine '%s'", name)
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1 AS ok"))
                row = result.fetchone()
        except Exception as e:
            await engine.dispose()
            raise SQLConfigurationError(
                f"AGE config '{name}' failed verification: {e}",
            ) from e

        if not row or row[0] != 1:
            await engine.dispose()
            raise SQLConfigurationError(
                f"Verification failed for AGE config '{name}'",
            )
        logger.info("AGE engine '%s' validated successfully", name)

    async def _validate_age_engines(self) -> None:
        """Validate connectivity for each AGE engine."""
        for name, engine in self.age_engines.items():
            await self.validate_age_engine(name, engine)


def _register_age_connect_hook(engine: AsyncEngine) -> None:
    """Wire per-connection AGE setup onto a SQLAlchemy async engine.

    Extracted as a module-level function so tests can patch this seam
    directly instead of patching ``sqlalchemy.event.listens_for``, which
    couples the test to internal SQLAlchemy plumbing.

    AGE's session state (extension load, search_path) is per-connection, so
    every newly created physical connection must run the setup before any
    Cypher query.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(
        dbapi_connection: Any,  # noqa: ANN401
        connection_record: Any,  # noqa: ANN401, ARG001
    ) -> None:
        setup_age_sync_connection(dbapi_connection)
