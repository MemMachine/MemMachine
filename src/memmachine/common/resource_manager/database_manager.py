"""Manage database engines for SQL, Neo4j, and NebulaGraph backends."""

import asyncio
import logging
from asyncio import Lock
from typing import TYPE_CHECKING, Any, Self

from neo4j import AsyncDriver, AsyncGraphDatabase
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine.common.configuration.database_conf import DatabasesConf
from memmachine.common.errors import Neo4JConfigurationError, SQLConfigurationError
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
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
        self.nebula_clients: dict[str, "NebulaAsyncClient"] = {}

        self._lock = Lock()
        self._neo4j_locks: dict[str, Lock] = {}
        self._sql_locks: dict[str, Lock] = {}
        self._nebula_locks: dict[str, Lock] = {}

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
        # Lazy build will occur in get_* calls, but build_all can trigger them
        tasks = neo4j_tasks + relation_db_tasks + nebula_tasks
        await asyncio.gather(*tasks)

        if validate:
            await asyncio.gather(
                self._validate_neo4j_drivers(),
                self._validate_sql_engines(),
                self._validate_nebula_clients(),
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
            await asyncio.gather(*tasks)
            self.graph_stores.clear()
            self.neo4j_drivers.clear()
            self.sql_engines.clear()
            self.nebula_clients.clear()
            self._neo4j_locks.clear()
            self._sql_locks.clear()
            self._nebula_locks.clear()

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

            driver_kwargs: dict[str, Any] = {
                "uri": conf.get_uri(),
                "auth": (conf.user, conf.password.get_secret_value()),
            }
            if conf.max_connection_pool_size is not None:
                driver_kwargs["max_connection_pool_size"] = (
                    conf.max_connection_pool_size
                )
            if conf.connection_acquisition_timeout is not None:
                driver_kwargs["connection_acquisition_timeout"] = (
                    conf.connection_acquisition_timeout
                )

            driver = AsyncGraphDatabase.driver(**driver_kwargs)
            if validate:
                await self.validate_neo4j_driver(name, driver)
            self.neo4j_drivers[name] = driver
            params_kwargs: dict[str, Any] = {
                "driver": driver,
                "force_exact_similarity_search": conf.force_exact_similarity_search,
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

            engine_kwargs = {
                "echo": False,
                "future": True,
            }
            if conf.pool_size is not None:
                engine_kwargs["pool_size"] = conf.pool_size
            if conf.max_overflow is not None:
                engine_kwargs["max_overflow"] = conf.max_overflow

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
            from memmachine.common.vector_graph_store.nebula_graph_vector_graph_store import (
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
        try:
            logger.info("Validating NebulaGraph client '%s'", name)
            result = await client.execute("RETURN 1 AS ok")
            row = result.one()
            ok = row["ok"]
            # Normalize wrapped values: some versions return ValueWrapper, others return primitives
            ok_value = ok.cast_primitive() if hasattr(ok, "cast_primitive") else ok
            logger.info("NebulaGraph client '%s' validated successfully", name)
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

    async def _validate_nebula_clients(self) -> None:
        """Validate connectivity to each NebulaGraph instance."""
        for name, client in self.nebula_clients.items():
            await self.validate_nebula_client(name, client)
