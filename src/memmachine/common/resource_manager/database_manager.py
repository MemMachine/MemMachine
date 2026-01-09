"""Manage database engines for SQL and Neo4j backends."""

import asyncio
import logging
from asyncio import Lock
from typing import Any, Self

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

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Create and manage database backends with lazy initialization."""

    def __init__(self, conf: DatabasesConf) -> None:
        """Initialize with database configuration."""
        self.conf = conf
        self.graph_stores: dict[str, VectorGraphStore] = {}
        self.sql_engines: dict[str, AsyncEngine] = {}
        self.neo4j_drivers: dict[str, AsyncDriver] = {}

        self._lock = Lock()
        self._neo4j_locks: dict[str, Lock] = {}
        self._sql_locks: dict[str, Lock] = {}

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
        # Lazy build will occur in get_* calls, but build_all can trigger them
        tasks = neo4j_tasks + relation_db_tasks
        await asyncio.gather(*tasks)

        if validate:
            await asyncio.gather(
                self._validate_neo4j_drivers(),
                self._validate_sql_engines(),
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
            await asyncio.gather(*tasks)
            self.graph_stores.clear()
            self.neo4j_drivers.clear()
            self.sql_engines.clear()
            self._neo4j_locks.clear()
            self._sql_locks.clear()

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

            driver = AsyncGraphDatabase.driver(
                conf.get_uri(),
                auth=(conf.user, conf.password.get_secret_value()),
            )
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
        """Return a vector graph store, initializing driver lazily if needed."""
        await self.async_get_neo4j_driver(name, validate=True)
        return self.graph_stores[name]

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
