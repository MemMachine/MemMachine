import functools
from typing import Self

import asyncpg
from asyncpg import Pool
from neo4j import AsyncGraphDatabase
from pgvector.asyncpg import register_vector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sympy import false

from memmachine.common.configuration.storage_conf import StorageConf
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)


class StorageMgr:
    def __init__(self, conf: StorageConf):
        self.conf = conf
        self.graph_stores: dict[str, VectorGraphStore] = {}
        self.postgres_pools: dict[str, Pool] = {}
        self.sql_engines: dict[str, AsyncEngine] = {}

    def build_all(self, validate=false) -> Self:
        """Build and verify all configured database connections."""
        self._build_neo4j()
        self._build_postgres()
        self._build_sql_engines()
        if validate:
            self._validate_neo4j()
            self._validate_postgres()
            self._validate_sql_engines()
        return self

    def close(self):
        """Close all database connections."""
        for store in self.graph_stores.values():
            store.close()
        for pool in self.postgres_pools.values():
            pool.close()
        for engine in self.sql_engines.values():
            engine.dispose()

    def get_graph_store(self, name: str) -> VectorGraphStore:
        if name not in self.graph_stores:
            raise ValueError(f"Neo4J driver '{name}' not found.")
        return self.graph_stores[name]

    def get_postgres(self, name: str) -> Pool:
        if name not in self.postgres_pools:
            raise ValueError(f"Postgres pool '{name}' not found.")
        return self.postgres_pools[name]

    def get_sql_engine(self, name: str) -> AsyncEngine:
        if name not in self.sql_engines:
            raise ValueError(f"SQLite connection '{name}' not found.")
        return self.sql_engines[name]

    async def _build_neo4j(self):
        for name, conf in self.conf.neo4j_confs.items():
            neo4j_host = conf.host
            if "neo4j+s://" in neo4j_host:
                neo4j_uri = neo4j_host
            else:
                neo4j_port = conf.port
                neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

            driver = AsyncGraphDatabase.driver(
                neo4j_uri, auth=(conf.user, conf.password)
            )
            params = Neo4jVectorGraphStoreParams(
                driver=driver,
                max_concurrent_transactions=conf.max_concurrent_transactions,
                force_exact_similarity_search=conf.force_exact_similarity_search,
            )
            self.graph_stores[name] = Neo4jVectorGraphStore(params)

    async def _validate_neo4j(self):
        for name, driver in self.conf.neo4j_confs.items():
            try:
                async with driver.session() as session:
                    result = await session.run("RETURN 1 AS ok")
                    record = await result.single()
                    if not record or record["ok"] != 1:
                        raise ConnectionError(
                            f"Verification failed for Neo4J config '{name}'"
                        )
            except Exception as e:
                await driver.close()
                raise ConnectionError(f"Neo4J config '{name}' failed verification: {e}")

    async def _build_postgres(self):
        for name, conf in self.conf.postgres_confs.items():
            pool = asyncpg.create_pool(
                host=conf.host,
                port=conf.port,
                user=conf.user,
                password=conf.password.get_secret_value(),
                database=conf.db_name,
                init=functools.partial(
                    register_vector,
                    schema=conf.vector_schema,
                ),
                statement_cache_size=conf.statement_cache_size,
            )
            self.postgres_pools[name] = pool

    async def _validate_postgres(self):
        for name, pool in self.postgres_pools.items():
            try:
                async with pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1;")
                    if result != 1:
                        raise ConnectionError(
                            f"Verification failed for Postgres config '{name}'"
                        )
            except Exception as e:
                await pool.close()
                raise ConnectionError(
                    f"Postgres config '{name}' failed verification: {e}"
                )

    async def _build_sql_engines(self):
        schema = "sqlite+aiosqlite:///"
        for name, conf in self.conf.sqlite_confs.items():
            db_url = conf.file_path
            if not conf.file_path.startswith(schema):
                db_url = f"sqlite+aiosqlite:///{db_url}"
            engine = create_async_engine(db_url, echo=False, future=True)
            self.sql_engines[name] = engine

    async def _validate_sql_engines(self):
        for name, engine in self.sql_engines.items():
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1;"))
                    row = result.fetchone()
                    if not row or row[0] != 1:
                        raise ConnectionError(
                            f"Verification failed for SQLite config '{name}'"
                        )
            except Exception as e:
                raise ConnectionError(
                    f"SQLite config '{name}' failed verification: {e}"
                )
