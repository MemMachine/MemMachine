"""Read-only backend probes with explicit client and worker ownership."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from neo4j import AsyncGraphDatabase, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.configuration import (
    Configuration,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.resource_manager.readiness import (
    StorageProbe,
    UnsupportedProbeError,
)

if TYPE_CHECKING:
    from memmachine_server.common.resource_manager.database_manager import (
        DatabaseManager,
    )


class StorageProbes:
    """Select enabled storage dependencies and reuse serving clients when present.

    Episodic projects may override their backend/resource IDs. While episodic
    long-term memory is enabled, the configured storage catalog is therefore
    required, including alternatives not yet used by this worker. Disabled
    memory services do not require their otherwise-unused storage. Embedders
    and language models are deliberately not called by readiness probes.
    """

    def __init__(self, conf: Configuration, databases: "DatabaseManager") -> None:
        """Retain configuration and borrowed resource-manager references."""
        self._conf = conf
        self._databases = databases
        self._timeout = 2
        self._workers = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="memmachine-readiness"
        )

    def selected(self) -> list[StorageProbe]:
        """Build a fresh catalog snapshot, with no connection or initialization I/O."""
        db = self._databases.conf
        catalogs = {
            "sql": db.relational_db_confs,
            "neo4j": db.neo4j_confs,
            "nebula": db.nebula_graph_confs,
            "qdrant": db.qdrant_confs,
            "milvus": db.milvus_confs,
            "sqlite_vector": db.sqlite_vector_store_confs,
            "sqlite_vec": db.sqlite_vec_vector_store_confs,
        }
        selected = {
            self._conf.session_manager.database,
            self._conf.episode_store.database,
        }
        episodic = self._conf.episodic_memory
        if (
            episodic.enabled is not False
            and episodic.long_term_memory_enabled is not False
        ):
            selected.update(name for catalog in catalogs.values() for name in catalog)
        semantic = self._conf.semantic_memory
        if semantic.enabled:
            selected.add(semantic.config_database or "")
            if semantic.storage_backend == SemanticMemoryStorageBackend.VECTOR_STORE:
                selected.update(
                    (semantic.feature_store or "", semantic.vector_collection or "")
                )
            else:
                selected.add(semantic.database or "")
        result = []
        known = set()
        for kind, catalog in catalogs.items():
            names = sorted(selected.intersection(catalog))
            known.update(names)
            for index, name in enumerate(names, start=1):
                result.append(
                    StorageProbe(
                        (kind, name),
                        f"{kind}_{index}",
                        partial(self.check, kind, name),
                    )
                )
        if selected - known:
            raise ValueError("A required storage resource is not configured")
        return result

    async def check(self, kind: str, name: str) -> None:
        """Perform a fresh operation; never invoke construction-time validators."""
        if kind == "milvus":
            # Shield the worker future: coroutine cancellation cannot stop a
            # running SDK call. The coordinator retains this task until the
            # worker really finishes, and there are only two worker threads.
            future = asyncio.get_running_loop().run_in_executor(
                self._workers, self._milvus, name
            )
            await asyncio.shield(future)
            return
        checks = {
            "sql": self._sql,
            "neo4j": self._neo4j,
            "nebula": self._nebula,
            "qdrant": self._qdrant,
            "sqlite_vector": partial(self._sqlite_vector, "sqlite_vector"),
            "sqlite_vec": partial(self._sqlite_vector, "sqlite_vec"),
        }
        if kind in {"qdrant", "neo4j", "nebula"}:
            async with asyncio.timeout(self._timeout):
                await checks[kind](name)
        else:
            await checks[kind](name)

    async def _select_one(self, engine: AsyncEngine) -> None:
        async with asyncio.timeout(self._timeout), engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))
            if result.scalar() != 1:
                raise ValueError("Storage probe returned an unexpected result")

    async def _sql(self, name: str) -> None:
        engine = self._databases.sql_engines.get(name)
        if engine is not None:
            await self._select_one(engine)
            return
        conf = self._databases.conf.relational_db_confs[name]
        if conf.dialect == "sqlite":
            await self._sqlite_path(conf.path or "")
            return
        if conf.dialect != "postgresql" or conf.driver != "asyncpg":
            raise UnsupportedProbeError("No bounded probe for this SQL driver")
        owned = create_async_engine(
            conf.uri,
            connect_args={
                "timeout": self._timeout,
                "command_timeout": self._timeout,
            },
        )
        try:
            await self._select_one(owned)
        finally:
            await owned.dispose()

    async def _sqlite_vector(self, kind: str, name: str) -> None:
        engine = self._databases.vector_store_sql_engines.get(name)
        if engine is not None:
            await self._select_one(engine)
            return
        catalog = (
            self._databases.conf.sqlite_vector_store_confs
            if kind == "sqlite_vector"
            else self._databases.conf.sqlite_vec_vector_store_confs
        )
        await self._sqlite_path(catalog[name].path)

    async def _sqlite_path(self, path: str) -> None:
        future = asyncio.get_running_loop().run_in_executor(
            self._workers, self._sqlite_uri, path
        )
        uri = await asyncio.shield(future)
        if uri is None:
            return
        owned = create_async_engine(uri, connect_args={"timeout": self._timeout})
        try:
            await self._select_one(owned)
        finally:
            await owned.dispose()

    @staticmethod
    def _sqlite_uri(path: str) -> str | None:
        if path == ":memory:":
            return "sqlite+aiosqlite:///:memory:"
        location = Path(path).resolve()
        if not location.exists():
            # Like a remote server with no collections yet, a local store
            # need not have been initialized. Check that its configured
            # directory can host the store without creating a database.
            if not os.access(location.parent, os.W_OK | os.X_OK):
                raise OSError("Local storage directory is unavailable")
            return None
        # mode=ro prevents a readiness request from creating a database.
        return f"sqlite+aiosqlite:///{location.as_uri()}?mode=ro&uri=true"

    async def _neo4j(self, name: str) -> None:
        driver = self._databases.neo4j_drivers.get(name)
        borrowed = driver is not None
        if driver is None:
            conf = self._databases.conf.neo4j_confs[name]
            driver = AsyncGraphDatabase.driver(
                conf.get_uri(),
                auth=(conf.user, conf.password.get_secret_value()),
                connection_timeout=self._timeout,
                connection_acquisition_timeout=self._timeout,
            )
        try:
            async with driver.session() as session:
                result = await session.run(
                    Query("RETURN 1 AS ok", timeout=self._timeout)
                )
                record = await result.single()
                if record is None or record["ok"] != 1:
                    raise ValueError("Storage probe returned an unexpected result")
        finally:
            if not borrowed:
                await driver.close()

    async def _qdrant(self, name: str) -> None:
        client = self._databases.qdrant_clients.get(name)
        borrowed = client is not None
        if client is None:
            from qdrant_client import AsyncQdrantClient

            conf = self._databases.conf.qdrant_confs[name]
            client = AsyncQdrantClient(
                host=conf.host,
                port=conf.port,
                grpc_port=conf.grpc_port,
                prefer_grpc=conf.prefer_grpc,
                https=conf.https,
                api_key=conf.api_key.get_secret_value() or None,
                timeout=self._timeout,
                check_compatibility=False,
            )
        try:
            await client.get_collections()
        finally:
            if not borrowed:
                await client.close()

    async def _nebula(self, name: str) -> None:
        client = self._databases.nebula_clients.get(name)
        borrowed = client is not None
        if client is None:
            from nebulagraph_python.client import (
                ConnectionConfig,
                NebulaAsyncClient,
                SessionConfig,
                SessionPoolConfig,
            )

            conf = self._databases.conf.nebula_graph_confs[name]
            client = await NebulaAsyncClient.connect(
                hosts=conf.get_hosts(),
                username=conf.username,
                password=conf.password.get_secret_value(),
                conn_config=ConnectionConfig.from_defaults(
                    conf.get_hosts(),
                    connect_timeout=self._timeout,
                    request_timeout=self._timeout,
                ),
                session_config=SessionConfig(
                    schema=conf.schema_name, graph=conf.graph_name
                ),
                session_pool_config=SessionPoolConfig(
                    size=1, wait_timeout=self._timeout
                ),
            )
        try:
            result = await client.execute("RETURN 1 AS ok", timeout=self._timeout)
            row = next(iter(result), None)
            if row is None:
                raise ValueError("Storage probe returned no result")
            value = row["ok"]
            if hasattr(value, "cast_primitive"):
                value = value.cast_primitive()
            if value != 1:
                raise ValueError("Storage probe returned an unexpected result")
        finally:
            if not borrowed:
                await client.close()

    def _milvus(self, name: str) -> None:
        client = self._databases.milvus_clients.get(name)
        borrowed = client is not None
        if client is None:
            conf = self._databases.conf.milvus_confs[name]
            if "://" not in conf.uri:
                # Constructing a cold Milvus Lite client starts a local server.
                raise UnsupportedProbeError("Milvus Lite must be initialized first")
            from pymilvus import MilvusClient

            client = MilvusClient(
                uri=conf.uri,
                token=conf.token.get_secret_value(),
                db_name=conf.db_name,
                timeout=self._timeout,
                dedicated=True,
            )
        try:
            client.list_collections(timeout=self._timeout, retry_on_rate_limit=False)
        finally:
            if not borrowed:
                client.close()

    def close(self) -> None:
        """Stop admitting owned worker jobs; active SDK calls retain their deadlines."""
        self._workers.shutdown(wait=False, cancel_futures=True)
