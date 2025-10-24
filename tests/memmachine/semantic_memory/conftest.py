import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from memmachine.semantic_memory.storage.asyncpg_profile import AsyncPgSemanticStorage
from memmachine.semantic_memory.storage.syncschema import sync_to as setup_pg_schema
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    InMemorySemanticStorage,
)


@pytest_asyncio.fixture
async def in_memory_profile_storage():
    store = InMemorySemanticStorage()
    await store.startup()
    yield store
    await store.cleanup()


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest_asyncio.fixture(scope="session")
async def pg_server(pg_container):
    host = pg_container.get_container_host_ip()
    port = int(pg_container.get_exposed_port(5432))
    database = pg_container.dbname
    user = pg_container.username
    password = pg_container.password

    await setup_pg_schema(
        database=database,
        host=host,
        port=f"{port}",
        user=user,
        password=password,
    )

    yield {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }


@pytest_asyncio.fixture
async def asyncpg_profile_storage(pg_server):
    storage = AsyncPgSemanticStorage(pg_server)
    await storage.startup()
    yield storage
    await storage.delete_all()
    await storage.cleanup()
