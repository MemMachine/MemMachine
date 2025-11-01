import pytest
import pytest_asyncio
import sqlalchemy
from testcontainers.postgres import PostgresContainer

from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)
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

    yield {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }


@pytest_asyncio.fixture
async def sqlalchemy_profile_storage(pg_server):
    sqlalchemy_engine = sqlalchemy.create_engine(
        f"postgresql://{pg_server['user']}:{pg_server['password']}@{pg_server['host']}:{pg_server['port']}/{pg_server['database']}"
    )
    storage = SqlAlchemyPgVectorSemanticStorage(sqlalchemy_engine)
    await storage.startup()
    yield storage
    await storage.delete_all()
    await storage.cleanup()
