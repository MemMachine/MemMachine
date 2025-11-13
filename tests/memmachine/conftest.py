# conftest.py
import os
from unittest.mock import create_autospec

import pytest
import pytest_asyncio
from sqlalchemy import StaticPool, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer

from memmachine.common.configuration.model_conf import AwsBedrockModelConf, OpenAIModelConf
from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.language_model import LanguageModel
from memmachine.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockLanguageModel,
)
from memmachine.common.language_model.openai_compatible_language_model import (
    OpenAICompatibleLanguageModel,
)
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel
from memmachine.history_store.history_sqlalchemy_store import (
    BaseHistoryStore,
    SqlAlchemyHistoryStore,
)
from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)
from tests.memmachine.common.reranker.fake_embedder import FakeEmbedder
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    InMemorySemanticStorage,
)


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="need --integration option to run")

    if not config.getoption("--integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture
def mock_llm_model():
    return create_autospec(LanguageModel, instance=True)


@pytest.fixture
def mock_llm_embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def openai_integration_config():
    open_api_key = os.environ.get("OPENAI_API_KEY")
    if not open_api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    return {
        "api_key": open_api_key,
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    }


@pytest.fixture(scope="session")
def openai_client(openai_integration_config):
    import openai

    return openai.AsyncOpenAI(api_key=openai_integration_config["api_key"])


@pytest.fixture(scope="session")
def openai_embedder(openai_client, openai_integration_config):
    return OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model=openai_integration_config["embedding_model"],
            dimensions=1536,
        )
    )


@pytest.fixture(scope="session")
def openai_llm_model(openai_integration_config):
    return OpenAILanguageModel(
        OpenAIModelConf(
            api_key=openai_integration_config["api_key"],
            model=openai_integration_config["llm_model"],
        ),
    )


@pytest.fixture(scope="session")
def openai_compatible_llm_config():
    ollama_host = os.environ.get("OLLAMA_HOST")
    if not ollama_host:
        pytest.skip("OLLAMA_HOST environment variable not set")

    return {
        "api_url": ollama_host,
        "api_key": "-",
        "model": "qwen3:8b",
    }


@pytest.fixture(scope="session")
def openai_compatible_llm_model(openai_compatible_llm_config):
    return OpenAICompatibleLanguageModel(
        {
            "base_url": openai_compatible_llm_config["api_url"],
            "model": openai_compatible_llm_config["model"],
            "api_key": openai_compatible_llm_config["api_key"],
        }
    )


@pytest.fixture(scope="session")
def bedrock_integration_config():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key_id or not aws_secret_access_key:
        pytest.skip("AWS credentials not set")

    return {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "model": "openai.gpt-oss-20b-1:0",
    }


@pytest.fixture(scope="session")
def bedrock_llm_model(bedrock_integration_config):
    return AmazonBedrockLanguageModel(
        AwsBedrockModelConf(
            aws_access_key_id=bedrock_integration_config["aws_access_key_id"],
            aws_secret_access_key=bedrock_integration_config["aws_secret_access_key"],
            model_id=bedrock_integration_config["model"],
        )
    )


@pytest.fixture(
    params=[
        pytest.param("bedrock", marks=pytest.mark.integration),
        pytest.param("openai", marks=pytest.mark.integration),
        pytest.param("openai_compatible", marks=pytest.mark.integration),
    ]
)
def real_llm_model(request):
    match request.param:
        case "bedrock":
            return request.getfixturevalue("bedrock_llm_model")
        case "openai":
            return request.getfixturevalue("openai_llm_model")
        case "openai_compatible":
            return request.getfixturevalue("openai_compatible_llm_model")
        case _:
            raise ValueError(f"Unknown LLM model type: {request.param}")


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


@pytest.fixture
def sqlalchemy_pg_engine(pg_server):
    return create_async_engine(
        URL.create(
            "postgresql+asyncpg",
            username=pg_server["user"],
            password=pg_server["password"],
            host=pg_server["host"],
            port=pg_server["port"],
            database=pg_server["database"],
        )
    )


@pytest_asyncio.fixture
async def pgvector_semantic_storage(sqlalchemy_pg_engine):
    storage = SqlAlchemyPgVectorSemanticStorage(sqlalchemy_pg_engine)
    await storage.startup()
    yield storage
    await storage.delete_all()
    await storage.cleanup()


@pytest_asyncio.fixture
async def in_memory_semantic_storage():
    store = InMemorySemanticStorage()
    await store.startup()
    yield store
    await store.cleanup()


@pytest.fixture(
    params=[
        pytest.param("pgvector_semantic_storage", marks=pytest.mark.integration),
        "in_memory_semantic_storage",
    ]
)
def semantic_storage(request):
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def sqlite_history_storage():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(BaseHistoryStore.metadata.create_all)

    storage = SqlAlchemyHistoryStore(engine)
    try:
        yield storage
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def postgres_history_storage(sqlalchemy_pg_engine):
    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseHistoryStore.metadata.create_all)
        await conn.execute(text("TRUNCATE TABLE history RESTART IDENTITY CASCADE"))

    storage = SqlAlchemyHistoryStore(sqlalchemy_pg_engine)
    try:
        yield storage
    finally:
        async with sqlalchemy_pg_engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE history RESTART IDENTITY CASCADE"))


@pytest.fixture(
    params=[
        "sqlite_history_storage",
        pytest.param("postgres_history_storage", marks=pytest.mark.integration),
    ]
)
def history_storage(request):
    return request.getfixturevalue(request.param)
