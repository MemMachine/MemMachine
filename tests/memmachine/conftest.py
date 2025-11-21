# conftest.py
import os
from unittest.mock import create_autospec

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from sqlalchemy import StaticPool
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.episode_store import CountCachingEpisodeStorage, EpisodeStorage
from memmachine.common.episode_store.episode_sqlalchemy_store import (
    BaseEpisodeStore,
    SqlAlchemyEpisodeStore,
)
from memmachine.common.language_model import LanguageModel
from memmachine.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockLanguageModel,
    AmazonBedrockLanguageModelParams,
)
from memmachine.common.language_model.openai_chat_completions_language_model import (
    OpenAIChatCompletionsLanguageModel,
    OpenAIChatCompletionsLanguageModelParams,
)
from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
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
        ),
    )


@pytest.fixture(scope="session")
def openai_llm_model(openai_client, openai_integration_config):
    return OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=openai_client,
            model=openai_integration_config["llm_model"],
        ),
    )


@pytest.fixture(scope="session")
def openai_chat_completions_llm_config():
    ollama_host = os.environ.get("OLLAMA_HOST")
    if not ollama_host:
        pytest.skip("OLLAMA_HOST environment variable not set")

    return {
        "api_url": ollama_host,
        "api_key": "-",
        "model": "qwen3:8b",
    }


@pytest.fixture(scope="session")
def openai_compat_client(openai_chat_completions_llm_config):
    import openai

    openai_compat_client = openai.AsyncOpenAI(
        api_key=openai_chat_completions_llm_config["api_key"],
        base_url=openai_chat_completions_llm_config["api_url"],
    )
    return openai_compat_client


@pytest.fixture(scope="session")
def openai_chat_completions_llm_model(
    openai_compat_client, openai_chat_completions_llm_config
):
    return OpenAIChatCompletionsLanguageModel(
        OpenAIChatCompletionsLanguageModelParams(
            client=openai_compat_client,
            model=openai_chat_completions_llm_config["model"],
        ),
    )


@pytest.fixture(scope="session")
def bedrock_integration_config():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    aws_region = os.environ.get("AWS_REGION")
    if not aws_access_key_id or not aws_secret_access_key or not aws_region:
        pytest.skip("AWS credentials not set")

    return {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
        "aws_region": aws_region,
    }


@pytest.fixture(scope="session")
def boto3_bedrock_runtime_client(bedrock_integration_config):
    import boto3

    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=bedrock_integration_config["aws_access_key_id"],
        aws_secret_access_key=bedrock_integration_config["aws_secret_access_key"],
        aws_session_token=bedrock_integration_config["aws_session_token"],
        region_name=bedrock_integration_config["aws_region"],
    )


@pytest.fixture(scope="session")
def boto3_bedrock_agent_runtime_client(bedrock_integration_config):
    import boto3

    return boto3.client(
        "bedrock-agent-runtime",
        aws_access_key_id=bedrock_integration_config["aws_access_key_id"],
        aws_secret_access_key=bedrock_integration_config["aws_secret_access_key"],
        aws_session_token=bedrock_integration_config["aws_session_token"],
        region_name=bedrock_integration_config["aws_region"],
    )


@pytest.fixture(scope="session")
def bedrock_llm_model(boto3_bedrock_runtime_client):
    return AmazonBedrockLanguageModel(
        AmazonBedrockLanguageModelParams(
            client=boto3_bedrock_runtime_client,
            model_id="openai.gpt-oss-20b-1:0",
        )
    )


@pytest.fixture(
    params=[
        pytest.param("bedrock", marks=pytest.mark.integration),
        pytest.param("openai", marks=pytest.mark.integration),
        pytest.param("openai_chat_completions", marks=pytest.mark.integration),
    ],
)
def real_llm_model(request):
    match request.param:
        case "bedrock":
            return request.getfixturevalue("bedrock_llm_model")
        case "openai":
            return request.getfixturevalue("openai_llm_model")
        case "openai_chat_completions":
            return request.getfixturevalue("openai_chat_completions_llm_model")
        case _:
            raise ValueError(f"Unknown LLM model type: {request.param}")


@pytest.fixture(scope="session")
def pg_container(pytestconfig):
    if not pytestconfig.getoption("--integration"):
        pytest.skip("need --integration option to start Postgres container")

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
async def sqlalchemy_pg_engine(pg_server):
    engine = create_async_engine(
        URL.create(
            "postgresql+asyncpg",
            username=pg_server["user"],
            password=pg_server["password"],
            host=pg_server["host"],
            port=pg_server["port"],
            database=pg_server["database"],
        ),
    )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def sqlalchemy_sqlite_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
    )

    yield engine
    await engine.dispose()


@pytest.fixture(
    params=[
        "sqlalchemy_sqlite_engine",
        pytest.param("sqlalchemy_pg_engine", marks=pytest.mark.integration),
    ],
)
def sqlalchemy_engine(request):
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def pgvector_semantic_storage(sqlalchemy_pg_engine):
    storage = SqlAlchemyPgVectorSemanticStorage(sqlalchemy_pg_engine)
    storage.backend_name = "postgres"
    await storage.startup()
    yield storage
    await storage.delete_all()
    await storage.cleanup()


@pytest_asyncio.fixture
async def in_memory_semantic_storage():
    store = InMemorySemanticStorage()
    store.backend_name = "sqlite"
    await store.startup()
    yield store
    await store.cleanup()


@pytest.fixture(scope="session")
def neo4j_container(pytestconfig):
    if not pytestconfig.getoption("--integration"):
        pytest.skip("need --integration option to start Neo4j container")

    username = "neo4j"
    password = "password"
    with Neo4jContainer(
        image="neo4j:5.23",
        username=username,
        password=password,
    ) as container:
        yield {
            "uri": container.get_connection_url(),
            "username": username,
            "password": password,
        }


@pytest_asyncio.fixture(scope="session")
async def neo4j_driver(neo4j_container):
    driver = AsyncGraphDatabase.driver(
        neo4j_container["uri"],
        auth=(neo4j_container["username"], neo4j_container["password"]),
    )
    yield driver
    await driver.close()


@pytest_asyncio.fixture
async def neo4j_semantic_storage(neo4j_driver):
    storage = Neo4jSemanticStorage(neo4j_driver)
    await storage.startup()
    await storage.delete_all()
    yield storage
    await storage.delete_all()
    await storage.cleanup()


@pytest.fixture(
    params=[
        pytest.param("pgvector_semantic_storage", marks=pytest.mark.integration),
        pytest.param("neo4j_semantic_storage", marks=pytest.mark.integration),
        "in_memory_semantic_storage",
    ],
)
def semantic_storage(request):
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def sql_db_episode_storage(sqlalchemy_engine: AsyncEngine):
    engine = sqlalchemy_engine
    async with engine.begin() as conn:
        await conn.run_sync(BaseEpisodeStore.metadata.create_all)

    storage = SqlAlchemyEpisodeStore(engine)
    try:
        await storage.delete_episode_messages()
        yield storage
    finally:
        await storage.delete_episode_messages()
        await engine.dispose()


@pytest.fixture
def count_cache_episode_storage(sql_db_episode_storage: EpisodeStorage):
    return CountCachingEpisodeStorage(sql_db_episode_storage)


@pytest.fixture(
    params=["sql_db_episode_storage", "count_cache_episode_storage"],
)
def episode_storage(
    request,
    sql_db_episode_storage: EpisodeStorage,
    count_cache_episode_storage: EpisodeStorage,
):
    match request.param:
        case "sql_db_episode_storage":
            return sql_db_episode_storage
        case "count_cache_episode_storage":
            return count_cache_episode_storage

    pytest.fail("Unknown episode storage type")
