import asyncio
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from memmachine import MemMachine, setup_nltk
from memmachine.common.configuration import (
    Configuration,
    EmbeddersConf,
    EpisodeStoreConf,
    LanguageModelsConf,
    LogConf,
    PromptConf,
    ResourcesConf,
    SemanticMemoryConf,
    SessionManagerConf,
)
from memmachine.common.configuration.database_conf import DatabasesConf
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.configuration.reranker_conf import RerankersConf
from memmachine.semantic_memory.semantic_model import SetIdT
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager


@pytest.fixture(scope="session")
def openai_embedder_config(
    openai_integration_config,
) -> tuple[str, EmbeddersConf]:
    embedder_id = "openai_embedder"
    return (
        embedder_id,
        EmbeddersConf.parse(
            {
                "embedders": {
                    embedder_id: {
                        "provider": "openai",
                        "config": {
                            "model": openai_integration_config["embedding_model"],
                            "api_key": openai_integration_config["api_key"],
                            "dimensions": 1536,
                        },
                    }
                }
            }
        ),
    )


@pytest.fixture(scope="session", params=["openai_embedder_config"])
def embedder_config(request) -> tuple[str, EmbeddersConf]:
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def openai_language_model_config(
    openai_integration_config,
) -> tuple[str, LanguageModelsConf]:
    language_model_id = "openai_responses_model"
    return (
        language_model_id,
        LanguageModelsConf.parse(
            {
                "language_models": {
                    language_model_id: {
                        "provider": "openai-responses",
                        "config": {
                            "model": openai_integration_config["llm_model"],
                            "api_key": openai_integration_config["api_key"],
                        },
                    }
                }
            }
        ),
    )


@pytest.fixture(scope="session")
def openai_chat_completions_language_model_config(
    openai_chat_completions_llm_config,
) -> tuple[str, LanguageModelsConf]:
    language_model_id = "openai_chat_completions_model"
    return (
        language_model_id,
        LanguageModelsConf.parse(
            {
                "language_models": {
                    language_model_id: {
                        "provider": "openai-chat-completions",
                        "config": {
                            "model": openai_chat_completions_llm_config["model"],
                            "api_key": openai_chat_completions_llm_config["api_key"],
                            "base_url": openai_chat_completions_llm_config["api_url"],
                        },
                    }
                }
            }
        ),
    )


@pytest.fixture(scope="session")
def bedrock_language_model_config(
    bedrock_integration_config,
) -> tuple[str, LanguageModelsConf]:
    language_model_id = "bedrock_model"
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return (
        language_model_id,
        LanguageModelsConf.parse(
            {
                "language_models": {
                    language_model_id: {
                        "provider": "amazon-bedrock",
                        "config": {
                            "model_id": bedrock_integration_config["model"],
                            "aws_access_key_id": bedrock_integration_config[
                                "aws_access_key_id"
                            ],
                            "aws_secret_access_key": bedrock_integration_config[
                                "aws_secret_access_key"
                            ],
                            "region": region or "us-east-1",
                        },
                    }
                }
            }
        ),
    )


@pytest.fixture(
    scope="session", params=["openai", "openai_chat_completions", "bedrock"]
)
def language_model_config(
    request,
) -> tuple[str, LanguageModelsConf]:
    fixture_by_provider = {
        "openai": "openai_language_model_config",
        "openai_chat_completions": "openai_chat_completions_language_model_config",
        "bedrock": "bedrock_language_model_config",
    }
    return request.getfixturevalue(fixture_by_provider[request.param])


@pytest.fixture(scope="session")
def identity_reranker_config() -> tuple[str, RerankersConf]:
    reranker_id = "identity_reranker"
    return (
        reranker_id,
        RerankersConf.parse(
            {"rerankers": {reranker_id: {"provider": "identity", "config": {}}}}
        ),
    )


@pytest.fixture(scope="session", params=["identity_reranker_config"])
def reranker_config(request) -> tuple[str, RerankersConf]:
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def databases_config(
    pg_server, neo4j_container
) -> tuple[dict[str, str], DatabasesConf]:
    postgres_id = "postgres_memmachine"
    neo4j_id = "neo4j_memmachine"

    neo4j_url = urlparse(neo4j_container["uri"])

    databases = DatabasesConf.parse(
        {
            "databases": {
                postgres_id: {
                    "provider": "postgres",
                    "config": {
                        "host": pg_server["host"],
                        "port": pg_server["port"],
                        "user": pg_server["user"],
                        "password": pg_server["password"],
                        "db_name": pg_server["database"],
                    },
                },
                neo4j_id: {
                    "provider": "neo4j",
                    "config": {
                        "host": neo4j_url.hostname or "localhost",
                        "port": neo4j_url.port or 7687,
                        "user": neo4j_container["username"],
                        "password": neo4j_container["password"],
                    },
                },
            }
        }
    )
    return {"postgres": postgres_id, "neo4j": neo4j_id}, databases


@pytest.fixture(scope="session")
def resources_config(
    embedder_config: tuple[str, EmbeddersConf],
    language_model_config: tuple[str, LanguageModelsConf],
    reranker_config: tuple[str, RerankersConf],
    databases_config: tuple[dict[str, str], DatabasesConf],
) -> ResourcesConf:
    _, embedders = embedder_config
    _, language_models = language_model_config
    _, rerankers = reranker_config
    _, databases = databases_config

    return ResourcesConf(
        embedders=embedders,
        language_models=language_models,
        rerankers=rerankers,
        databases=databases,
    )


@pytest.fixture(scope="session")
def memmachine_config(
    embedder_config: tuple[str, EmbeddersConf],
    language_model_config: tuple[str, LanguageModelsConf],
    reranker_config: tuple[str, RerankersConf],
    databases_config: tuple[dict[str, str], DatabasesConf],
    resources_config: ResourcesConf,
) -> Configuration:
    embedder_id, _ = embedder_config
    language_model_id, _ = language_model_config
    reranker_id, _ = reranker_config
    database_ids, _ = databases_config

    postgres_db = database_ids["postgres"]
    neo4j_db = database_ids["neo4j"]

    return Configuration(
        episodic_memory=EpisodicMemoryConfPartial(
            long_term_memory=LongTermMemoryConfPartial(
                vector_graph_store=neo4j_db,
                embedder=embedder_id,
                reranker=reranker_id,
            ),
            short_term_memory=ShortTermMemoryConfPartial(
                llm_model=language_model_id,
            ),
        ),
        semantic_memory=SemanticMemoryConf(
            database=postgres_db,
            llm_model=language_model_id,
            embedding_model=embedder_id,
        ),
        logging=LogConf(),
        prompt=PromptConf(),
        session_manager=SessionManagerConf(database=postgres_db),
        resources=resources_config,
        episode_store=EpisodeStoreConf(database=postgres_db),
    )


@pytest.fixture
def memmachine_top(memmachine_config: Configuration):
    return MemMachine(memmachine_config)


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    setup_nltk()


@pytest_asyncio.fixture
async def memmachine(memmachine_top: MemMachine):
    await memmachine_top.start()
    yield memmachine_top
    await memmachine_top.stop()


@pytest_asyncio.fixture
async def session_data(memmachine: MemMachine):
    @dataclass
    class _SessionData:
        user_profile_id: SetIdT | None
        session_id: SetIdT | None
        role_profile_id: SetIdT | None
        session_key: str | None

    s_data = _SessionData(
        user_profile_id="test_user",
        session_id="test_session",
        session_key="test_session",
        role_profile_id=None,
    )
    semantic_session: SemanticSessionManager = (
        await memmachine._resources.get_semantic_session_manager()
    )

    await asyncio.gather(
        semantic_session.delete_feature_set(session_data=s_data),
        semantic_session.delete_messages(session_data=s_data),
    )

    await memmachine.create_session(s_data.session_key)

    yield s_data

    await asyncio.gather(
        memmachine.delete_session(session_data=s_data),
        semantic_session.delete_feature_set(session_data=s_data),
        semantic_session.delete_messages(session_data=s_data),
    )
