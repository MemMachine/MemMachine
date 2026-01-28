"""tests for resource_manager.py"""

import pytest
from pydantic import SecretStr

from memmachine.common.configuration import (
    Configuration,
    DatabasesConf,
    EmbeddersConf,
    EpisodeStoreConf,
    EpisodicMemoryConfPartial,
    LanguageModelsConf,
    LogConf,
    RerankersConf,
    ResourcesConf,
    SemanticMemoryConf,
    SessionManagerConf,
)
from memmachine.common.configuration.database_conf import Neo4jConf, SqlAlchemyConf
from memmachine.common.configuration.embedder_conf import OpenAIEmbedderConf
from memmachine.common.configuration.language_model_conf import (
    OpenAIResponsesLanguageModelConf,
)
from memmachine.common.configuration.reranker_conf import EmbedderRerankerConf
from memmachine.common.errors import (
    InvalidEmbedderError,
    InvalidLanguageModelError,
    InvalidRerankerError,
)
from memmachine.common.resource_manager import CommonResourceManager
from memmachine.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)

RERANKER_ID = "my_reranker"
EMBEDDER_ID = "my_embedder"
MODEL_ID = "my_model"
NEO4J_ID = "my_neo4j"
SQLDB_ID = "my_sqldb"


@pytest.fixture
def invalid_configure() -> Configuration:
    return Configuration(
        resources=ResourcesConf(
            rerankers=RerankersConf(
                embedder={
                    RERANKER_ID: EmbedderRerankerConf(
                        embedder_id=EMBEDDER_ID,
                    )
                },
            ),
            embedders=EmbeddersConf(
                openai={
                    EMBEDDER_ID: OpenAIEmbedderConf(
                        model="text-embedding-3-small",
                        api_key=SecretStr("invalid-api-key"),
                    )
                }
            ),
            language_models=LanguageModelsConf(
                openai_responses_language_model_confs={
                    MODEL_ID: OpenAIResponsesLanguageModelConf(
                        model="gpt-5-nano", api_key=SecretStr("invalid-api-key")
                    )
                }
            ),
            databases=DatabasesConf(
                neo4j_confs={
                    NEO4J_ID: Neo4jConf(
                        host="a.b.c.d",
                        port=9876,
                        user="neo4j",
                        password=SecretStr("invalid-password"),
                    )
                },
                relational_db_confs={
                    SQLDB_ID: SqlAlchemyConf(
                        host="e.f.g.h",
                        port=8765,
                        path="invalid_db_path",
                        dialect="postgresql",
                        driver="asyncpg",
                        user="db_user",
                        password=SecretStr("invalid-password"),
                        db_name="invalid_db_name",
                    )
                },
            ),
        ),
        episodic_memory=EpisodicMemoryConfPartial(),
        semantic_memory=SemanticMemoryConf(
            database="my_database",
            llm_model=MODEL_ID,
            embedding_model=EMBEDDER_ID,
        ),
        logging=LogConf(),
        session_manager=SessionManagerConf(),
        episode_store=EpisodeStoreConf(),
    )


@pytest.fixture
def invalid_resource_manager(invalid_configure) -> CommonResourceManager:
    return ResourceManagerImpl(invalid_configure)


@pytest.mark.asyncio
async def test_invalid_reranker(invalid_resource_manager):
    with pytest.raises(InvalidRerankerError, match="AuthenticationError"):
        _ = await invalid_resource_manager.get_reranker(RERANKER_ID, validate=True)


@pytest.mark.asyncio
async def test_invalid_embedder(invalid_resource_manager):
    with pytest.raises(InvalidEmbedderError, match="AuthenticationError"):
        _ = await invalid_resource_manager.get_embedder(EMBEDDER_ID, validate=True)


@pytest.mark.asyncio
async def test_invalid_language_model(invalid_resource_manager):
    with pytest.raises(InvalidLanguageModelError, match="AuthenticationError"):
        _ = await invalid_resource_manager.get_language_model(MODEL_ID, validate=True)


@pytest.mark.asyncio
async def test_invalid_neo4j_driver(invalid_resource_manager):
    with pytest.raises(
        Exception, match=f"Neo4j config '{NEO4J_ID}' failed verification"
    ):
        _ = await invalid_resource_manager.get_neo4j_driver(NEO4J_ID, validate=True)


@pytest.mark.asyncio
async def test_invalid_relational_db_engine(invalid_resource_manager):
    with pytest.raises(Exception, match=f"SQL config '{SQLDB_ID}' failed verification"):
        _ = await invalid_resource_manager.get_sql_engine(SQLDB_ID, validate=True)


def test_save_config_calls_configuration_save(invalid_configure, tmp_path):
    """Test that save_config() delegates to Configuration.save()."""
    # Set up the config file path
    config_file = tmp_path / "test_config.yml"
    config_file.write_text(invalid_configure.to_yaml(), encoding="utf-8")

    # Reload configuration from file (so it has config_file_path set)
    conf = Configuration.load_yml_file(str(config_file))
    resource_manager = ResourceManagerImpl(conf)

    # Call save_config
    resource_manager.save_config()

    # Verify the file was updated (it should exist and be readable)
    assert config_file.exists()
    reloaded = Configuration.load_yml_file(str(config_file))
    assert reloaded == conf


def test_save_config_with_explicit_path(invalid_configure, tmp_path):
    """Test that save_config() can save to an explicit path."""
    resource_manager = ResourceManagerImpl(invalid_configure)

    # Save to explicit path
    save_path = tmp_path / "explicit_config.yml"
    resource_manager.save_config(str(save_path))

    # Verify file was created
    assert save_path.exists()
    reloaded = Configuration.load_yml_file(str(save_path))
    # Compare YAML output since _config_file_path differs between instances
    assert reloaded.to_yaml() == invalid_configure.to_yaml()


def test_resource_manager_config_property(invalid_configure):
    """Test that config property returns the configuration."""
    resource_manager = ResourceManagerImpl(invalid_configure)
    assert resource_manager.config == invalid_configure
