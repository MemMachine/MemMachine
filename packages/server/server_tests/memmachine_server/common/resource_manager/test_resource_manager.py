"""tests for resource_manager.py"""

import asyncio
import gc
import weakref
from unittest.mock import create_autospec

import pytest
from pydantic import SecretStr

from memmachine_server.common.configuration import (
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
from memmachine_server.common.configuration.database_conf import (
    Neo4jConf,
    SqlAlchemyConf,
)
from memmachine_server.common.configuration.embedder_conf import OpenAIEmbedderConf
from memmachine_server.common.configuration.language_model_conf import (
    OpenAIResponsesLanguageModelConf,
)
from memmachine_server.common.configuration.reranker_conf import EmbedderRerankerConf
from memmachine_server.common.errors import (
    InvalidEmbedderError,
    InvalidLanguageModelError,
    InvalidRerankerError,
    ResourceManagerClosedError,
)
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.resource_manager import (
    resource_manager as resource_manager_module,
)
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.episodic_memory.event_memory.segment_store import SegmentStore

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
            config_database="my_database",
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


@pytest.mark.asyncio
async def test_segment_store_purge_loop_ticks_and_survives_failures(
    invalid_resource_manager, monkeypatch
):
    """The background purge keeps driving the store past a failing call."""
    monkeypatch.setattr(
        resource_manager_module, "_SEGMENT_STORE_PURGE_INTERVAL_SECONDS", 0
    )
    store = create_autospec(SegmentStore, instance=True)
    calls = 0
    third_call = asyncio.Event()

    async def counting_purge() -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("dropped mid-purge")
        if calls >= 3:
            third_call.set()
        return False

    store.purge_deleted_partitions.side_effect = counting_purge

    task = asyncio.create_task(
        resource_manager_module._purge_deleted_partitions_forever(store)
    )
    await asyncio.wait_for(third_call.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_segment_store_purge_drains_backlog_without_waiting(
    invalid_resource_manager, monkeypatch
):
    """True from a purge call means more work: the next call comes after
    the short busy pause, never the idle tick."""
    monkeypatch.setattr(
        resource_manager_module, "_SEGMENT_STORE_PURGE_INTERVAL_SECONDS", 3600
    )
    monkeypatch.setattr(
        resource_manager_module, "_SEGMENT_STORE_PURGE_BUSY_PAUSE_SECONDS", 0
    )
    store = create_autospec(SegmentStore, instance=True)
    calls = 0
    drained = asyncio.Event()

    async def backlogged_purge() -> bool:
        nonlocal calls
        calls += 1
        if calls >= 3:
            drained.set()
            return False
        return True

    store.purge_deleted_partitions.side_effect = backlogged_purge

    task = asyncio.create_task(
        resource_manager_module._purge_deleted_partitions_forever(store)
    )
    await asyncio.wait_for(drained.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_close_cancels_segment_store_purge_tasks(
    invalid_resource_manager, monkeypatch
):
    """close() ends the background purge before shutting stores down."""
    monkeypatch.setattr(
        resource_manager_module, "_SEGMENT_STORE_PURGE_INTERVAL_SECONDS", 0
    )
    store = create_autospec(SegmentStore, instance=True)
    started = asyncio.Event()

    async def signalling_purge() -> bool:
        started.set()
        return False

    store.purge_deleted_partitions.side_effect = signalling_purge

    task = asyncio.create_task(
        resource_manager_module._purge_deleted_partitions_forever(store)
    )
    invalid_resource_manager._segment_store_purge_tasks.append(task)
    await asyncio.wait_for(started.wait(), 5)

    await invalid_resource_manager.close()

    assert task.cancelled()


@pytest.mark.asyncio
async def test_purge_task_does_not_pin_the_manager(invalid_configure, monkeypatch):
    """A pending purge task keeps its store alive, not the manager that
    started it: a manager dropped without close() is collectable."""
    monkeypatch.setattr(
        resource_manager_module, "_SEGMENT_STORE_PURGE_INTERVAL_SECONDS", 3600
    )
    manager = ResourceManagerImpl(invalid_configure)
    store = create_autospec(SegmentStore, instance=True)
    store.purge_deleted_partitions.return_value = False
    task = asyncio.create_task(
        resource_manager_module._purge_deleted_partitions_forever(store)
    )
    manager._segment_store_purge_tasks.append(task)
    manager_ref = weakref.ref(manager)
    await asyncio.sleep(0)  # the loop makes its first call and parks in sleep

    del manager
    gc.collect()

    assert manager_ref() is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_get_segment_store_after_close_raises(invalid_resource_manager):
    """A get racing or following close must refuse, not rebuild on a dead engine."""
    await invalid_resource_manager.close()
    with pytest.raises(ResourceManagerClosedError):
        await invalid_resource_manager.get_segment_store(SQLDB_ID)
