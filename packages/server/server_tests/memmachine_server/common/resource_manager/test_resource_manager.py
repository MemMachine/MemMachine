"""tests for resource_manager.py"""

import asyncio
import gc
import weakref
from collections.abc import Coroutine
from datetime import timedelta
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    MagicMock,
    NonCallableMagicMock,
    create_autospec,
    patch,
)

import pytest
from pydantic import SecretStr
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

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
)
from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.common.partition_registry import (
    PartitionedStore,
    PartitionManager,
)
from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyPartitionRegistry,
)
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.resource_manager import (
    resource_manager as resource_manager_module,
)
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.common.session_manager.session_data_manager import (
    SessionDataManager,
)
from memmachine_server.common.sql import enable_sqlite_explicit_begin
from memmachine_server.common.sql_lease_lock import SQLLeaseLockService
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

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
                        model="gpt-6-luna", api_key=SecretStr("invalid-api-key")
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
async def test_sql_lock_service_uses_session_database_and_is_shared(
    invalid_configure, tmp_path
):
    """Concurrent callers get one ready service backed by the session database."""
    invalid_configure.session_manager.database = SQLDB_ID
    invalid_configure.resources.databases.relational_db_confs = {
        SQLDB_ID: SqlAlchemyConf(
            dialect="sqlite", driver="aiosqlite", path=str(tmp_path / "session.db")
        ),
        "other": SqlAlchemyConf(
            dialect="sqlite", driver="aiosqlite", path=str(tmp_path / "other.db")
        ),
    }
    resource_manager = ResourceManagerImpl(invalid_configure)
    try:
        first, second = await asyncio.gather(
            resource_manager.get_sql_lock_service(),
            resource_manager.get_sql_lock_service(),
        )
        assert isinstance(first, SQLLeaseLockService)
        assert first is second

        lease = await first.try_acquire(
            "resource", lease_duration=timedelta(seconds=10)
        )
        assert lease is not None
        await lease.release()

        session_engine = await resource_manager.get_sql_engine(SQLDB_ID)
        other_engine = await resource_manager.get_sql_engine("other")
        async with session_engine.connect() as conn:
            session_tables = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        async with other_engine.connect() as conn:
            other_tables = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        assert "lease_lock" in session_tables
        assert "lease_lock" not in other_tables
    finally:
        await resource_manager.close()


@pytest.mark.asyncio
async def test_session_cache_size_configures_episodic_manager(invalid_configure):
    """The configured capacity reaches the manager created by the resource manager."""
    invalid_configure.session_manager.instance_cache_size = 3
    resource_manager = ResourceManagerImpl(invalid_configure)
    session_storage = AsyncMock(spec=SessionDataManager)

    with patch.object(
        resource_manager,
        "get_session_data_manager",
        new=AsyncMock(return_value=session_storage),
    ):
        manager = await resource_manager.get_episodic_memory_manager()

    assert manager._instance_cache.capacity == 3
    await manager.close()


def _partition_manager() -> NonCallableMagicMock:
    return create_autospec(PartitionManager, instance=True)


@pytest.mark.asyncio
async def test_purge_rounds_tick_and_survive_failures(monkeypatch):
    """The background purge keeps running rounds past a failing one."""
    monkeypatch.setattr(
        resource_manager_module, "_PARTITION_REGISTRY_PURGE_PAUSE_SECONDS", 0
    )
    partition_manager = _partition_manager()
    calls = 0
    third_call = asyncio.Event()

    async def counting_round() -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("dropped mid-purge")
        if calls >= 3:
            third_call.set()
        return False

    partition_manager.run_purge_round.side_effect = counting_round

    task = asyncio.create_task(
        resource_manager_module._run_purge_rounds_forever(partition_manager, "ns")
    )
    await asyncio.wait_for(third_call.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_get_partition_registry_starts_each_namespace_once_with_its_purge_rounds(
    invalid_resource_manager, monkeypatch, tmp_path
):
    """A namespace's first get starts its registry and runs its purge rounds
    over the stores that get is given, later gets start nothing more, and
    close() cancels the purges."""
    monkeypatch.setattr(
        resource_manager_module, "_PARTITION_REGISTRY_PURGE_PAUSE_SECONDS", 3600
    )
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    enable_sqlite_explicit_begin(engine)
    monkeypatch.setattr(
        invalid_resource_manager, "get_sql_engine", AsyncMock(return_value=engine)
    )
    monkeypatch.setattr(
        invalid_resource_manager._database_manager, "close", AsyncMock()
    )
    startup = SQLAlchemyPartitionRegistry.startup
    started: list[str] = []

    async def counted_startup(registry: SQLAlchemyPartitionRegistry) -> None:
        started.append(registry._partition_namespace)
        await startup(registry)

    monkeypatch.setattr(SQLAlchemyPartitionRegistry, "startup", counted_startup)
    sweepers: list[tuple[str, tuple[PartitionedStore, ...]]] = []

    async def recorded_sweeper(
        partition_manager: PartitionManager, partition_namespace: str
    ) -> None:
        sweepers.append((partition_namespace, partition_manager._partitioned_stores))
        await asyncio.Event().wait()

    monkeypatch.setattr(
        resource_manager_module, "_run_purge_rounds_forever", recorded_sweeper
    )
    store = create_autospec(PartitionedStore, instance=True)
    later_store = create_autospec(PartitionedStore, instance=True)
    other_store = create_autospec(PartitionedStore, instance=True)

    first = await invalid_resource_manager.get_partition_registry(
        "db",
        partition_namespace="a",
        tombstone_retention_seconds=1,
        partitioned_stores=[store],
    )
    second = await invalid_resource_manager.get_partition_registry(
        "db",
        partition_namespace="a",
        tombstone_retention_seconds=1,
        partitioned_stores=[later_store],
    )
    other = await invalid_resource_manager.get_partition_registry(
        "db",
        partition_namespace="b",
        tombstone_retention_seconds=1,
        partitioned_stores=[other_store],
    )
    await asyncio.sleep(0)

    assert started == ["a", "b"]
    for registry, partition_namespace in ((first, "a"), (second, "a"), (other, "b")):
        assert isinstance(registry, SQLAlchemyPartitionRegistry)
        assert registry._partition_namespace == partition_namespace
    assert sweepers == [("a", (store,)), ("b", (other_store,))]
    tasks = list(invalid_resource_manager._partition_registry_purge_tasks)
    assert len(tasks) == 2

    # Bounded: close() waits for the purges it cancels.
    await asyncio.wait_for(invalid_resource_manager.close(), 30)

    assert all(task.cancelled() for task in tasks)
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_first_gets_of_a_namespace_start_its_registry_and_purge_rounds_once(
    invalid_resource_manager, monkeypatch, tmp_path
):
    """Gets that all find the namespace unstarted before the first startup
    ends start its registry and its purge rounds once."""
    engine = _sqlite_engine(tmp_path / "registry.db")
    gets = 3
    gets_begun = 0
    all_gets_begun = asyncio.Event()

    async def get_sql_engine(name: str, validate: bool = False) -> AsyncEngine:
        nonlocal gets_begun
        gets_begun += 1
        if gets_begun == gets:
            all_gets_begun.set()
        return engine

    monkeypatch.setattr(invalid_resource_manager, "get_sql_engine", get_sql_engine)
    monkeypatch.setattr(
        invalid_resource_manager._database_manager, "close", AsyncMock()
    )
    startup = SQLAlchemyPartitionRegistry.startup
    started: list[str] = []
    end_startup = asyncio.Event()

    async def held_startup(registry: SQLAlchemyPartitionRegistry) -> None:
        started.append(registry._partition_namespace)
        await end_startup.wait()
        await startup(registry)

    monkeypatch.setattr(SQLAlchemyPartitionRegistry, "startup", held_startup)
    purge_rounds = _record_purge_rounds(monkeypatch)
    store = create_autospec(PartitionedStore, instance=True)

    concurrent_gets = asyncio.gather(
        *(
            invalid_resource_manager.get_partition_registry(
                "db",
                partition_namespace="a",
                tombstone_retention_seconds=1,
                partitioned_stores=[store],
            )
            for _ in range(gets)
        )
    )
    # Every get has found the namespace unstarted, and one is in its startup.
    await all_gets_begun.wait()
    end_startup.set()
    await concurrent_gets

    assert started == ["a"]
    assert [
        (partition_namespace, partition_manager._partitioned_stores)
        for partition_namespace, partition_manager in purge_rounds
    ] == [("a", (store,))]
    await invalid_resource_manager.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_a_get_whose_registry_startup_fails_starts_no_purge_rounds_and_the_next_get_starts_them(
    invalid_resource_manager, monkeypatch, tmp_path
):
    engine = _sqlite_engine(tmp_path / "registry.db")
    monkeypatch.setattr(
        invalid_resource_manager, "get_sql_engine", AsyncMock(return_value=engine)
    )
    monkeypatch.setattr(
        invalid_resource_manager._database_manager, "close", AsyncMock()
    )
    startup = SQLAlchemyPartitionRegistry.startup
    startups = 0

    async def startup_failing_once(registry: SQLAlchemyPartitionRegistry) -> None:
        nonlocal startups
        startups += 1
        if startups == 1:
            raise ConnectionError("the database is down")
        await startup(registry)

    monkeypatch.setattr(SQLAlchemyPartitionRegistry, "startup", startup_failing_once)
    purge_rounds = _record_purge_rounds(monkeypatch)
    store = create_autospec(PartitionedStore, instance=True)

    with pytest.raises(ConnectionError):
        await invalid_resource_manager.get_partition_registry(
            "db",
            partition_namespace="a",
            tombstone_retention_seconds=1,
            partitioned_stores=[store],
        )

    assert purge_rounds == []

    await invalid_resource_manager.get_partition_registry(
        "db",
        partition_namespace="a",
        tombstone_retention_seconds=1,
        partitioned_stores=[store],
    )

    assert startups == 2
    assert [
        (partition_namespace, partition_manager._partitioned_stores)
        for partition_namespace, partition_manager in purge_rounds
    ] == [("a", (store,))]
    await invalid_resource_manager.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_one_namespace_in_two_databases_starts_a_registry_and_purge_rounds_in_each(
    invalid_resource_manager, monkeypatch, tmp_path
):
    engines = {
        database: _sqlite_engine(tmp_path / f"{database}.db")
        for database in ("db1", "db2")
    }

    async def get_sql_engine(name: str, validate: bool = False) -> AsyncEngine:
        return engines[name]

    monkeypatch.setattr(invalid_resource_manager, "get_sql_engine", get_sql_engine)
    monkeypatch.setattr(
        invalid_resource_manager._database_manager, "close", AsyncMock()
    )
    startup = SQLAlchemyPartitionRegistry.startup
    started: list[SQLAlchemyPartitionRegistry] = []

    async def counted_startup(registry: SQLAlchemyPartitionRegistry) -> None:
        started.append(registry)
        await startup(registry)

    monkeypatch.setattr(SQLAlchemyPartitionRegistry, "startup", counted_startup)
    purge_rounds = _record_purge_rounds(monkeypatch)
    stores = {
        database: create_autospec(PartitionedStore, instance=True)
        for database in engines
    }

    for database, store in stores.items():
        await invalid_resource_manager.get_partition_registry(
            database,
            partition_namespace="a",
            tombstone_retention_seconds=1,
            partitioned_stores=[store],
        )

    assert [
        (registry._engine, registry._partition_namespace) for registry in started
    ] == [(engines["db1"], "a"), (engines["db2"], "a")]
    assert [
        (
            partition_namespace,
            partition_manager._partition_registry,
            partition_manager._partitioned_stores,
        )
        for partition_namespace, partition_manager in purge_rounds
    ] == [
        ("a", started[0], (stores["db1"],)),
        ("a", started[1], (stores["db2"],)),
    ]
    await invalid_resource_manager.close()
    for engine in engines.values():
        await engine.dispose()


@pytest.mark.asyncio
async def test_purge_rounds_are_timed_under_their_namespace_with_the_prometheus_metrics_factory(
    invalid_resource_manager, monkeypatch, tmp_path
):
    engine = _sqlite_engine(tmp_path / "registry.db")
    monkeypatch.setattr(
        invalid_resource_manager, "get_sql_engine", AsyncMock(return_value=engine)
    )
    monkeypatch.setattr(
        invalid_resource_manager._database_manager, "close", AsyncMock()
    )
    metrics_factory = MagicMock(spec=MetricsFactory)
    get_metrics_factory = AsyncMock(return_value=metrics_factory)
    monkeypatch.setattr(ResourceManagerImpl, "get_metrics_factory", get_metrics_factory)
    purge_rounds = _record_purge_rounds(monkeypatch)

    await invalid_resource_manager.get_partition_registry(
        "db",
        partition_namespace="a",
        tombstone_retention_seconds=1,
        partitioned_stores=[create_autospec(PartitionedStore, instance=True)],
    )
    [(_, partition_manager)] = purge_rounds

    assert await partition_manager.run_purge_round() is False

    get_metrics_factory.assert_awaited_with("prometheus")
    # The namespace is the prefix of its purge rounds' metrics.
    assert metrics_factory.get_histogram.call_args.args[0] == "a_latency_seconds"
    [observation] = metrics_factory.get_histogram.return_value.observe.call_args_list
    assert observation.kwargs["labels"] == {
        "operation": "run_purge_round",
        "status": "ok",
    }
    await invalid_resource_manager.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_get_partition_registry_refuses_a_database_the_configuration_does_not_define(
    invalid_resource_manager,
):
    with pytest.raises(ValueError, match="SQL config 'undefined' not found"):
        await invalid_resource_manager.get_partition_registry(
            "undefined",
            partition_namespace="a",
            tombstone_retention_seconds=1,
            partitioned_stores=[],
        )


def _sqlite_engine(path: Path) -> AsyncEngine:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    enable_sqlite_explicit_begin(engine)
    return engine


def _record_purge_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, PartitionManager]]:
    """Replace each purge task's purge rounds with a wait that never ends, and record the task's namespace and partition manager as the task is created."""
    purge_rounds: list[tuple[str, PartitionManager]] = []

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    def recorded_purge_rounds(
        partition_manager: PartitionManager, partition_namespace: str
    ) -> Coroutine[None, None, None]:
        purge_rounds.append((partition_namespace, partition_manager))
        return wait_forever()

    monkeypatch.setattr(
        resource_manager_module, "_run_purge_rounds_forever", recorded_purge_rounds
    )
    return purge_rounds


@pytest.mark.asyncio
async def test_get_event_memory_builds_one_event_memory_and_runs_its_jobs_until_close(
    invalid_resource_manager, monkeypatch
):
    """Concurrent first accesses build the deployment's EventMemory once, and close() ends both kinds of its jobs."""
    event_memory = create_autospec(EventMemory, instance=True)
    indexed = asyncio.Event()
    purged = asyncio.Event()

    def idle_job(ran: asyncio.Event):
        async def job() -> bool:
            ran.set()
            return False

        return job

    event_memory.index_pending_segments.side_effect = idle_job(indexed)
    event_memory.purge_forgotten_segments.side_effect = idle_job(purged)
    build = AsyncMock(return_value=event_memory)
    monkeypatch.setattr(resource_manager_module, "event_memory_from_config", build)

    handed_out = await asyncio.gather(
        invalid_resource_manager.get_event_memory(),
        invalid_resource_manager.get_event_memory(),
    )
    await asyncio.wait_for(asyncio.gather(indexed.wait(), purged.wait()), 5)
    jobs = invalid_resource_manager._event_memory_jobs

    await invalid_resource_manager.close()

    assert handed_out == [event_memory, event_memory]
    build.assert_awaited_once_with(invalid_resource_manager)
    assert len(jobs) == 2
    assert all(job.cancelled() for job in jobs)


@pytest.mark.asyncio
async def test_event_memory_jobs_loop_survives_a_failing_call(monkeypatch):
    """The loop keeps running jobs past a call that raised."""
    monkeypatch.setattr(
        resource_manager_module, "_EVENT_MEMORY_JOBS_INTERVAL_SECONDS", 0
    )
    calls = 0
    third_call = asyncio.Event()

    async def counting_job() -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("dropped mid-job")
        if calls >= 3:
            third_call.set()
        return False

    task = asyncio.create_task(
        resource_manager_module._run_event_memory_jobs_forever(
            counting_job, "indexing of pending segments"
        )
    )
    await asyncio.wait_for(third_call.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_event_memory_jobs_loop_runs_the_next_job_at_once_after_one_ran(
    monkeypatch,
):
    """A call that ran a job is followed by the next without the interval."""
    monkeypatch.setattr(
        resource_manager_module, "_EVENT_MEMORY_JOBS_INTERVAL_SECONDS", 3600
    )
    calls = 0
    third_call = asyncio.Event()

    async def busy_job() -> bool:
        nonlocal calls
        calls += 1
        if calls >= 3:
            third_call.set()
            return False
        return True

    task = asyncio.create_task(
        resource_manager_module._run_event_memory_jobs_forever(
            busy_job, "indexing of pending segments"
        )
    )
    await asyncio.wait_for(third_call.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_purge_task_does_not_pin_the_manager(invalid_configure, monkeypatch):
    """A pending purge task keeps its partition manager alive, not the
    resource manager that started it: a resource manager dropped without
    close() is collectable."""
    monkeypatch.setattr(
        resource_manager_module, "_PARTITION_REGISTRY_PURGE_PAUSE_SECONDS", 3600
    )
    manager = ResourceManagerImpl(invalid_configure)
    partition_manager = _partition_manager()
    partition_manager.run_purge_round.return_value = False
    task = asyncio.create_task(
        resource_manager_module._run_purge_rounds_forever(partition_manager, "ns")
    )
    manager._partition_registry_purge_tasks.append(task)
    manager_ref = weakref.ref(manager)
    await asyncio.sleep(0)  # the loop makes its first call and parks in sleep

    del manager
    gc.collect()

    assert manager_ref() is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
