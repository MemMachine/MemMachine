import asyncio
import sqlite3
import threading
from unittest.mock import AsyncMock

import httpx
import pytest

from memmachine_server.common.configuration import Configuration, SemanticMemoryConf
from memmachine_server.common.configuration.database_conf import (
    MilvusConf,
    Neo4jConf,
    QdrantConf,
    SQLiteVecVectorStoreConf,
)
from memmachine_server.common.resource_manager import readiness as readiness_module
from memmachine_server.common.resource_manager.database_manager import DatabaseManager
from memmachine_server.common.resource_manager.readiness import (
    ComponentStatus,
    ReadinessService,
    StorageProbe,
)
from memmachine_server.common.resource_manager.storage_probes import StorageProbes
from memmachine_server.main.memmachine import MemMachine
from memmachine_server.server.app import MemMachineAPI


def local_config():
    return Configuration.model_validate(
        {
            "episodic_memory": {
                "enabled": False,
                "long_term_memory_enabled": False,
                "short_term_memory_enabled": False,
            },
            "logging": {"path": ""},
            "session_manager": {"database": "local"},
            "episode_store": {"database": "local"},
            "resources": {
                "databases": {
                    "local": {"provider": "sqlite", "config": {"path": ":memory:"}}
                },
                "embedders": {},
                "language_models": {},
                "rerankers": {},
            },
        }
    )


class StatefulQdrant:
    def __init__(self):
        self.closed = False
        self.available = True
        self.calls = 0

    async def get_collections(self):
        self.calls += 1
        if self.closed or not self.available:
            raise RuntimeError("https://user:secret@private-database:6333 failed")
        return []

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_failed_probe_preserves_serving_client_and_recovers():
    conf = local_config()
    conf.resources.databases.qdrant_confs["private-name"] = QdrantConf()
    databases = DatabaseManager(conf.resources.databases)
    client = StatefulQdrant()
    databases.qdrant_clients["private-name"] = client
    probes = StorageProbes(conf, databases)
    service = ReadinessService(
        lambda: [
            StorageProbe(
                ("qdrant", "private-name"),
                "qdrant_1",
                lambda: probes.check("qdrant", "private-name"),
            )
        ],
        cache_seconds=0,
    )
    service.started()
    try:
        assert (await service.check()).status == "ready"
        client.available = False
        result = await service.check()
        assert result.status == "not_ready"
        assert result.components == {"qdrant_1": ComponentStatus.UNAVAILABLE}
        for secret in ("secret", "private-database", "private-name", "RuntimeError"):
            assert secret not in result.model_dump_json()
        assert not client.closed
        client.available = True
        assert (await service.check()).status == "ready"
        assert await client.get_collections() == []
        assert databases.qdrant_clients["private-name"] is client
    finally:
        await service.close()
        probes.close()
    assert not client.closed


@pytest.mark.asyncio
async def test_healthy_result_expires_and_detects_outage(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(readiness_module, "monotonic", lambda: now[0])
    check = AsyncMock()
    service = ReadinessService(lambda: [StorageProbe(("sql", "db"), "sql_1", check)])
    service.started()
    try:
        assert (await service.check()).status == "ready"
        check.side_effect = ConnectionError("private endpoint")
        now[0] += 0.9
        assert (await service.check()).status == "ready"
        assert check.await_count == 1
        now[0] += 0.2
        assert (await service.check()).status == "not_ready"
        assert check.await_count == 2
        check.side_effect = None
        now[0] += 1.1
        assert (await service.check()).status == "ready"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cancelled_requests_share_a_single_refresh():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def check():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()

    service = ReadinessService(lambda: [StorageProbe(("sql", "db"), "sql_1", check)])
    service.started()
    requests = [asyncio.create_task(service.check()) for _ in range(20)]
    try:
        await asyncio.wait_for(entered.wait(), 1)
        for request in requests[:10]:
            request.cancel()
        await asyncio.gather(*requests[:10], return_exceptions=True)
        assert calls == 1
        release.set()
        results = await asyncio.gather(*requests[10:])
        assert all(result.status == "ready" for result in results)
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(*requests, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
async def test_blocked_milvus_worker_reservation_survives_repeated_timeouts():
    entered = threading.Event()
    release = threading.Event()

    class BlockingMilvus:
        calls = 0
        closed = False

        def list_collections(self, *, timeout, retry_on_rate_limit):
            assert timeout == 2.0
            assert retry_on_rate_limit is False
            assert not self.closed
            self.calls += 1
            entered.set()
            assert release.wait(5), "test must release its worker"
            return []

        def close(self):
            self.closed = True

    conf = local_config()
    conf.resources.databases.milvus_confs["vector"] = MilvusConf(
        uri="http://localhost:19530"
    )
    databases = DatabaseManager(conf.resources.databases)
    client = BlockingMilvus()
    databases.milvus_clients["vector"] = client
    probes = StorageProbes(conf, databases)
    service = ReadinessService(
        lambda: [
            StorageProbe(
                ("milvus", "vector"),
                "milvus_1",
                lambda: probes.check("milvus", "vector"),
            )
        ],
        check_timeout=0.01,
        response_timeout=0.1,
        cache_seconds=0,
    )
    service.started()
    try:
        for _ in range(12):
            result = await service.check()
            assert result.components == {"milvus_1": ComponentStatus.TIMEOUT}
        assert entered.is_set()
        assert client.calls == 1
        assert len(service._operations) == 1
        assert not client.closed
        release.set()
        await asyncio.wait_for(asyncio.gather(*service._operations.values()), 1)
        assert (await service.check()).status == "ready"
        assert client.calls == 2
        assert not client.closed
    finally:
        release.set()
        await service.close()
        probes.close()


@pytest.mark.asyncio
async def test_parallel_probes_obey_response_and_concurrency_limits():
    release = asyncio.Event()
    active = 0
    peak = 0
    calls = 0

    async def check():
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        try:
            await release.wait()
        finally:
            active -= 1

    probes = [StorageProbe(("sql", str(i)), f"sql_{i}", check) for i in range(8)]
    service = ReadinessService(
        lambda: probes,
        check_timeout=0.01,
        response_timeout=0.1,
        cache_seconds=0,
        max_concurrency=2,
    )
    service.started()
    try:
        for _ in range(5):
            result = await asyncio.wait_for(service.check(), 0.2)
            assert set(result.components.values()) == {ComponentStatus.TIMEOUT}
        assert calls == 2
        assert peak == 2
        assert len(service._operations) == 8
    finally:
        release.set()
        await service.close()
    assert peak == 2


def test_selection_covers_project_alternatives_but_excludes_disabled_services():
    conf = local_config()
    conf.resources.databases.qdrant_confs.update(
        {"default": QdrantConf(), "project-alternative": QdrantConf()}
    )
    conf.resources.databases.neo4j_confs["graph-alternative"] = Neo4jConf()
    probes = StorageProbes(conf, DatabaseManager(conf.resources.databases))
    try:
        assert [probe.key for probe in probes.selected()] == [("sql", "local")]
        conf.episodic_memory.enabled = True
        conf.episodic_memory.long_term_memory_enabled = True
        assert {probe.key for probe in probes.selected()} == {
            ("sql", "local"),
            ("qdrant", "default"),
            ("qdrant", "project-alternative"),
            ("neo4j", "graph-alternative"),
        }
        conf.episodic_memory.enabled = False
        conf.semantic_memory = SemanticMemoryConf(
            enabled=True,
            storage_backend="vector_store",
            feature_store="local",
            vector_collection="project-alternative",
            config_database="local",
            embedding_model="embedder",
            llm_model="model",
        )
        assert {probe.key for probe in probes.selected()} == {
            ("sql", "local"),
            ("qdrant", "project-alternative"),
        }
    finally:
        probes.close()


@pytest.mark.asyncio
async def test_readonly_cold_local_probe_creates_no_database(tmp_path):
    conf = local_config()
    path = tmp_path / "not-initialized.sqlite"
    conf.resources.databases.sqlite_vec_vector_store_confs["vector"] = (
        SQLiteVecVectorStoreConf(path=str(path))
    )
    databases = DatabaseManager(conf.resources.databases)
    probes = StorageProbes(conf, databases)
    try:
        await probes.check("sqlite_vec", "vector")
        assert not path.exists()
        assert not databases.vector_store_sql_engines
        assert not databases.vector_stores
        conf.resources.databases.sqlite_vec_vector_store_confs["vector"].path = str(
            tmp_path / "missing-directory" / "db.sqlite"
        )
        with pytest.raises(OSError, match="directory is unavailable"):
            await probes.check("sqlite_vec", "vector")
    finally:
        probes.close()


@pytest.mark.asyncio
async def test_cold_existing_sqlite_probe_does_not_modify_the_database(tmp_path):
    conf = local_config()
    path = tmp_path / "existing.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE existing (value TEXT)")
        connection.execute("INSERT INTO existing VALUES ('preserved')")
    before = path.read_bytes()
    conf.resources.databases.relational_db_confs["local"].path = str(path)
    databases = DatabaseManager(conf.resources.databases)
    probes = StorageProbes(conf, databases)
    try:
        await probes.check("sql", "local")
        assert path.read_bytes() == before
        assert not databases.sql_engines
    finally:
        probes.close()


@pytest.mark.asyncio
async def test_shutdown_reports_stopping_to_inflight_and_late_start_requests():
    entered = asyncio.Event()
    released = asyncio.Event()

    async def check():
        entered.set()
        await released.wait()

    service = ReadinessService(
        lambda: [StorageProbe(("sql", "db"), "sql_1", check)],
        response_timeout=0.02,
    )
    service.started()
    request = asyncio.create_task(service.check())
    await entered.wait()
    try:
        await service.close()
        result = await request
        assert result.components == {"lifecycle": ComponentStatus.STOPPING}
        service.started()
        assert (await service.check()).components == {
            "lifecycle": ComponentStatus.STOPPING
        }
    finally:
        released.set()
        await asyncio.gather(*service._operations.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_missing_or_unsupported_dependency_cannot_report_ready():
    conf = local_config()
    conf.episode_store.database = "unconfigured-private-name"
    probes = StorageProbes(conf, DatabaseManager(conf.resources.databases))
    service = ReadinessService(probes.selected, cache_seconds=0)
    service.started()
    try:
        result = await service.check()
        assert result.status == "not_ready"
        assert result.components == {"configuration": ComponentStatus.UNAVAILABLE}
        assert "unconfigured-private-name" not in result.model_dump_json()
        conf.episode_store.database = "local"
        conf.episodic_memory.enabled = True
        conf.episodic_memory.long_term_memory_enabled = True
        conf.resources.databases.milvus_confs["cold-lite"] = MilvusConf(
            uri="uninitialized.db"
        )
        result = await service.check()
        assert result.status == "not_ready"
        assert result.components["milvus_1"] == ComponentStatus.UNSUPPORTED
    finally:
        await service.close()
        probes.close()


@pytest.mark.asyncio
async def test_http_lifecycle_and_local_only_readiness(tmp_path):
    app = MemMachineAPI()
    conf = local_config()
    conf.logging.path = str(tmp_path / "server.log")
    machine = MemMachine(conf)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v2/ready")
        assert response.status_code == 503
        assert response.json()["components"] == {"lifecycle": "starting"}
        app.state.mem_machine = machine
        assert (await client.get("/api/v2/ready")).status_code == 503
        health_before = await client.get("/api/v2/health")
        assert health_before.status_code == 200
        await machine.start()
        try:
            response = await client.get("/api/v2/ready")
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.json()["components"] == {"sql_1": "ready"}
            machine.begin_shutdown()
            response = await client.get("/api/v2/ready")
            assert response.status_code == 503
            assert response.json()["components"] == {"lifecycle": "stopping"}
            assert (await client.get("/api/v2/health")).json() == health_before.json()
        finally:
            await machine.stop()
        assert (await client.get("/api/v2/ready")).status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_initial_start", [False, True])
async def test_stop_before_start_and_restart_create_a_fresh_readiness_service(
    fail_initial_start, monkeypatch, tmp_path
):
    conf = local_config()
    conf.logging.path = str(tmp_path / "server.log")
    machine = MemMachine(conf)
    if fail_initial_start:
        initialize = machine._queue_pending_session_deletions
        monkeypatch.setattr(
            machine,
            "_queue_pending_session_deletions",
            AsyncMock(side_effect=RuntimeError("controlled early startup failure")),
        )
        with pytest.raises(RuntimeError, match="controlled early startup failure"):
            await machine.start()
        monkeypatch.setattr(machine, "_queue_pending_session_deletions", initialize)
    await machine.stop()
    assert (await machine.readiness()).components == {
        "lifecycle": ComponentStatus.STOPPING
    }
    for _ in range(2):
        try:
            await machine.start()
            assert (await machine.readiness()).status == "ready"
        finally:
            await machine.stop()
        assert (await machine.readiness()).status == "not_ready"


@pytest.mark.asyncio
async def test_startup_waits_for_semantic_service_and_failure_is_not_ready(
    monkeypatch, tmp_path
):
    conf = local_config()
    conf.logging.path = str(tmp_path / "server.log")
    conf.semantic_memory = SemanticMemoryConf(
        enabled=True,
        database="local",
        config_database="local",
        embedding_model="embedder",
        llm_model="model",
    )
    machine = MemMachine(conf)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_start():
        entered.set()
        await release.wait()
        raise RuntimeError("controlled startup failure")

    semantic = AsyncMock()
    semantic.start.side_effect = fail_start
    monkeypatch.setattr(
        machine.resource_manager,
        "get_semantic_service",
        AsyncMock(return_value=semantic),
    )
    startup = asyncio.create_task(machine.start())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert (await machine.readiness()).components == {
            "lifecycle": ComponentStatus.STARTING
        }
        release.set()
        with pytest.raises(RuntimeError, match="controlled startup failure"):
            await startup
        assert (await machine.readiness()).status == "not_ready"
    finally:
        release.set()
        await machine.stop()
