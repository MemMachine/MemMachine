"""Real PostgreSQL, Neo4j, and Qdrant recovery in one application process."""

import asyncio
from time import monotonic
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.configuration.database_conf import Neo4jConf
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.main.memmachine import MemMachine
from memmachine_server.server.app import MemMachineAPI
from server_tests.memmachine_server.common.reranker.fake_embedder import FakeEmbedder
from server_tests.memmachine_server.conftest import is_docker_available


@pytest.fixture
def readiness_containers():
    if not is_docker_available():
        pytest.skip("Docker is required for the real paused-storage recovery test")
    name = f"memmachine-readiness-{uuid4().hex[:12]}"
    # Each test owns these exact containers and their random published ports.
    # No shared Compose project, named volume, or existing database is touched.
    postgres = PostgresContainer(
        "pgvector/pgvector:pg18",
        username="readiness",
        password=uuid4().hex,
        dbname="readiness",
    ).with_name(f"{name}-postgres")
    qdrant = QdrantContainer("qdrant/qdrant:v1.19.1").with_name(f"{name}-qdrant")
    neo4j = (
        Neo4jContainer("neo4j:5.23-community", username="neo4j", password=uuid4().hex)
        .with_name(f"{name}-neo4j")
        .with_env("NEO4J_server_memory_heap_initial__size", "128M")
        .with_env("NEO4J_server_memory_heap_max__size", "256M")
        .with_env("NEO4J_server_memory_pagecache_size", "128M")
    )
    with postgres, qdrant, neo4j:
        yield postgres, qdrant, neo4j


def readiness_config(postgres, qdrant, log_path, neo4j=None):
    conf = Configuration.model_validate(
        {
            "logging": {"path": str(log_path)},
            "session_manager": {"database": "postgres"},
            "episode_store": {"database": "postgres"},
            "semantic_memory": {"enabled": False},
            "episodic_memory": {
                "enabled": True,
                "long_term_memory_enabled": True,
                "short_term_memory_enabled": False,
                "long_term_memory": {
                    "backend": "event",
                    "embedder": "deterministic",
                    "vector_store": "qdrant",
                    "segment_store": "postgres",
                },
            },
            "resources": {
                "databases": {
                    "postgres": {
                        "provider": "postgres",
                        "config": {
                            "host": postgres.get_container_host_ip(),
                            "port": int(postgres.get_exposed_port(5432)),
                            "user": postgres.username,
                            "password": postgres.password,
                            "db_name": postgres.dbname,
                        },
                    },
                    "qdrant": {
                        "provider": "qdrant",
                        "config": {
                            "host": qdrant.get_container_host_ip(),
                            "port": int(qdrant.get_exposed_port(6333)),
                            "grpc_port": int(qdrant.get_exposed_port(6334)),
                        },
                    },
                },
                "embedders": {
                    "deterministic": {
                        "provider": "openai",
                        "config": {
                            "model": "test-only",
                            "api_key": "test-only-not-used",
                            "dimensions": 2,
                        },
                    }
                },
                "language_models": {},
                "rerankers": {"identity": {"provider": "identity", "config": {}}},
            },
        }
    )
    if neo4j is not None:
        conf.resources.databases.neo4j_confs["neo4j"] = Neo4jConf(
            host=neo4j.get_container_host_ip(),
            port=int(neo4j.get_exposed_port(7687)),
            user=neo4j.username,
            password=neo4j.password,
        )
    return conf


async def assert_readiness(client, expected):
    deadline = monotonic() + 6
    while True:
        started = monotonic()
        result = await asyncio.wait_for(client.get("/api/v2/ready"), timeout=4)
        assert monotonic() - started < 3.5
        if result.status_code == expected:
            return result
        if monotonic() >= deadline:
            pytest.fail(f"Readiness did not reach {expected}: {result.json()}")
        await asyncio.sleep(0.05)


async def add_and_search(client, project, text):
    added = await client.post(
        "/api/v2/memories",
        json={**project, "types": ["episodic"], "messages": [{"content": text}]},
    )
    assert added.status_code == 200, added.text
    found = await client.post(
        "/api/v2/memories/search",
        json={**project, "types": ["episodic"], "query": text, "top_k": 10},
    )
    assert found.status_code == 200, found.text
    episodes = found.json()["content"]["episodic_memory"]["long_term_memory"][
        "episodes"
    ]
    assert any(episode["content"] == text for episode in episodes), found.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paused_storage_recovers_without_restarting_or_replacing_clients(
    readiness_containers, monkeypatch, tmp_path
):
    postgres, qdrant, neo4j = readiness_containers
    conf = readiness_config(postgres, qdrant, tmp_path / "server.log", neo4j)
    resources = ResourceManagerImpl(conf)
    # Only the external embedding provider is substituted. All application,
    # HTTP routing, storage clients, data writes, and searches are real.
    monkeypatch.setattr(
        resources, "get_embedder", AsyncMock(return_value=FakeEmbedder())
    )
    machine = MemMachine(conf, resources)
    app = MemMachineAPI()
    app.state.mem_machine = machine
    projects = [
        ({"org_id": "readiness", "project_id": "event-project"}, {}),
        (
            {"org_id": "readiness", "project_id": "graph-project"},
            {
                "backend": "declarative",
                "vector_graph_store": "neo4j",
                "embedder": "deterministic",
                "reranker": "identity",
            },
        ),
    ]
    paused = None
    try:
        await machine.start()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            # Cold checks must not initialize the selectable vector/graph stores.
            await assert_readiness(client, 200)
            assert not resources.database_manager.qdrant_clients
            assert not resources.database_manager.neo4j_drivers
            for project, project_config in projects:
                created = await client.post(
                    "/api/v2/projects", json={**project, "config": project_config}
                )
                assert created.status_code == 201, created.text
                await add_and_search(
                    client, project, "memory before the storage outage"
                )
            databases = resources.database_manager
            serving_qdrant = databases.qdrant_clients["qdrant"]
            serving_store = databases.vector_stores["qdrant"]
            serving_neo4j = databases.neo4j_drivers["neo4j"]
            serving_sql = databases.sql_engines["postgres"]
            await assert_readiness(client, 200)
            health = (await client.get("/api/v2/health")).json()
            for component, backend in (
                ("qdrant_1", qdrant),
                ("neo4j_1", neo4j),
                ("sql_1", postgres),
            ):
                paused = backend.get_wrapped_container()
                await asyncio.to_thread(paused.pause)
                try:
                    failure = await assert_readiness(client, 503)
                    assert failure.json()["components"][component] in {
                        "timeout",
                        "unavailable",
                    }
                    live = await client.get("/api/v2/health")
                    assert live.status_code == 200
                    assert live.json() == health
                finally:
                    await asyncio.to_thread(paused.unpause)
                    paused = None
                await assert_readiness(client, 200)
                assert databases.qdrant_clients["qdrant"] is serving_qdrant
                assert databases.vector_stores["qdrant"] is serving_store
                assert databases.neo4j_drivers["neo4j"] is serving_neo4j
                assert databases.sql_engines["postgres"] is serving_sql
                for project, _ in projects:
                    await add_and_search(
                        client, project, f"memory after the {component} outage"
                    )
            # Recovery retains old project data, not just the ability to create
            # a new client, fresh project, or empty store.
            for project, _ in projects:
                found = await client.post(
                    "/api/v2/memories/search",
                    json={
                        **project,
                        "types": ["episodic"],
                        "query": "memory",
                        "top_k": 10,
                    },
                )
                assert found.status_code == 200, found.text
                episodes = found.json()["content"]["episodic_memory"][
                    "long_term_memory"
                ]["episodes"]
                assert {episode["content"] for episode in episodes} >= {
                    "memory before the storage outage",
                    "memory after the qdrant_1 outage",
                    "memory after the neo4j_1 outage",
                    "memory after the sql_1 outage",
                }
    finally:
        if paused is not None:
            await asyncio.to_thread(paused.unpause)
        await machine.stop()
