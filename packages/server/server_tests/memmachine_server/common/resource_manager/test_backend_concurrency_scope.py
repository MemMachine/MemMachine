from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.concurrency_scope import ConcurrencyScope
from memmachine_server.common.configuration import (
    PromptConf,
    SemanticMemoryConf,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.configuration.database_conf import (
    DatabasesConf,
    NebulaGraphConf,
    Neo4jConf,
)
from memmachine_server.common.errors import ConfigurationError
from memmachine_server.common.resource_manager.database_manager import DatabaseManager
from memmachine_server.common.resource_manager.semantic_manager import (
    SemanticResourceManager,
)
from memmachine_server.common.vector_graph_store import VectorGraphStore
from memmachine_server.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
)


def _graph_databases(backend: str) -> DatabasesConf:
    if backend == "neo4j":
        return DatabasesConf(neo4j_confs={"graph": Neo4jConf()})
    return DatabasesConf(nebula_graph_confs={"graph": NebulaGraphConf()})


@pytest.mark.parametrize("backend", ["neo4j", "nebula"])
def test_graph_backend_declares_process_scope(backend: str):
    store_class = DatabaseManager.vector_graph_store_class(
        _graph_databases(backend), "graph"
    )
    assert store_class.CONCURRENCY_SCOPE == ConcurrencyScope.PROCESS
    assert Neo4jSemanticStorage.CONCURRENCY_SCOPE == ConcurrencyScope.PROCESS


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["neo4j", "nebula"])
@pytest.mark.parametrize("scope", [ConcurrencyScope.HOST, ConcurrencyScope.CLUSTER])
@pytest.mark.parametrize("cached", [False, True])
async def test_graph_factory_rejects_before_connecting(
    backend: str, scope: ConcurrencyScope, cached: bool, monkeypatch: pytest.MonkeyPatch
):
    manager = DatabaseManager(_graph_databases(backend))
    neo4j = AsyncMock()
    nebula = AsyncMock()
    monkeypatch.setattr(manager, "async_get_neo4j_driver", neo4j)
    monkeypatch.setattr(manager, "async_get_nebula_client", nebula)
    if cached:
        manager.graph_stores["graph"] = MagicMock(spec=VectorGraphStore)
    with pytest.raises(ConfigurationError, match="VectorGraphStore 'graph'"):
        await manager.get_vector_graph_store("graph", deployment_scope=scope)
    neo4j.assert_not_awaited()
    nebula.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["neo4j", "nebula"])
async def test_graph_factory_preserves_single_process_access(
    backend: str, monkeypatch: pytest.MonkeyPatch
):
    manager = DatabaseManager(_graph_databases(backend))
    store = MagicMock(spec=VectorGraphStore)
    manager.graph_stores["graph"] = store
    connector = AsyncMock()
    method = (
        "async_get_neo4j_driver" if backend == "neo4j" else "async_get_nebula_client"
    )
    monkeypatch.setattr(manager, method, connector)
    assert await manager.get_vector_graph_store("graph") is store
    connector.assert_awaited_once_with("graph", validate=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [ConcurrencyScope.HOST, ConcurrencyScope.CLUSTER])
@pytest.mark.parametrize(
    "backend", [SemanticMemoryStorageBackend.NEO4J, SemanticMemoryStorageBackend.AUTO]
)
async def test_semantic_factory_rejects_explicit_and_fallback_neo4j(
    scope: ConcurrencyScope, backend: SemanticMemoryStorageBackend
):
    resources = MagicMock()
    resources.get_sql_engine = AsyncMock(side_effect=ValueError("not a SQL database"))
    resources.get_neo4j_driver = AsyncMock()
    manager = SemanticResourceManager(
        semantic_conf=SemanticMemoryConf(
            enabled=False, database="graph", storage_backend=backend
        ),
        prompt_conf=PromptConf(),
        resource_manager=resources,
        episode_storage=MagicMock(),
        deployment_scope=scope,
    )
    with pytest.raises(ConfigurationError, match="Neo4jSemanticStorage 'graph'"):
        await manager.get_semantic_storage()
    resources.get_neo4j_driver.assert_not_awaited()
    if backend == SemanticMemoryStorageBackend.AUTO:
        resources.get_sql_engine.assert_awaited_once_with("graph", validate=True)
    else:
        resources.get_sql_engine.assert_not_awaited()
