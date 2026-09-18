"""Fixtures for the v1 API tests: a resolver over the event memory fakes."""

import pytest
from fastapi.testclient import TestClient

from memmachine_server.common.reranker import Reranker
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from memmachine_server.server.api_v1.router import get_tenant_event_memories
from memmachine_server.server.api_v1.tenant_event_memories import (
    ComponentNotEnabledError,
    EpisodicMemoryDefaults,
    TenantEventMemories,
    TenantEventMemory,
    TenantNotFoundError,
)
from memmachine_server.server.app import MemMachineAPI
from server_tests.memmachine_server.common.reranker.fake_embedder import FakeEmbedder
from server_tests.memmachine_server.episodic_memory.event_memory.conftest import (
    AngleEmbedder,
    InMemoryEventMemoryStorePartition,
    make_collection,
)


def build_event_memory(embedder: FakeEmbedder) -> EventMemory:
    """An EventMemory over the in-memory event memory store and vector collection."""
    return EventMemory(
        EventMemoryParams(
            event_memory_store_partition=InMemoryEventMemoryStorePartition(),
            vector_store_collection=make_collection(embedder),
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([WholeTextDeriver()]),
            embedder=embedder,
        )
    )


class FakeTenantEventMemories(TenantEventMemories):
    """Tenants in a dict, each with an event memory over the fakes."""

    def __init__(
        self,
        *,
        embedder: FakeEmbedder | None = None,
        reranker: Reranker | None = None,
        defaults: EpisodicMemoryDefaults | None = None,
    ) -> None:
        self.embedder = embedder if embedder is not None else FakeEmbedder()
        self.reranker = reranker
        self.defaults = defaults if defaults is not None else EpisodicMemoryDefaults()
        self.memories: dict[str, EventMemory] = {}
        self.enabled = True
        self.failure: Exception | None = None

    async def create(self, tenant):
        self._check()
        if tenant in self.memories:
            return False
        self.memories[tenant] = build_event_memory(self.embedder)
        return True

    async def exists(self, tenant):
        self._check()
        return tenant in self.memories

    async def delete(self, tenant):
        self._check()
        self.memories.pop(tenant, None)

    async def resolve(self, tenant):
        self._check()
        memory = self.memories.get(tenant)
        if memory is None:
            raise TenantNotFoundError(tenant)
        return TenantEventMemory(
            event_memory=memory,
            reranker=self.reranker,
            defaults=self.defaults,
        )

    def _check(self):
        if self.failure is not None:
            raise self.failure
        if not self.enabled:
            raise ComponentNotEnabledError("Episodic memory is not enabled")


@pytest.fixture
def angle_embedder():
    return AngleEmbedder({"near": 0.0, "far": 1.2, "query": 0.0})


@pytest.fixture
def memories():
    return FakeTenantEventMemories()


@pytest.fixture
def client(memories):
    app = MemMachineAPI()
    app.dependency_overrides[get_tenant_event_memories] = lambda: memories

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides = {}
