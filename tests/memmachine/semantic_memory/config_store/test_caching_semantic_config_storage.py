import pytest

from memmachine.semantic_memory.config_store.caching_semantic_config_storage import (
    CachingSemanticConfigStorage,
)
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage


class DummySemanticConfigStorage:
    def __init__(self) -> None:
        self.get_setid_config_calls = 0
        self.get_category_calls = 0
        self.register_set_id_org_set_calls = 0
        self._configs: dict[str, SemanticConfigStorage.Config] = {}

    async def startup(self) -> None:
        return None

    async def delete_all(self) -> None:
        self._configs.clear()

    async def set_setid_config(
        self,
        *,
        set_id: str,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        self._configs[set_id] = SemanticConfigStorage.Config(
            embedder_name=embedder_name,
            llm_name=llm_name,
            disabled_categories=None,
            categories=[],
        )

    async def get_setid_config(self, *, set_id: str) -> SemanticConfigStorage.Config:
        self.get_setid_config_calls += 1
        return self._configs[set_id]

    async def get_category(self, *, category_id: str):
        self.get_category_calls += 1
        return

    async def register_set_id_org_set(self, *, set_id: str, org_set_id: str) -> None:
        self.register_set_id_org_set_calls += 1
        return


@pytest.mark.asyncio
async def test_get_setid_config_cached_and_invalidated() -> None:
    wrapped = DummySemanticConfigStorage()
    storage = CachingSemanticConfigStorage(wrapped)  # type: ignore[arg-type]

    await storage.set_setid_config(set_id="set-a", embedder_name="e1", llm_name="l1")

    config1 = await storage.get_setid_config(set_id="set-a")
    config2 = await storage.get_setid_config(set_id="set-a")

    assert config1.embedder_name == "e1"
    assert config2.embedder_name == "e1"
    assert wrapped.get_setid_config_calls == 1

    await storage.set_setid_config(set_id="set-a", embedder_name="e2", llm_name="l2")

    config3 = await storage.get_setid_config(set_id="set-a")
    assert config3.embedder_name == "e2"
    assert wrapped.get_setid_config_calls == 2


@pytest.mark.asyncio
async def test_get_category_negative_cached() -> None:
    wrapped = DummySemanticConfigStorage()
    storage = CachingSemanticConfigStorage(wrapped)  # type: ignore[arg-type]

    assert await storage.get_category(category_id="missing") is None
    assert await storage.get_category(category_id="missing") is None

    assert wrapped.get_category_calls == 1


@pytest.mark.asyncio
async def test_register_set_id_org_set_short_circuits() -> None:
    wrapped = DummySemanticConfigStorage()
    storage = CachingSemanticConfigStorage(wrapped)  # type: ignore[arg-type]

    await storage.register_set_id_org_set(set_id="set-a", org_set_id="1")
    await storage.register_set_id_org_set(set_id="set-a", org_set_id="2")

    assert wrapped.register_set_id_org_set_calls == 1
