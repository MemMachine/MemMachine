import pytest
import pytest_asyncio

from memmachine_server.semantic_memory.config_store.caching_semantic_config_storage import (
    CachingSemanticConfigStorage,
)
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.config_store.config_store_sqlalchemy import (
    SemanticConfigStorageSqlAlchemy,
)


class DummySemanticConfigStorage:
    def __init__(self) -> None:
        self.get_setid_config_calls = 0
        self.get_category_calls = 0
        self.register_set_id_set_type_calls = 0
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

    async def register_set_id_set_type(self, *, set_id: str, set_type_id: str) -> None:
        self.register_set_id_set_type_calls += 1
        return


@pytest.mark.asyncio
async def test_get_setid_config_cached_and_invalidated() -> None:
    wrapped = DummySemanticConfigStorage()
    storage = CachingSemanticConfigStorage(wrapped)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

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
    storage = CachingSemanticConfigStorage(wrapped)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    assert await storage.get_category(category_id="missing") is None
    assert await storage.get_category(category_id="missing") is None

    assert wrapped.get_category_calls == 1


@pytest.mark.asyncio
async def test_register_set_id_set_type_short_circuits() -> None:
    wrapped = DummySemanticConfigStorage()
    storage = CachingSemanticConfigStorage(wrapped)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    await storage.register_set_id_set_type(set_id="set-a", set_type_id="1")
    await storage.register_set_id_set_type(set_id="set-a", set_type_id="2")

    assert wrapped.register_set_id_set_type_calls == 1


@pytest_asyncio.fixture
async def cached_sqlite_config_storage(sqlalchemy_sqlite_engine):
    storage = SemanticConfigStorageSqlAlchemy(sqlalchemy_sqlite_engine)
    await storage.startup()
    return CachingSemanticConfigStorage(storage)


@pytest.mark.asyncio
async def test_created_category_replaces_cached_absence(
    cached_sqlite_config_storage: CachingSemanticConfigStorage,
) -> None:
    # Given an id the cache has seen missing, which the store hands out again
    category_id = await cached_sqlite_config_storage.create_category(
        set_id="set-a", category_name="hobbies", prompt="Hobbies"
    )
    await cached_sqlite_config_storage.delete_category(category_id=category_id)
    assert (
        await cached_sqlite_config_storage.get_category(category_id=category_id) is None
    )

    # When a new category gets that id
    new_id = await cached_sqlite_config_storage.create_category(
        set_id="set-b", category_name="music", prompt="Music"
    )
    assert new_id == category_id

    # Then the cache returns the new category
    category = await cached_sqlite_config_storage.get_category(category_id=new_id)
    assert category is not None
    assert (category.name, category.set_id) == ("music", "set-b")


@pytest.mark.asyncio
async def test_deleted_set_type_drops_cached_categories_and_tags(
    cached_sqlite_config_storage: CachingSemanticConfigStorage,
) -> None:
    set_type_id = await cached_sqlite_config_storage.add_set_type_id(
        org_id="org", metadata_tags=["repo"]
    )
    category_id = await cached_sqlite_config_storage.create_set_type_category(
        set_type_id=set_type_id, category_name="code", prompt="Code"
    )
    tag_id = await cached_sqlite_config_storage.add_tag(
        category_id=category_id, tag_name="style", description="Style"
    )
    assert (
        await cached_sqlite_config_storage.get_category(category_id=category_id)
        is not None
    )
    assert await cached_sqlite_config_storage.get_tag(tag_id=tag_id) is not None

    await cached_sqlite_config_storage.delete_set_type_id(set_type_id=set_type_id)

    assert (
        await cached_sqlite_config_storage.get_category(category_id=category_id) is None
    )
    assert await cached_sqlite_config_storage.get_tag(tag_id=tag_id) is None
