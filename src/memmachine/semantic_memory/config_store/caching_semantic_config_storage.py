"""SemanticConfigStorage wrapper providing an in-memory cache."""

from __future__ import annotations

from memmachine.common import rw_locks
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_model import (
    CategoryIdT,
    OrgSetIdEntry,
    SemanticCategory,
    SetIdT,
    TagIdT,
)


class CachingSemanticConfigStorage(SemanticConfigStorage):
    """Add an incoherent in-memory cache to a wrapped SemanticConfigStorage.

    This cache is intended to reduce repetitive reads within a single process.
    Cache entries are invalidated on write operations.
    """

    def __init__(self, wrapped: SemanticConfigStorage) -> None:
        """Initialize the cache with a wrapped storage backend."""
        self._wrapped = wrapped

        # Per-set_id locks
        self._setid_locks = rw_locks.AsyncRWLockPool()

        # Shared lock for non-set_id caches
        self._other_lock = rw_locks.AsyncRWLock()

        self._setid_config_cache: dict[SetIdT, SemanticConfigStorage.Config] = {}
        self._registered_set_id_org_sets: set[SetIdT] = set()

        self._category_cache: dict[
            CategoryIdT, SemanticConfigStorage.Category | None
        ] = {}
        self._tag_cache: dict[TagIdT, SemanticConfigStorage.Tag | None] = {}
        self._org_set_categories_cache: dict[str, list[SemanticCategory]] = {}
        self._org_set_ids_cache: dict[str, list[OrgSetIdEntry]] = {}

    async def startup(self) -> None:
        await self._wrapped.startup()

    async def delete_all(self) -> None:
        await self._wrapped.delete_all()
        await self._clear_all_caches()

    async def set_setid_config(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        await self._wrapped.set_setid_config(
            set_id=set_id,
            embedder_name=embedder_name,
            llm_name=llm_name,
        )

        async with self._setid_locks.write_lock(set_id):
            self._setid_config_cache.pop(set_id, None)

    async def get_setid_config(self, *, set_id: SetIdT) -> SemanticConfigStorage.Config:
        async with self._setid_locks.read_lock(set_id):
            cached = self._setid_config_cache.get(set_id)
            if cached is not None:
                return cached

        async with self._setid_locks.write_lock(set_id):
            cached = self._setid_config_cache.get(set_id)
            if cached is not None:
                return cached

            config = await self._wrapped.get_setid_config(set_id=set_id)
            self._setid_config_cache[set_id] = config
            return config

    async def register_set_id_org_set(self, *, set_id: SetIdT, org_set_id: str) -> None:
        async with self._setid_locks.write_lock(set_id):
            if set_id in self._registered_set_id_org_sets:
                return

            self._registered_set_id_org_sets.add(set_id)

            try:
                await self._wrapped.register_set_id_org_set(
                    set_id=set_id,
                    org_set_id=org_set_id,
                )
            except Exception:
                self._registered_set_id_org_sets.discard(set_id)
                raise

            self._setid_config_cache.pop(set_id, None)

    async def get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> SemanticConfigStorage.Category | None:
        async with self._other_lock.read_lock():
            if category_id in self._category_cache:
                return self._category_cache[category_id]

        category = await self._wrapped.get_category(category_id=category_id)

        async with self._other_lock.write_lock():
            self._category_cache[category_id] = category

        return category

    async def get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> list[SetIdT]:
        return await self._wrapped.get_category_set_ids(category_id=category_id)

    async def create_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        category_id = await self._wrapped.create_category(
            set_id=set_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )
        await self._invalidate_set_id_config(set_id)
        return category_id

    async def clone_category(
        self,
        *,
        category_id: CategoryIdT,
        new_set_id: SetIdT,
        new_name: str,
    ) -> CategoryIdT:
        cloned_id = await self._wrapped.clone_category(
            category_id=category_id,
            new_set_id=new_set_id,
            new_name=new_name,
        )
        await self._invalidate_set_id_config(new_set_id)
        return cloned_id

    async def delete_category(self, *, category_id: CategoryIdT) -> None:
        await self._wrapped.delete_category(category_id=category_id)

        async with self._setid_locks.all_write_locks():
            self._setid_config_cache.clear()

        async with self._other_lock.write_lock():
            self._category_cache.pop(category_id, None)

            # Deleting a category also deletes its tags.
            self._tag_cache.clear()

    async def add_disabled_category_to_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        await self._wrapped.add_disabled_category_to_setid(
            set_id=set_id,
            category_name=category_name,
        )
        await self._invalidate_set_id_config(set_id)

    async def remove_disabled_category_from_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        await self._wrapped.remove_disabled_category_from_setid(
            set_id=set_id,
            category_name=category_name,
        )
        await self._invalidate_set_id_config(set_id)

    async def create_org_set_category(
        self,
        *,
        org_set_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        category_id = await self._wrapped.create_org_set_category(
            org_set_id=org_set_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

        async with self._other_lock.write_lock():
            self._org_set_categories_cache.pop(org_set_id, None)

        async with self._setid_locks.all_write_locks():
            self._setid_config_cache.clear()

        return category_id

    async def get_org_set_categories(
        self,
        *,
        org_set_id: str,
    ) -> list[SemanticCategory]:
        async with self._other_lock.read_lock():
            if org_set_id in self._org_set_categories_cache:
                return self._org_set_categories_cache[org_set_id]

        categories = await self._wrapped.get_org_set_categories(org_set_id=org_set_id)

        async with self._other_lock.write_lock():
            self._org_set_categories_cache[org_set_id] = categories

        return categories

    async def get_tag(self, *, tag_id: str) -> SemanticConfigStorage.Tag | None:
        async with self._other_lock.read_lock():
            if tag_id in self._tag_cache:
                return self._tag_cache[tag_id]

        tag = await self._wrapped.get_tag(tag_id=tag_id)

        async with self._other_lock.write_lock():
            self._tag_cache[tag_id] = tag

        return tag

    async def add_tag(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        description: str,
    ) -> TagIdT:
        tag_id = await self._wrapped.add_tag(
            category_id=category_id,
            tag_name=tag_name,
            description=description,
        )
        await self._invalidate_tags_and_setid_configs()
        return tag_id

    async def update_tag(
        self,
        *,
        tag_id: str,
        tag_name: str,
        tag_description: str,
    ) -> None:
        await self._wrapped.update_tag(
            tag_id=tag_id,
            tag_name=tag_name,
            tag_description=tag_description,
        )
        await self._invalidate_tags_and_setid_configs()

    async def delete_tag(self, *, tag_id: str) -> None:
        await self._wrapped.delete_tag(tag_id=tag_id)

        async with self._setid_locks.all_write_locks():
            self._setid_config_cache.clear()

        async with self._other_lock.write_lock():
            self._tag_cache.pop(tag_id, None)

    async def add_org_set_id(
        self,
        *,
        org_id: str,
        org_level_set: bool = False,
        metadata_tags: list[str],
    ) -> str:
        org_set_id = await self._wrapped.add_org_set_id(
            org_id=org_id,
            org_level_set=org_level_set,
            metadata_tags=metadata_tags,
        )

        async with self._other_lock.write_lock():
            self._org_set_ids_cache.pop(org_id, None)

        return org_set_id

    async def list_org_set_ids(self, *, org_id: str) -> list[OrgSetIdEntry]:
        async with self._other_lock.read_lock():
            if org_id in self._org_set_ids_cache:
                return self._org_set_ids_cache[org_id]

        entries = await self._wrapped.list_org_set_ids(org_id=org_id)

        async with self._other_lock.write_lock():
            self._org_set_ids_cache[org_id] = entries

        return entries

    async def delete_org_set_id(self, *, org_set_id: str) -> None:
        await self._wrapped.delete_org_set_id(org_set_id=org_set_id)

        async with self._other_lock.write_lock():
            self._org_set_categories_cache.pop(org_set_id, None)
            self._org_set_ids_cache.clear()

        async with self._setid_locks.all_write_locks():
            self._setid_config_cache.clear()
            self._registered_set_id_org_sets.clear()

    async def _clear_all_caches(self) -> None:
        async with self._setid_locks.all_write_locks():
            self._setid_config_cache.clear()
            self._registered_set_id_org_sets.clear()

        async with self._other_lock.write_lock():
            self._category_cache.clear()
            self._tag_cache.clear()
            self._org_set_categories_cache.clear()
            self._org_set_ids_cache.clear()

    async def _invalidate_set_id_config(self, set_id: SetIdT) -> None:
        async with self._setid_locks.write_lock(set_id):
            self._setid_config_cache.pop(set_id, None)

    async def _invalidate_tags_and_setid_configs(self) -> None:
        async with self._setid_locks.all_write_locks():
            self._setid_config_cache.clear()

        async with self._other_lock.write_lock():
            self._tag_cache.clear()
