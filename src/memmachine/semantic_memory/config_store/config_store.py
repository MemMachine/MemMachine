"""Abstract interface for storing semantic configuration data."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from memmachine.semantic_memory.semantic_model import (
    CategoryIdT,
    OrgSetTypeEntry,
    SemanticCategory,
    SetIdT,
    TagIdT,
)


@runtime_checkable
class SemanticConfigStorage(Protocol):
    """Contract for persisting and retrieving semantic memory configuration."""

    async def startup(self) -> None: ...

    async def delete_all(self) -> None: ...

    async def set_setid_config(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None: ...

    @dataclass(frozen=True)
    class Config:
        """Configuration values tied to a specific set identifier."""

        embedder_name: str | None
        llm_name: str | None
        disabled_categories: list[str] | None
        categories: list[SemanticCategory]

    async def get_setid_config(
        self,
        *,
        set_id: SetIdT,
    ) -> Config: ...

    async def register_set_id_org_set(
        self,
        *,
        set_id: SetIdT,
        org_set_id: str,
    ) -> None: ...

    @dataclass(frozen=True)
    class Category:
        """Represents a semantic category as stored in the database."""

        id: CategoryIdT
        name: str
        prompt: str
        description: str | None

    async def get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> Category | None: ...

    async def get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> list[SetIdT]: ...

    async def create_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT: ...

    async def clone_category(
        self,
        *,
        category_id: CategoryIdT,
        new_set_id: SetIdT,
        new_name: str,
    ) -> CategoryIdT: ...

    async def delete_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> None: ...

    async def add_disabled_category_to_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None: ...

    async def remove_disabled_category_from_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None: ...

    async def create_org_set_category(
        self,
        *,
        org_set_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT: ...

    async def get_org_set_categories(
        self,
        *,
        org_set_id: str,
    ) -> list[SemanticCategory]: ...

    @dataclass(frozen=True)
    class Tag:
        """Represents a tag associated with a category as represented in the database."""

        id: str
        name: str
        description: str

    async def get_tag(
        self,
        *,
        tag_id: str,
    ) -> Tag | None: ...

    async def add_tag(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        description: str,
    ) -> TagIdT: ...

    async def update_tag(
        self,
        *,
        tag_id: str,
        tag_name: str,
        tag_description: str,
    ) -> None: ...

    async def delete_tag(
        self,
        *,
        tag_id: str,
    ) -> None: ...

    async def add_org_set_id(
        self,
        *,
        org_id: str,
        org_level_set: bool = False,
        metadata_tags: list[str],
        name: str | None = None,
        description: str | None = None,
    ) -> str: ...

    async def list_org_set_ids(self, *, org_id: str) -> list[OrgSetTypeEntry]: ...

    async def delete_org_set_id(self, *, org_set_id: str) -> None: ...
