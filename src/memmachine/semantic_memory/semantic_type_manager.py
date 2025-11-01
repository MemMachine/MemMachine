from abc import ABC, abstractmethod

from pydantic import BaseModel

from memmachine.semantic_memory.semantic_model import SemanticType


# TODO: Will be used to dynamically create, delete, and update prompts.
class SemanticTypeManager(ABC):
    @abstractmethod
    async def create_semantic_type(
        self, owner_id: str, semantic_type: SemanticType
    ) -> SemanticType:
        raise NotImplementedError

    @abstractmethod
    async def get_semantic_types(self, owner_ids: list[str]) -> list[SemanticType]:
        raise NotImplementedError

    @abstractmethod
    async def delete_semantic_types(self, semantic_type_id: list[int]):
        raise NotImplementedError

    class SemanticTypeSearchOptions(BaseModel):
        name: list[str] | None = None
        tag: list[str] | None = None
        owner_ids: list[str] | None = None
        k: int = 1000

    @abstractmethod
    async def search_semantic_types(
        self, options: SemanticTypeSearchOptions
    ) -> list[SemanticType]:
        raise NotImplementedError

    @abstractmethod
    async def delete_search_semantic_types(
        self, options: SemanticTypeSearchOptions
    ) -> list[int]:
        raise NotImplementedError
