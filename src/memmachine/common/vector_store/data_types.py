"""Data types for vector store."""

from dataclasses import dataclass
from uuid import UUID

from memmachine.common.data_types import PropertyValue


@dataclass(kw_only=True)
class Record:
    """A record in the vector store."""

    uuid: UUID
    vector: list[float] | None = None
    properties: dict[str, PropertyValue] | None = None

    def __eq__(self, other: object) -> bool:
        """Compare nodes by UID, vector, and properties."""
        if not isinstance(other, Record):
            return False
        return (
            self.uuid == other.uuid
            and self.vector == other.vector
            and self.properties == other.properties
        )

    def __hash__(self) -> int:
        """Hash a record by its UID."""
        return hash(self.uuid)


@dataclass(kw_only=True)
class QueryResult:
    """Result of a vector store query."""

    score: float
    record: Record
