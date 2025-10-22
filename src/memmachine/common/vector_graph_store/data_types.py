"""
Data types for nodes and edges in a vector graph store.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

# Types that can be used as property values in nodes and edges.
Property = (
    bool
    | int
    | float
    | str
    | datetime
    | list[bool]
    | list[int]
    | list[float]
    | list[str]
    | list[datetime]
    | None
)


@dataclass(kw_only=True)
class Node:
    uuid: UUID
    properties: dict[str, Property] = field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)


@dataclass(kw_only=True)
class Edge:
    uuid: UUID
    source_uuid: UUID
    target_uuid: UUID
    properties: dict[str, Property] = field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)
