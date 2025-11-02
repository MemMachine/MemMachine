"""
Data types for nodes and edges in a vector graph store.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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


class EntityType(Enum):
    NODE = "node"
    EDGE = "edge"


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


def mangle_property_name(property_name: str) -> str:
    """
    Mangle the property name
    to avoid conflicts with field names.

    Args:
        property_name (str):
            The original property name.

    Returns:
        str: The mangled property name.
    """
    return f"property_{property_name}"


def demangle_property_name(mangled_property_name: str) -> str:
    """
    Demangle the property name
    to retrieve the original property name.

    Args:
        mangled_property_name (str):
            The mangled property name.

    Returns:
        str: The original property name.
    """
    return mangled_property_name.removeprefix("property_")


def is_mangled_property_name(candidate_name: str) -> bool:
    """
    Check if the candidate name
    is a mangled property name.

    Args:
        candidate_name (str):
            The candidate property name.

    Returns:
        bool:
            Whether the candidate name is a mangled property name.
    """
    return candidate_name.startswith("property_")
