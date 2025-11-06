"""
Data types for nodes and edges in a vector graph store.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID

from memmachine.common.data_types import SimilarityMetric

# Types that can be used as property values in nodes and edges.
PropertyValue = (
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

OrderedPropertyValue = int | float | str | datetime


class EntityType(Enum):
    NODE = "node"
    EDGE = "edge"


@dataclass(kw_only=True)
class Node:
    uuid: UUID
    properties: dict[str, PropertyValue] = field(default_factory=dict)
    embeddings: dict[str, tuple[list[float], SimilarityMetric]] = field(
        default_factory=dict
    )

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
    properties: dict[str, PropertyValue] = field(default_factory=dict)
    embeddings: dict[str, tuple[list[float], SimilarityMetric]] = field(
        default_factory=dict
    )

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)


def mangle_property_name(property_name: str) -> str:
    """
    Mangle the property name
    to avoid conflicts with other names.

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
            The candidate name.

    Returns:
        bool:
            Whether the candidate name is a mangled property name.
    """
    return candidate_name.startswith("property_")


def mangle_embedding_name(embedding_name: str) -> str:
    """
    Mangle the embedding name
    to avoid conflicts with other names.

    Args:
        embedding_name (str):
            The original embedding name.

    Returns:
        str: The mangled embedding name.
    """
    return f"embedding_{embedding_name}"


def demangle_embedding_name(mangled_embedding_name: str) -> str:
    """
    Demangle the embedding name
    to retrieve the original embedding name.

    Args:
        mangled_embedding_name (str):
            The mangled embedding name.

    Returns:
        str: The original embedding name.
    """
    return mangled_embedding_name.removeprefix("embedding_")


def is_mangled_embedding_name(candidate_name: str) -> bool:
    """
    Check if the candidate name
    is a mangled embedding name.

    Args:
        candidate_name (str):
            The candidate name.

    Returns:
        bool:
            Whether the candidate name is a mangled embedding name.
    """
    return candidate_name.startswith("embedding_")
