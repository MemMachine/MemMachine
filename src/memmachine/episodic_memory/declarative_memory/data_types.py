from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

FilterablePropertyValue = bool | int | str
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ContentType(Enum):
    STRING = "string"


@dataclass(kw_only=True)
class Episode:
    uuid: UUID
    episode_type: str
    content_type: ContentType
    content: Any
    timestamp: datetime
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None


@dataclass(kw_only=True)
class EpisodeCluster:
    uuid: UUID
    episodes: list[Episode] = field(default_factory=list)
    timestamp: datetime | None = None
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None


@dataclass(kw_only=True)
class Derivative:
    uuid: UUID
    derivative_type: str
    content_type: ContentType
    content: Any
    timestamp: datetime | None = None
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None


def mangle_filterable_property_key(key: str) -> str:
    """
    Mangle the filterable property key
    to avoid conflicts with field names.

    Args:
        key (str):
            The original filterable property key.

    Returns:
        str: The mangled filterable property key.
    """
    return f"filterable_{key}"


def demangle_filterable_property_key(mangled_key: str) -> str:
    """
    Demangle the filterable property key
    to retrieve the original key.

    Args:
        mangled_key (str):
            The mangled filterable property key.

    Returns:
        str: The original filterable property key.
    """
    return mangled_key.removeprefix("filterable_")


def is_mangled_filterable_property_key(candidate_key: str) -> bool:
    """
    Check if the candidate key is a mangled filterable property key.

    Args:
        candidate_key (str):
            The candidate key to check.

    Returns:
        bool:
            Whether the candidate key is a mangled filterable property key,
    """
    return candidate_key.startswith("filterable_")
