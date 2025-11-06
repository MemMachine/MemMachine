from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

FilterablePropertyValue = bool | int | str
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ContentType(Enum):
    MESSAGE = "message"
    TEXT = "text"


@dataclass(kw_only=True)
class Episode:
    uuid: UUID
    timestamp: datetime
    source: str
    content_type: ContentType
    content: Any
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None

    def __eq__(self, other):
        if not isinstance(other, Episode):
            return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)


@dataclass(kw_only=True)
class Chunk:
    uuid: UUID
    episode_uuid: UUID
    sequence_number: int
    timestamp: datetime
    source: str
    content_type: ContentType
    content: Any
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict
    )
    user_metadata: JSONValue = None

    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)


@dataclass(kw_only=True)
class Derivative:
    uuid: UUID
    content_type: ContentType
    content: Any
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict
    )

    def __eq__(self, other):
        if not isinstance(other, Derivative):
            return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)


def mangle_filterable_property_key(key: str) -> str:
    return f"filterable_{key}"


def demangle_filterable_property_key(mangled_key: str) -> str:
    return mangled_key.removeprefix("filterable_")


def is_mangled_filterable_property_key(candidate_key: str) -> bool:
    return candidate_key.startswith("filterable_")
