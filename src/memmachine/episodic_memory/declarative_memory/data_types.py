"""Data structures for declarative episodic memory entries."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import JsonValue

from memmachine.common.data_types import FilterablePropertyValue


class ContentType(Enum):
    """Types of content stored in declarative memory."""

    MESSAGE = "message"
    TEXT = "text"


@dataclass(kw_only=True)
class Episode:
    """A single episodic memory entry."""

    uuid: UUID
    timestamp: datetime
    source: str
    content_type: ContentType
    content: Any
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict,
    )
    user_metadata: JsonValue = None

    def __eq__(self, other: object) -> bool:
        """Compare episodes by UUID."""
        if not isinstance(other, Episode):
            return False
        return self.uuid == other.uuid

    def __hash__(self) -> int:
        """Hash an episode by its UUID."""
        return hash(self.uuid)


@dataclass(kw_only=True)
class Derivative:
    """A derived episodic memory linked to a source episode."""

    uuid: UUID
    timestamp: datetime
    source: str
    content_type: ContentType
    content: Any
    filterable_properties: dict[str, FilterablePropertyValue] = field(
        default_factory=dict,
    )

    def __eq__(self, other: object) -> bool:
        """Compare derivatives by UUID."""
        if not isinstance(other, Derivative):
            return False
        return self.uuid == other.uuid

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


_MANGLE_FILTERABLE_PROPERTY_KEY_PREFIX = "filterable_"


def mangle_filterable_property_key(key: str) -> str:
    """Prefix filterable property keys with the mangling token."""
    return _MANGLE_FILTERABLE_PROPERTY_KEY_PREFIX + key


def demangle_filterable_property_key(mangled_key: str) -> str:
    """Remove the mangling prefix from a filterable property key."""
    return mangled_key.removeprefix(_MANGLE_FILTERABLE_PROPERTY_KEY_PREFIX)


def is_mangled_filterable_property_key(candidate_key: str) -> bool:
    """Check whether the provided key contains the mangling prefix."""
    return candidate_key.startswith(_MANGLE_FILTERABLE_PROPERTY_KEY_PREFIX)
