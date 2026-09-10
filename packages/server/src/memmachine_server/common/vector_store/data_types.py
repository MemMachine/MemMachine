"""Data types for vector store."""

from collections.abc import Collection, Iterable, Mapping
from typing import Annotated
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
)

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
    PropertyValue,
)

from .utils import validate_identifier


def _coerce_property_types(value: object) -> object:
    # Deployment configuration names a type ("str", "datetime"); code passes
    # the type itself. Both are accepted, and the names are resolved here.
    if not isinstance(value, Mapping):
        return value
    resolved: dict[str, object] = {}
    for key, property_type in value.items():
        if isinstance(property_type, str):
            if property_type not in PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE:
                raise ValueError(
                    f"Unknown property type name {property_type!r} for key {key!r}; "
                    f"expected one of {sorted(PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE)}"
                )
            resolved[key] = PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE[property_type]
        else:
            resolved[key] = property_type
    return resolved


def _validate_property_keys(value: dict[str, PropertyType]) -> dict[str, PropertyType]:
    for key in value:
        if not validate_identifier(key):
            raise ValueError(
                f"Property key {key!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
    return value


IndexedProperties = Annotated[
    dict[str, PropertyType],
    BeforeValidator(_coerce_property_types),
    AfterValidator(_validate_property_keys),
]
"""
The one schema a store declares for every collection it holds: each key a
store indexes and filters on, with the type its values hold. Declared once,
from deployment configuration merged with the system keys of the consumer the
store is built for; a record or a filter naming any other key is rejected.
"""


def indexed_property_names(
    indexed_properties: Mapping[str, PropertyType],
) -> dict[str, str]:
    """The JSON form of a declared schema: each type by its name."""
    return {
        key: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]
        for key, property_type in sorted(indexed_properties.items())
    }


class VectorStoreCollectionConfig(BaseModel):
    """
    Configuration for a logical collection in a vector store.

    Attributes:
        vector_dimensions (int):
            Dimensionality of vectors stored in the collection.
    """

    vector_dimensions: int


class VectorStoreCollectionAlreadyExistsError(Exception):
    """Raised when creating a collection that already exists."""

    def __init__(self, namespace: str, name: str) -> None:
        """Initialize with the namespace and name of the existing collection."""
        self.namespace = namespace
        self.name = name
        super().__init__(f"Collection ({namespace!r}, {name!r}) already exists.")


class VectorStoreCollectionConfigMismatchError(Exception):
    """Raised when opening a collection with a different configuration than it was created with."""

    def __init__(
        self,
        namespace: str,
        name: str,
        existing_config: VectorStoreCollectionConfig,
        requested_config: VectorStoreCollectionConfig,
    ) -> None:
        """Initialize with the namespace, name, and configurations."""
        self.namespace = namespace
        self.name = name
        self.existing_config = existing_config
        self.requested_config = requested_config
        super().__init__(
            f"Collection ({namespace!r}, {name!r}) already exists with a different configuration. "
            f"Existing config: {existing_config.model_dump_json()}, "
            f"requested config: {requested_config.model_dump_json()}."
        )


class IndexedPropertiesMismatchError(Exception):
    """
    Raised when a collection's stored schema differs from the store's declaration.

    A store built with one `indexed_properties` schema holds columns and
    indexes for exactly those keys; a collection created under another
    schema cannot be served without a migration, which nothing here performs.
    """

    def __init__(
        self,
        namespace: str,
        name: str,
        stored: Mapping[str, str],
        declared: Mapping[str, str],
    ) -> None:
        """Initialize with the collection and the two schemas, as type names."""
        self.namespace = namespace
        self.name = name
        self.stored = dict(stored)
        self.declared = dict(declared)
        super().__init__(
            f"Collection ({namespace!r}, {name!r}) was created with indexed "
            f"properties {self.stored}, but the store declares {self.declared}."
        )


class UndeclaredPropertyKeyError(ValueError):
    """Raised when a record or a filter names a key the store has not declared."""

    def __init__(self, keys: Iterable[str], declared: Collection[str]) -> None:
        """Initialize with the offending keys and the declared ones."""
        self.keys = sorted(set(keys))
        self.declared = sorted(declared)
        super().__init__(
            f"Property keys {self.keys} are not declared by the vector store; "
            f"declared keys: {self.declared}."
        )


class PropertyTypeMismatchError(ValueError):
    """Raised when a record's value is not of its key's declared type."""

    def __init__(self, key: str, declared: PropertyType, value: PropertyValue) -> None:
        """Initialize with the key, its declared type and the offending value."""
        self.key = key
        self.declared = declared
        self.value = value
        super().__init__(
            f"Property {key!r} is declared as "
            f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared]}, "
            f"got {type(value).__name__} {value!r}."
        )


class UnsupportedFilterError(ValueError):
    """Raised when a filter uses a node the store cannot evaluate during a search."""

    def __init__(self, nodes: Iterable[type], supported: Collection[type]) -> None:
        """Initialize with the offending node classes and the supported ones."""
        self.nodes = sorted({node.__name__ for node in nodes})
        self.supported = sorted(node.__name__ for node in supported)
        super().__init__(
            f"Filter nodes {self.nodes} are not evaluated by this vector store; "
            f"supported nodes: {self.supported}."
        )


class Record(BaseModel):
    """
    A record to write to a vector store collection.

    Records are only ever written. A collection stores vectors to search
    them and properties to filter on them, and answers a query with
    `QueryMatch`; neither a vector nor a property is read back out.

    Attributes:
        uuid (UUID):
            Unique identifier for the record.
        vector (list[float]):
            Vector for similarity search.
        properties (dict[str, PropertyValue]):
            Property key-value pairs, each key one the store declares.
            Stored for property filtering; never returned
            (default: `{}`).
    """

    uuid: UUID
    vector: list[float]
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("properties", mode="after")
    @classmethod
    def _validate_property_keys(
        cls, v: dict[str, PropertyValue]
    ) -> dict[str, PropertyValue]:
        if v:
            for key in v:
                if not validate_identifier(key):
                    raise ValueError(
                        f"Property key {key!r} must match [a-z0-9_]+ and be at most 32 bytes"
                    )
        return v

    def __hash__(self) -> int:
        """Hash a record by its UID."""
        return hash(self.uuid)


class QueryMatch(BaseModel):
    """
    A single vector store query match.

    Attributes:
        cosine_similarity (float):
            Cosine similarity between the query vector and the matched
            record's vector, in [-1, 1]. Higher is a better match.
        record_uuid (UUID):
            UUID of the matched record.
    """

    cosine_similarity: float
    record_uuid: UUID


class QueryResult(BaseModel):
    """
    Result of a vector store query.

    Matches are ordered from best to worst.
    """

    matches: list[QueryMatch]
