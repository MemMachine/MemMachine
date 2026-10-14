"""Data types for vector store."""

from collections.abc import Collection, Iterable, Mapping
from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    FiniteFloat,
    field_validator,
)

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
    PropertyValue,
)

from .utils import validate_identifier

MIN_INT_VALUE = -(2**63)
MAX_INT_VALUE = 2**63 - 1
"""The range of an int a record or a filter holds: every backend holds an int in 64 signed bits."""


def _coerce_property_types(value: object) -> object:
    # Deployment configuration names a type ("str", "datetime"); code passes
    # the type itself. Both are accepted, and the names are resolved here.
    if not isinstance(value, Mapping):
        return value
    resolved: dict[object, object] = {}
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
The one schema a store declares for every partition it holds: each key a
store indexes and filters on, with the type its values hold. Declared once,
at construction, from the system keys of the consumer the store is built
for; a record or a filter naming any other key is rejected.
"""


def indexed_property_names(
    indexed_properties: Mapping[str, PropertyType],
) -> dict[str, str]:
    """The JSON form of a declared schema: each type by its name."""
    return {
        key: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]
        for key, property_type in sorted(indexed_properties.items())
    }


def validate_vector_store_name(name: str) -> None:
    """Raise ValueError unless `name` can name a vector store on every backend."""
    if not validate_identifier(name):
        raise ValueError(
            f"Vector store name {name!r} must match [a-z0-9_]+ and be at most 32 bytes."
        )


class PartitionSchema(BaseModel):
    """
    What a partition was created under: its store's dimensions and schema.

    Recorded beside the partition so a store built with other dimensions or
    another declared schema fails loudly instead of reading columns or
    vectors that are not there.
    """

    vector_dimensions: int
    indexed_properties: dict[str, str]
    """The declared schema, each type by its name."""


class VectorStorePartitionAlreadyExistsError(Exception):
    """Raised when creating a partition that already exists."""

    def __init__(self, vector_store_name: str, partition_key: str) -> None:
        """Initialize with the vector store and the key of the existing partition."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        super().__init__(
            f"Partition {partition_key!r} of vector store {vector_store_name!r} already exists."
        )


class VectorStorePartitionPendingError(Exception):
    """Raised when opening a partition whose creation has not completed."""

    def __init__(
        self,
        vector_store_name: str,
        partition_key: str,
        registered_at: datetime,
        schema: PartitionSchema,
    ) -> None:
        """Initialize with the pending partition's vector store, key, registration time, and schema."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        self.registered_at = registered_at
        self.schema = schema
        super().__init__(
            f"Partition {partition_key!r} of vector store {vector_store_name!r} has "
            f"been pending since {registered_at.isoformat()}; if its creation was "
            "abandoned, delete it to create it again."
        )


class VectorStorePartitionDeletedError(Exception):
    """Raised when a partition is deleted before its creation completes."""

    def __init__(self, vector_store_name: str, partition_key: str) -> None:
        """Initialize with the vector store and the key of the deleted partition."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        super().__init__(
            f"Partition {partition_key!r} of vector store {vector_store_name!r} was "
            "deleted before its creation completed"
        )


class VectorStorePartitionHandleStaleError(Exception):
    """Raised when a handle is used after its partition was deleted."""

    def __init__(self, vector_store_name: str, partition_key: str) -> None:
        """Record the vector store and the key the stale handle belonged to."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        super().__init__(
            f"Stale handle for partition {partition_key!r} of vector store "
            f"{vector_store_name!r}: the partition was deleted (or re-created) after this "
            "handle was bound"
        )


class VectorStoreAttemptsExhaustedError(Exception):
    """Raised when an operation gave up after repeated attempts that made no progress."""


class VectorStorePartitionSchemaMismatchError(Exception):
    """
    Raised when a partition's recorded schema differs from its store's.

    A store built with one dimensionality and one `indexed_properties`
    schema holds columns and indexes for exactly those; a partition created
    under others cannot be served without a migration, which nothing here
    performs.
    """

    def __init__(
        self,
        vector_store_name: str,
        partition_key: str,
        stored: PartitionSchema,
        declared: PartitionSchema,
    ) -> None:
        """Initialize with the partition and the two schemas."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        self.stored = stored
        self.declared = declared
        super().__init__(
            f"Partition {partition_key!r} of vector store {vector_store_name!r} was created "
            f"under {stored.model_dump(mode='json')}, but the store declares "
            f"{declared.model_dump(mode='json')}."
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
    """
    Raised when a record's or a filter's value is not of its key's declared type.

    An int is of a float key's type, as the float it equals, and a bool is
    never a number; a float key holds and is compared with finite values
    only, and any key with ints in 64 signed bits only.
    """

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
    A record to write to a vector store partition.

    Attributes:
        uuid (UUID):
            Unique identifier for the record.
        vector (list[float]):
            Vector for similarity search, of finite coordinates.
        properties (dict[str, PropertyValue]):
            Property key-value pairs to filter on
            (default: `{}`).
    """

    uuid: UUID
    vector: list[FiniteFloat]
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
