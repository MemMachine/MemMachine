"""Data types for vector store."""

from collections.abc import Mapping
from typing import Annotated
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    field_validator,
)

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
    PropertyValue,
    SimilarityMetric,
)

from .utils import validate_identifier


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
store indexes for filtering, with the type its values hold. Declared once,
at construction, from the system keys of the consumer the store is built for.
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
    What a partition was created under: its store's dimensions, metric and schema.

    Recorded beside the partition so a store built with other dimensions,
    another metric or another declared schema fails loudly instead of
    reading columns or vectors that are not there.
    """

    vector_dimensions: int
    similarity_metric: SimilarityMetric
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


class VectorStorePartitionHandleStaleError(Exception):
    """A partition handle outlived the partition incarnation it was bound to."""

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
    """The store exhausted its internal attempts; diagnose the cause.

    Raised when an operation kept failing in a way that should not recur
    under normal operation. An immediate retry is unlikely to succeed;
    the underlying error is chained as the cause.
    """


class VectorStorePartitionSchemaMismatchError(Exception):
    """
    Raised when a partition's recorded schema differs from its store's.

    A store built with one dimensionality, one metric and one
    `indexed_properties` schema holds columns and indexes for exactly those;
    a partition created under others cannot be served without a migration,
    which nothing here performs.
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


class Record(BaseModel):
    """
    A record in the vector store.

    Attributes:
        uuid (UUID):
            Unique identifier for the record.
        vector (list[float] | None):
            Vector for similarity search.
            `None` is not allowed on input.
            `None` on output means the vector was not requested (`return_vector=False`)
            (default: None).
        properties (dict[str, PropertyValue] | None):
            Property key-value pairs.
            Use `{}` to represent missing properties; `None` on input is treated as `{}`.
            `None` on output means the properties were not requested (`return_properties=False`)
            (default: None).
    """

    uuid: UUID
    vector: list[float] | None = None
    properties: dict[str, PropertyValue] | None = None

    @field_validator("properties")
    @classmethod
    def _validate_property_keys(
        cls, v: dict[str, PropertyValue] | None
    ) -> dict[str, PropertyValue] | None:
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
        score (float):
            The meaning depends on the store's `SimilarityMetric`:
            - *cosine*: cosine similarity in [-1, 1].
            - *dot*: raw dot product [0, inf).
            - *euclidean*: Euclidean distance [0, inf).
            - *manhattan*: Manhattan distance [0, inf).

            Use `SimilarityMetric.higher_is_better` to determine which
            direction indicates a better match.
        record (Record):
            The matched record.
    """

    score: float
    record: Record


class QueryResult(BaseModel):
    """
    Result of a vector store query.

    Matches are ordered from best to worst.
    """

    matches: list[QueryMatch]
