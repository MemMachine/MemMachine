"""
Data types for vector store.

Imports nothing from this package; the rest of it builds on these types.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from typing import Annotated
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
)

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
    PropertyValue,
    SimilarityMetric,
)


def _coerce_property_types(value: object) -> object:
    # A caller may name a type or pass the type itself.
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
        if not is_valid_identifier(key):
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
The properties a store declares for every partition it holds: each key the
store indexes and filters on, with the type its values hold. A record or a
filter naming any other key is rejected.
"""


def validate_vector_store_name(name: str) -> None:
    """Raise ValueError unless `name` can name a vector store on every backend."""
    if not is_valid_identifier(name):
        raise ValueError(
            f"Vector store name {name!r} must match [a-z0-9_]+ and be at most 32 bytes."
        )


# Matched with fullmatch: `$` also matches before a trailing newline.
_IDENTIFIER_RE = re.compile(r"[a-z0-9_]+")
_IDENTIFIER_MAX_BYTES = 32


def is_valid_identifier(value: str) -> bool:
    """
    Whether `value` is portable as a name wherever a backend restricts names.

    Such a name holds only lowercase letters, digits, and underscores, and is
    at most 32 bytes, so every backend accepts it.
    """
    return (
        bool(_IDENTIFIER_RE.fullmatch(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )


class VectorStoreSchema(BaseModel):
    """
    The shape of a vector store's records and the metric its queries score by.

    Attributes:
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        similarity_metric (SimilarityMetric):
            The metric every query of the store scores by (default: cosine).
        indexed_properties (IndexedProperties):
            The properties every partition of the store declares.
    """

    model_config = ConfigDict(frozen=True)

    vector_dimensions: int = Field(
        ..., gt=0, description="Dimensionality of every vector in the store"
    )
    similarity_metric: SimilarityMetric = Field(
        SimilarityMetric.COSINE,
        description="The metric every query of the store scores by",
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The properties every partition of the store declares",
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
                if not is_valid_identifier(key):
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
        record_uuid (UUID):
            UUID of the matched record.
    """

    score: float
    record_uuid: UUID


class QueryResult(BaseModel):
    """
    Result of a vector store query.

    Matches are ordered from best to worst.
    """

    matches: list[QueryMatch]
