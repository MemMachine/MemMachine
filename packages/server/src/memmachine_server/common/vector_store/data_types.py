"""Data types for vector store."""

from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer, field_validator

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)

from .utils import validate_identifier


class VectorStoreCollectionConfig(BaseModel):
    """
    Configuration for a logical collection in a vector store.

    Attributes:
        vector_dimensions (int):
            Dimensionality of vectors stored in the collection.
        indexed_properties_schema (dict[str, type[PropertyValue]]):
            Schema suggesting which properties should be indexed for filtering.
    """

    vector_dimensions: int
    indexed_properties_schema: dict[str, type[PropertyValue]] = Field(
        default_factory=dict
    )

    @field_validator("indexed_properties_schema", mode="after")
    @classmethod
    def _validate_property_keys(
        cls, v: dict[str, type[PropertyValue]]
    ) -> dict[str, type[PropertyValue]]:
        for key in v:
            if not validate_identifier(key):
                raise ValueError(
                    f"Property key {key!r} must match [a-z0-9_]+ and be at most 32 bytes"
                )
        return v

    @field_validator("indexed_properties_schema", mode="before")
    @classmethod
    def _coerce_indexed_properties_schema(cls, v: object) -> object:
        if v is None:
            return {}
        if isinstance(v, Mapping):
            result = {}
            for key, value in v.items():
                if isinstance(value, str):
                    if value not in PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE:
                        raise ValueError(
                            f"Unknown property type name {value!r} for key {key!r}."
                        )
                    result[key] = PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE[value]
                else:
                    result[key] = value
            return result

        return v

    @field_serializer("indexed_properties_schema")
    def _serialize_indexed_properties_schema(
        self, v: dict[str, type[PropertyValue]]
    ) -> dict[str, str]:
        return {
            k: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[val] for k, val in sorted(v.items())
        }


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
            Property key-value pairs.
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
