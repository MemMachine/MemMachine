"""Data types for vector store."""

from collections.abc import Mapping
from datetime import datetime
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    FiniteFloat,
    field_serializer,
    field_validator,
)

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
    SimilarityMetric,
)

from .utils import validate_identifier


class VectorStoreCollectionConfig(BaseModel):
    """
    Configuration for a logical collection in a vector store.

    Attributes:
        vector_dimensions (int):
            Dimensionality of vectors stored in the collection.
        similarity_metric (SimilarityMetric):
            Metric used to compare vectors.
        indexed_properties_schema (dict[str, type[PropertyValue]]):
            Schema suggesting which properties should be indexed for filtering.
    """

    vector_dimensions: int
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
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


class VectorStoreCollectionPendingError(Exception):
    """Raised when opening a collection whose creation has not completed."""

    def __init__(
        self,
        namespace: str,
        name: str,
        registered_at: datetime,
        config: VectorStoreCollectionConfig,
    ) -> None:
        """Initialize with the pending collection's namespace, name, registration time, and configuration."""
        self.namespace = namespace
        self.name = name
        self.registered_at = registered_at
        self.config = config
        super().__init__(
            f"Collection ({namespace!r}, {name!r}) has been pending since "
            f"{registered_at.isoformat()}; if its creation was abandoned, "
            "delete it to create it again."
        )


class VectorStoreCollectionDeletedError(Exception):
    """Raised when a collection is deleted before its creation completes."""

    def __init__(self, namespace: str, name: str) -> None:
        """Initialize with the namespace and name of the deleted collection."""
        self.namespace = namespace
        self.name = name
        super().__init__(
            f"Collection ({namespace!r}, {name!r}) was deleted before its "
            "creation completed"
        )


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


class VectorStoreCollectionHandleStaleError(Exception):
    """Raised when a handle is used after its collection was deleted."""

    def __init__(self, namespace: str, name: str) -> None:
        """Record the namespace and name the stale handle belonged to."""
        self.namespace = namespace
        self.name = name
        super().__init__(
            f"Stale handle for collection ({namespace!r}, {name!r}): the collection "
            "was deleted (or re-created) after this handle was bound"
        )


class VectorStoreAttemptsExhaustedError(Exception):
    """Raised when an operation gave up after repeated attempts that made no progress."""


class Record(BaseModel):
    """
    A record to write to a vector store collection.

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
        score (float):
            The meaning depends on the collection's `SimilarityMetric`:
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
