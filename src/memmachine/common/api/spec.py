"""API v2 specification models for request and response structures."""

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Self

import regex
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from memmachine.common.api import EpisodeType, MemoryType
from memmachine.common.api.doc import Examples, SpecDoc

DEFAULT_ORG_AND_PROJECT_ID = "universal"


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Client-safe DTOs
#
# NOTE:
# These models intentionally live in this API spec module (and do NOT import the
# internal/core models) so that the client distribution can import the API schema
# without pulling in server-only packages.
# --------------------------------------------------------------------------------------

EpisodeIdT = str


class ContentType(Enum):
    """Enumeration for the type of content within an Episode."""

    STRING = "string"


class EpisodeEntry(BaseModel):
    """Payload used when creating a new episode entry."""

    content: str
    producer_id: str
    producer_role: str
    produced_for_id: str | None = None
    episode_type: EpisodeType | None = None
    metadata: dict[str, JsonValue] | None = None
    created_at: AwareDatetime | None = None


class EpisodeResponse(EpisodeEntry):
    """Episode data returned in search responses."""

    uid: EpisodeIdT
    score: float | None = None


class Episode(BaseModel):
    """Episode data returned in list responses."""

    uid: EpisodeIdT
    content: str
    session_key: str
    created_at: AwareDatetime

    producer_id: str
    producer_role: str
    produced_for_id: str | None = None

    sequence_num: int = 0

    episode_type: EpisodeType = EpisodeType.MESSAGE
    content_type: ContentType = ContentType.STRING
    filterable_metadata: dict[str, Any] | None = None
    metadata: dict[str, JsonValue] | None = None

    def __hash__(self) -> int:
        """Hash an episode by its UID."""
        return hash(self.uid)


SetIdT = str
FeatureIdT = str


class SemanticFeature(BaseModel):
    """Semantic memory entry returned in API responses."""

    class Metadata(BaseModel):
        """Storage metadata for a semantic feature, including id and citations."""

        citations: list[EpisodeIdT] | None = None
        id: FeatureIdT | None = None
        other: dict[str, Any] | None = None

    set_id: SetIdT | None = None
    category: str
    tag: str
    feature_name: str
    value: str
    metadata: Metadata = Field(default_factory=Metadata)


class InvalidNameError(ValueError):
    """Custom error for invalid names."""


class InvalidTimestampError(ValueError):
    """Custom error for invalid timestamps."""


def _is_valid_name(v: str) -> str:
    if not regex.fullmatch(r"^[\p{L}\p{N}_:-]+$", v):
        raise InvalidNameError(
            "ID can only contain letters, numbers, underscore, hyphen, "
            f"colon, or Unicode characters, found: '{v}'",
        )
    return v


def _validate_int_compatible(v: str) -> str:
    try:
        int(v)
    except ValueError as e:
        raise ValueError("ID must be int-compatible") from e
    return v


IntCompatibleId = Annotated[str, AfterValidator(_validate_int_compatible), Field(...)]


SafeId = Annotated[str, AfterValidator(_is_valid_name), Field(...)]
SafeIdWithDefault = Annotated[SafeId, Field(default=DEFAULT_ORG_AND_PROJECT_ID)]


class _WithOrgAndProj(BaseModel):
    org_id: Annotated[
        SafeIdWithDefault,
        Field(description=SpecDoc.ORG_ID, examples=Examples.ORG_ID),
    ]
    project_id: Annotated[
        SafeIdWithDefault,
        Field(description=SpecDoc.PROJECT_ID, examples=Examples.PROJECT_ID),
    ]


class ProjectConfig(BaseModel):
    """
    Project configuration model.

    This section defines which reranker and embedder models should be used for
    the project.  If any field is left empty (""), the system automatically falls
    back to the globally configured defaults in the server configuration file.
    """

    reranker: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.RERANKER_ID,
            examples=Examples.RERANKER,
        ),
    ]

    embedder: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.EMBEDDER_ID,
            examples=Examples.EMBEDDER,
        ),
    ]


class CreateProjectSpec(BaseModel):
    """
    Specification model for creating a new project.

    A project belongs to an organization and has its own identifiers,
    description, and configuration. The project ID must be unique within
    the organization.
    """

    org_id: Annotated[
        SafeId,
        Field(
            description=SpecDoc.ORG_ID,
            examples=Examples.ORG_ID,
        ),
    ]

    project_id: Annotated[
        SafeId,
        Field(
            description=SpecDoc.PROJECT_ID,
            examples=Examples.PROJECT_ID,
        ),
    ]

    description: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.PROJECT_DESCRIPTION,
            examples=Examples.PROJECT_DESCRIPTION,
        ),
    ]

    config: ProjectConfig = Field(
        default_factory=ProjectConfig,
        description=SpecDoc.PROJECT_CONFIG,
    )


class ProjectResponse(BaseModel):
    """
    Response model returned after project operations (e.g., creation, update, fetch).

    Contains the resolved identifiers and configuration of the project as stored
    in the system. Field formats follow the same validation rules as in
    `CreateProjectSpec`.
    """

    org_id: Annotated[
        SafeId,
        Field(description=SpecDoc.ORG_ID_RETURN),
    ]

    project_id: Annotated[
        SafeId,
        Field(description=SpecDoc.PROJECT_ID_RETURN),
    ]

    description: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.PROJECT_DESCRIPTION,
        ),
    ]

    config: Annotated[
        ProjectConfig,
        Field(
            default_factory=ProjectConfig,
            description=SpecDoc.PROJECT_CONFIG,
        ),
    ]


class GetProjectSpec(BaseModel):
    """
    Specification model for retrieving a project.

    This model defines the parameters required to fetch an existing project.
    Both the organization ID and project ID follow the standard `SafeId`
    validation rules.

    The combination of `org_id` and `project_id` uniquely identifies the
    project to retrieve.
    """

    org_id: Annotated[
        SafeId,
        Field(
            description=SpecDoc.ORG_ID,
            examples=Examples.ORG_ID,
        ),
    ]
    project_id: Annotated[
        SafeId,
        Field(
            description=SpecDoc.PROJECT_ID,
            examples=Examples.PROJECT_ID,
        ),
    ]


class EpisodeCountResponse(BaseModel):
    """
    Response model representing the number of episodes associated with a project.

    This model is typically returned by analytics or monitoring endpoints
    that track usage activity (e.g., number of computation episodes, workflow
    runs, or operational cycles).

    The count reflects the current recorded total at the time of the request.
    """

    count: Annotated[
        int,
        Field(
            ...,
            description=SpecDoc.EPISODE_COUNT,
            ge=0,
        ),
    ]


class DeleteProjectSpec(BaseModel):
    """
    Specification model for deleting a project.

    This model defines the identifiers required to delete a project from a
    specific organization. The identifiers must comply with the `SafeId`
    rules.

    Deletion operations are typically irreversible and remove both metadata and
    associated configuration for the specified project.
    """

    org_id: Annotated[
        SafeId,
        Field(
            description=SpecDoc.ORG_ID,
            examples=Examples.ORG_ID,
        ),
    ]
    project_id: Annotated[
        SafeId,
        Field(
            description=SpecDoc.PROJECT_ID,
            examples=Examples.PROJECT_ID,
        ),
    ]


TimestampInput = datetime | int | float | str | None


class MemoryMessage(BaseModel):
    """Model representing a memory message."""

    content: Annotated[
        str,
        Field(..., description=SpecDoc.MEMORY_CONTENT),
    ]
    producer: Annotated[
        str,
        Field(
            default="user",
            description=SpecDoc.MEMORY_PRODUCER,
        ),
    ]
    produced_for: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.MEMORY_PRODUCE_FOR,
        ),
    ]
    timestamp: Annotated[
        datetime,
        Field(
            default_factory=lambda: datetime.now(UTC),
            description=SpecDoc.MEMORY_TIMESTAMP,
        ),
    ]
    role: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.MEMORY_ROLE,
        ),
    ]
    metadata: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description=SpecDoc.MEMORY_METADATA,
        ),
    ]
    episode_type: Annotated[
        EpisodeType | None,
        Field(
            default=None,
            description=SpecDoc.MEMORY_EPISODIC_TYPE,
        ),
    ]

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: TimestampInput) -> datetime:
        if v is None:
            return datetime.now(UTC)

        # Already a datetime
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=UTC)

        # Unix timestamp (seconds or milliseconds)
        if isinstance(v, (int, float)):
            # Heuristic: > 10^12 is probably milliseconds
            if v > 1_000_000_000_000:
                v = v / 1000
            return datetime.fromtimestamp(v, tz=UTC)

        # String date
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                pass

        raise InvalidTimestampError(f"Unsupported timestamp: {v}")


class AddMemoriesSpec(_WithOrgAndProj):
    """Specification model for adding memories."""

    types: Annotated[
        list[MemoryType],
        Field(
            default_factory=list,
            description=SpecDoc.MEMORY_TYPES,
            examples=Examples.MEMORY_TYPES,
        ),
    ]

    messages: Annotated[
        list[MemoryMessage],
        Field(
            min_length=1,
            description=SpecDoc.MEMORY_MESSAGES,
        ),
    ]


class AddMemoryResult(BaseModel):
    """Response model for adding memories."""

    uid: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.MEMORY_UID,
        ),
    ]


class AddMemoriesResponse(BaseModel):
    """Response model for adding memories."""

    results: Annotated[
        list[AddMemoryResult],
        Field(
            ...,
            description=SpecDoc.ADD_MEMORY_RESULTS,
        ),
    ]


class SearchMemoriesSpec(_WithOrgAndProj):
    """Specification model for searching memories."""

    top_k: Annotated[
        int,
        Field(
            default=10,
            description=SpecDoc.TOP_K,
            examples=Examples.TOP_K,
        ),
    ]
    query: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.QUERY,
            examples=Examples.QUERY,
        ),
    ]
    filter: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.FILTER_MEM,
            examples=Examples.FILTER_MEM,
        ),
    ]
    score_threshold: Annotated[
        float | None,
        Field(
            default=None,
            description=SpecDoc.SCORE_THRESHOLD,
            examples=Examples.SCORE_THRESHOLD,
        ),
    ]
    types: Annotated[
        list[MemoryType],
        Field(
            default_factory=list,
            description=SpecDoc.MEMORY_TYPES,
            examples=Examples.MEMORY_TYPES,
        ),
    ]


class ListMemoriesSpec(_WithOrgAndProj):
    """Specification model for listing memories."""

    page_size: Annotated[
        int,
        Field(
            default=100,
            description=SpecDoc.PAGE_SIZE,
            examples=Examples.PAGE_SIZE,
        ),
    ]
    page_num: Annotated[
        int,
        Field(
            default=0,
            description=SpecDoc.PAGE_NUM,
            examples=Examples.PAGE_NUM,
        ),
    ]
    filter: Annotated[
        str,
        Field(
            default="",
            description=SpecDoc.FILTER_MEM,
            examples=Examples.FILTER_MEM,
        ),
    ]
    type: Annotated[
        MemoryType | None,
        Field(
            default=None,
            description=SpecDoc.MEMORY_TYPE_SINGLE,
            examples=Examples.MEMORY_TYPE_SINGLE,
        ),
    ]


class DeleteEpisodicMemorySpec(_WithOrgAndProj):
    """Specification model for deleting episodic memories."""

    episodic_id: Annotated[
        SafeId,
        Field(
            default="",
            description=SpecDoc.EPISODIC_ID,
            examples=Examples.EPISODIC_ID,
        ),
    ]
    episodic_ids: Annotated[
        list[SafeId],
        Field(
            default=[],
            description=SpecDoc.EPISODIC_IDS,
            examples=Examples.EPISODIC_IDS,
        ),
    ]

    def get_ids(self) -> list[str]:
        """Get a list of episodic IDs to delete."""
        id_set = set(self.episodic_ids)
        if len(self.episodic_id) > 0:
            id_set.add(self.episodic_id)
        id_set = {i.strip() for i in id_set if i and i.strip()}
        return sorted(id_set)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Ensure at least one ID is provided."""
        if len(self.get_ids()) == 0:
            raise ValueError("At least one episodic ID must be provided")
        return self


# ---


class DeleteSemanticMemorySpec(_WithOrgAndProj):
    """Specification model for deleting semantic memories."""

    semantic_id: Annotated[
        SafeId,
        Field(
            default="",
            description=SpecDoc.SEMANTIC_ID,
            examples=Examples.SEMANTIC_ID,
        ),
    ]
    semantic_ids: Annotated[
        list[SafeId],
        Field(
            default=[],
            description=SpecDoc.SEMANTIC_IDS,
            examples=Examples.SEMANTIC_IDS,
        ),
    ]

    def get_ids(self) -> list[str]:
        """Get a list of semantic IDs to delete."""
        id_set = set(self.semantic_ids)
        if len(self.semantic_id) > 0:
            id_set.add(self.semantic_id)
        id_set = {i.strip() for i in id_set if len(i.strip()) > 0}
        return sorted(id_set)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Ensure at least one ID is provided."""
        if len(self.get_ids()) == 0:
            raise ValueError("At least one semantic ID must be provided")
        return self


class SearchResult(BaseModel):
    """Response model for memory search results."""

    status: Annotated[
        int,
        Field(
            default=0,
            description=SpecDoc.STATUS,
            examples=Examples.SEARCH_RESULT_STATUS,
        ),
    ]
    content: Annotated[
        "SearchResultContent",
        Field(
            ...,
            description=SpecDoc.CONTENT,
        ),
    ]


class EpisodicSearchShortTermMemory(BaseModel):
    """Short-term episodic memory search results."""

    episodes: list[EpisodeResponse]
    episode_summary: list[str]


class EpisodicSearchLongTermMemory(BaseModel):
    """Long-term episodic memory search results."""

    episodes: list[EpisodeResponse]


class EpisodicSearchResult(BaseModel):
    """Episodic payload returned by `/memories/search`."""

    long_term_memory: EpisodicSearchLongTermMemory
    short_term_memory: EpisodicSearchShortTermMemory


class SearchResultContent(BaseModel):
    """Payload for SearchResult.content returned by `/memories/search`."""

    model_config = ConfigDict(extra="forbid")

    episodic_memory: EpisodicSearchResult | None = None
    semantic_memory: list[SemanticFeature] | None = None


class ListResult(BaseModel):
    """Response model for memory list results."""

    status: Annotated[
        int,
        Field(
            default=0,
            description=SpecDoc.STATUS,
            examples=Examples.SEARCH_RESULT_STATUS,
        ),
    ]
    content: Annotated[
        "ListResultContent",
        Field(
            ...,
            description=SpecDoc.CONTENT,
        ),
    ]


class ListResultContent(BaseModel):
    """Payload for ListResult.content returned by `/memories/list`."""

    model_config = ConfigDict(extra="forbid")

    episodic_memory: list[Episode] | None = None
    semantic_memory: list[SemanticFeature] | None = None


class RestErrorModel(BaseModel):
    """Model representing an error response."""

    code: Annotated[
        int,
        Field(
            ...,
            description=SpecDoc.ERROR_CODE,
            examples=[422, 404],
        ),
    ]
    message: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.ERROR_MESSAGE,
        ),
    ]
    internal_error: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.ERROR_INTERNAL,
        ),
    ]
    exception: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.ERROR_EXCEPTION,
        ),
    ]
    trace: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.ERROR_TRACE,
        ),
    ]
