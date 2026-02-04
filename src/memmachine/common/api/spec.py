"""API v2 specification models for request and response structures."""

import logging
from datetime import datetime, timezone
from enum import Enum

# Python 3.11+ has UTC as a constant, Python 3.10 uses timezone.utc
try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

from typing import Annotated, Any

# Python 3.11+ has Self in typing, Python 3.10 uses typing_extensions
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

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

from memmachine.common.data_types import PropertyValue

from . import EpisodeType, MemoryType
from .doc import Examples, SpecDoc

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

    content: Annotated[
        str,
        Field(..., description=SpecDoc.EPISODE_CONTENT),
    ]
    producer_id: Annotated[
        str,
        Field(..., description=SpecDoc.EPISODE_PRODUCER_ID),
    ]
    producer_role: Annotated[
        str,
        Field(..., description=SpecDoc.EPISODE_PRODUCER_ROLE),
    ]
    produced_for_id: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.EPISODE_PRODUCED_FOR_ID),
    ]
    episode_type: Annotated[
        EpisodeType | None,
        Field(default=None, description=SpecDoc.EPISODE_TYPE),
    ]
    metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(default=None, description=SpecDoc.EPISODE_METADATA),
    ]
    created_at: Annotated[
        AwareDatetime | None,
        Field(default=None, description=SpecDoc.EPISODE_CREATED_AT),
    ]


class EpisodeResponse(EpisodeEntry):
    """Episode data returned in search responses."""

    uid: Annotated[
        EpisodeIdT,
        Field(..., description=SpecDoc.EPISODE_UID),
    ]
    score: Annotated[
        float | None,
        Field(default=None, description=SpecDoc.EPISODE_SCORE),
    ]


class Episode(BaseModel):
    """Episode data returned in list responses."""

    uid: Annotated[
        EpisodeIdT,
        Field(..., description=SpecDoc.EPISODE_UID),
    ]
    content: Annotated[
        str,
        Field(..., description=SpecDoc.EPISODE_CONTENT),
    ]
    session_key: Annotated[
        str,
        Field(..., description=SpecDoc.EPISODE_SESSION_KEY),
    ]
    created_at: Annotated[
        AwareDatetime,
        Field(..., description=SpecDoc.EPISODE_CREATED_AT),
    ]

    producer_id: Annotated[
        str,
        Field(..., description=SpecDoc.EPISODE_PRODUCER_ID),
    ]
    producer_role: Annotated[
        str,
        Field(..., description=SpecDoc.EPISODE_PRODUCER_ROLE),
    ]
    produced_for_id: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.EPISODE_PRODUCED_FOR_ID),
    ]

    sequence_num: Annotated[
        int,
        Field(default=0, description=SpecDoc.EPISODE_SEQUENCE_NUM),
    ]

    episode_type: Annotated[
        EpisodeType,
        Field(default=EpisodeType.MESSAGE, description=SpecDoc.EPISODE_TYPE),
    ]
    content_type: Annotated[
        ContentType,
        Field(default=ContentType.STRING, description=SpecDoc.EPISODE_CONTENT_TYPE),
    ]
    filterable_metadata: Annotated[
        dict[str, PropertyValue] | None,
        Field(default=None, description=SpecDoc.EPISODE_FILTERABLE_METADATA),
    ]
    metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(default=None, description=SpecDoc.EPISODE_METADATA),
    ]

    def __hash__(self) -> int:
        """Hash an episode by its UID."""
        return hash(self.uid)


SetIdT = str
FeatureIdT = str


class SemanticFeature(BaseModel):
    """Semantic memory entry returned in API responses."""

    class Metadata(BaseModel):
        """Storage metadata for a semantic feature, including id and citations."""

        citations: Annotated[
            list[EpisodeIdT] | None,
            Field(default=None, description=SpecDoc.SEMANTIC_METADATA_CITATIONS),
        ]
        id: Annotated[
            FeatureIdT | None,
            Field(default=None, description=SpecDoc.SEMANTIC_METADATA_ID),
        ]
        other: Annotated[
            dict[str, Any] | None,
            Field(default=None, description=SpecDoc.SEMANTIC_METADATA_OTHER),
        ]

    set_id: Annotated[
        SetIdT | None,
        Field(default=None, description=SpecDoc.SEMANTIC_SET_ID),
    ]
    category: Annotated[
        str,
        Field(..., description=SpecDoc.SEMANTIC_CATEGORY),
    ]
    tag: Annotated[
        str,
        Field(..., description=SpecDoc.SEMANTIC_TAG),
    ]
    feature_name: Annotated[
        str,
        Field(..., description=SpecDoc.SEMANTIC_FEATURE_NAME),
    ]
    value: Annotated[
        str,
        Field(..., description=SpecDoc.SEMANTIC_VALUE),
    ]
    metadata: Annotated[
        Metadata,
        Field(default_factory=Metadata, description=SpecDoc.SEMANTIC_METADATA),
    ]


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


# Type alias for timestamp input
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
    set_metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(
            default=None,
            description=SpecDoc.SET_METADATA,
        ),
    ]
    expand_context: Annotated[
        int,
        Field(
            default=0,
            description=SpecDoc.EXPAND_CONTEXT,
            examples=Examples.EXPAND_CONTEXT,
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
    agent_mode: Annotated[
        bool,
        Field(
            default=False,
            description=SpecDoc.AGENT_MODE,
            examples=Examples.AGENT_MODE,
        ),
    ]


class DeleteMemoriesSpec(_WithOrgAndProj):
    """Specification model for deleting memories."""

    episodic_memory_uids: Annotated[
        list[EpisodeIdT],
        Field(
            default=[],
            description=SpecDoc.EPISODIC_IDS,
            examples=Examples.EPISODIC_IDS,
        ),
    ]

    semantic_memory_uids: Annotated[
        list[FeatureIdT],
        Field(
            default=[],
            description=SpecDoc.SEMANTIC_IDS,
            examples=Examples.SEMANTIC_IDS,
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
    set_metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(
            default=None,
            description=SpecDoc.SET_METADATA,
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


class AddFeatureSpec(_WithOrgAndProj):
    """Specification model for adding a semantic feature."""

    set_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.FEATURE_SET_ID,
        ),
    ]
    category_name: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.FEATURE_CATEGORY_NAME,
        ),
    ]
    tag: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.FEATURE_TAG,
        ),
    ]
    feature: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.FEATURE_NAME,
        ),
    ]
    value: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.FEATURE_VALUE,
        ),
    ]
    feature_metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(
            default=None,
            description=SpecDoc.FEATURE_METADATA,
        ),
    ]
    citations: Annotated[
        list[EpisodeIdT] | None,
        Field(
            default=None,
            description=SpecDoc.FEATURE_CITATIONS,
        ),
    ]


class AddFeatureResponse(BaseModel):
    """Response model for adding a semantic feature."""

    feature_id: Annotated[
        FeatureIdT,
        Field(
            ...,
            description=SpecDoc.FEATURE_ID,
        ),
    ]


class GetFeatureSpec(_WithOrgAndProj):
    """Specification model for getting a semantic feature."""

    feature_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.FEATURE_ID,
        ),
    ]
    load_citations: Annotated[
        bool,
        Field(
            default=False,
            description=SpecDoc.FEATURE_LOAD_CITATIONS,
        ),
    ]


class UpdateFeatureSpec(_WithOrgAndProj):
    """Specification model for updating a semantic feature."""

    feature_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.FEATURE_ID,
        ),
    ]
    category_name: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.FEATURE_CATEGORY_NAME,
        ),
    ]
    tag: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.FEATURE_TAG,
        ),
    ]
    feature: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.FEATURE_NAME,
        ),
    ]
    value: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.FEATURE_VALUE,
        ),
    ]
    metadata: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=SpecDoc.FEATURE_METADATA,
        ),
    ]


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

    episodes: Annotated[
        list[EpisodeResponse],
        Field(..., description=SpecDoc.EPISODIC_SHORT_EPISODES),
    ]
    episode_summary: Annotated[
        list[str],
        Field(..., description=SpecDoc.EPISODIC_SHORT_SUMMARY),
    ]


class EpisodicSearchLongTermMemory(BaseModel):
    """Long-term episodic memory search results."""

    episodes: Annotated[
        list[EpisodeResponse],
        Field(..., description=SpecDoc.EPISODIC_LONG_EPISODES),
    ]


class EpisodicSearchResult(BaseModel):
    """Episodic payload returned by `/memories/search`."""

    long_term_memory: Annotated[
        EpisodicSearchLongTermMemory,
        Field(..., description=SpecDoc.EPISODIC_LONG_TERM),
    ]
    short_term_memory: Annotated[
        EpisodicSearchShortTermMemory,
        Field(..., description=SpecDoc.EPISODIC_SHORT_TERM),
    ]


class SearchResultContent(BaseModel):
    """Payload for SearchResult.content returned by `/memories/search`."""

    model_config = ConfigDict(extra="forbid")

    episodic_memory: Annotated[
        EpisodicSearchResult | None,
        Field(default=None, description=SpecDoc.SEARCH_EPISODIC_MEMORY),
    ]
    semantic_memory: Annotated[
        list[SemanticFeature] | None,
        Field(default=None, description=SpecDoc.SEARCH_SEMANTIC_MEMORY),
    ]


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

    episodic_memory: Annotated[
        list[Episode] | None,
        Field(default=None, description=SpecDoc.LIST_EPISODIC_MEMORY),
    ]
    semantic_memory: Annotated[
        list[SemanticFeature] | None,
        Field(default=None, description=SpecDoc.LIST_SEMANTIC_MEMORY),
    ]


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


# --- Semantic Set Type API Models ---


class CreateSemanticSetTypeSpec(_WithOrgAndProj):
    """Specification model for creating a semantic set type."""

    is_org_level: Annotated[
        bool,
        Field(
            default=False,
            description=SpecDoc.SET_TYPE_IS_ORG_LEVEL,
        ),
    ]
    metadata_tags: Annotated[
        list[str],
        Field(
            ...,
            description=SpecDoc.SET_TYPE_METADATA_TAGS,
        ),
    ]
    name: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_TYPE_NAME,
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_TYPE_DESCRIPTION,
        ),
    ]


class CreateSemanticSetTypeResponse(BaseModel):
    """Response model for creating a semantic set type."""

    set_type_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SET_TYPE_ID,
        ),
    ]


class ListSemanticSetTypesSpec(_WithOrgAndProj):
    """Specification model for listing semantic set types."""


class SemanticSetTypeEntry(BaseModel):
    """A semantic set type entry."""

    id: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_TYPE_ID,
        ),
    ]
    is_org_level: Annotated[
        bool,
        Field(
            ...,
            description=SpecDoc.SET_TYPE_IS_ORG_LEVEL,
        ),
    ]
    tags: Annotated[
        list[str],
        Field(
            ...,
            description=SpecDoc.SET_TYPE_METADATA_TAGS,
        ),
    ]
    name: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_TYPE_NAME,
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_TYPE_DESCRIPTION,
        ),
    ]


class ListSemanticSetTypesResponse(BaseModel):
    """Response model for listing semantic set types."""

    set_types: Annotated[
        list[SemanticSetTypeEntry],
        Field(
            ...,
            description=SpecDoc.SET_TYPES_LIST,
        ),
    ]


class DeleteSemanticSetTypeSpec(_WithOrgAndProj):
    """Specification model for deleting a semantic set type."""

    set_type_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SET_TYPE_ID,
        ),
    ]


class GetSemanticSetIdSpec(_WithOrgAndProj):
    """Specification model for getting a semantic set ID."""

    is_org_level: Annotated[
        bool,
        Field(
            default=False,
            description=SpecDoc.SET_TYPE_IS_ORG_LEVEL,
        ),
    ]
    metadata_tags: Annotated[
        list[str],
        Field(
            ...,
            description=SpecDoc.SET_TYPE_METADATA_TAGS,
        ),
    ]
    set_metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(
            default=None,
            description=SpecDoc.SET_METADATA,
        ),
    ]


class GetSemanticSetIdResponse(BaseModel):
    """Response model for getting a semantic set ID."""

    set_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SEMANTIC_SET_ID,
        ),
    ]


class ListSemanticSetIdsSpec(_WithOrgAndProj):
    """Specification model for listing semantic set IDs."""

    set_metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(
            default=None,
            description=SpecDoc.SET_METADATA,
        ),
    ]


class SemanticSetEntry(BaseModel):
    """A semantic set entry."""

    id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SEMANTIC_SET_ID,
        ),
    ]
    is_org_level: Annotated[
        bool,
        Field(
            ...,
            description=SpecDoc.SET_TYPE_IS_ORG_LEVEL,
        ),
    ]
    tags: Annotated[
        list[str],
        Field(
            ...,
            description=SpecDoc.SET_TYPE_METADATA_TAGS,
        ),
    ]
    name: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_TYPE_NAME,
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_TYPE_DESCRIPTION,
        ),
    ]


class ListSemanticSetIdsResponse(BaseModel):
    """Response model for listing semantic set IDs."""

    sets: Annotated[
        list[SemanticSetEntry],
        Field(
            ...,
            description=SpecDoc.SETS_LIST,
        ),
    ]


class ConfigureSemanticSetSpec(_WithOrgAndProj):
    """Specification model for configuring a semantic set."""

    set_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SEMANTIC_SET_ID,
        ),
    ]
    embedder_name: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_EMBEDDER_NAME,
        ),
    ]
    llm_name: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.SET_LLM_NAME,
        ),
    ]


# --- Semantic Category API Models ---


class GetSemanticCategorySpec(_WithOrgAndProj):
    """Specification model for getting a semantic category."""

    category_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_ID,
        ),
    ]


class SemanticCategoryEntry(BaseModel):
    """A semantic category entry."""

    id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_ID,
        ),
    ]
    name: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_NAME,
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_PROMPT,
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.CATEGORY_DESCRIPTION,
        ),
    ]


class AddSemanticCategorySpec(_WithOrgAndProj):
    """Specification model for adding a semantic category."""

    set_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SEMANTIC_SET_ID,
        ),
    ]
    category_name: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_NAME,
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_PROMPT,
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.CATEGORY_DESCRIPTION,
        ),
    ]


class AddSemanticCategoryResponse(BaseModel):
    """Response model for adding a semantic category."""

    category_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_ID,
        ),
    ]


class AddSemanticCategoryTemplateSpec(_WithOrgAndProj):
    """Specification model for adding a semantic category template."""

    set_type_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SET_TYPE_ID,
        ),
    ]
    category_name: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_NAME,
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_PROMPT,
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.CATEGORY_DESCRIPTION,
        ),
    ]


class ListSemanticCategoryTemplatesSpec(_WithOrgAndProj):
    """Specification model for listing semantic category templates."""

    set_type_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SET_TYPE_ID,
        ),
    ]


class SemanticCategoryTemplateEntry(BaseModel):
    """A semantic category template entry."""

    id: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.CATEGORY_ID,
        ),
    ]
    name: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_NAME,
        ),
    ]
    origin_type: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.CATEGORY_ORIGIN_TYPE,
        ),
    ]
    origin_id: Annotated[
        str | None,
        Field(
            default=None,
            description=SpecDoc.CATEGORY_ORIGIN_ID,
        ),
    ]
    inherited: Annotated[
        bool | None,
        Field(
            default=None,
            description=SpecDoc.CATEGORY_INHERITED,
        ),
    ]


class ListSemanticCategoryTemplatesResponse(BaseModel):
    """Response model for listing semantic category templates."""

    categories: Annotated[
        list[SemanticCategoryTemplateEntry],
        Field(
            ...,
            description=SpecDoc.CATEGORIES_LIST,
        ),
    ]


class DisableSemanticCategorySpec(_WithOrgAndProj):
    """Specification model for disabling a semantic category."""

    set_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SEMANTIC_SET_ID,
        ),
    ]
    category_name: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_NAME,
        ),
    ]


class GetSemanticCategorySetIdsSpec(_WithOrgAndProj):
    """Specification model for getting set IDs for a category."""

    category_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_ID,
        ),
    ]


class GetSemanticCategorySetIdsResponse(BaseModel):
    """Response model for getting set IDs for a category."""

    set_ids: Annotated[
        list[str],
        Field(
            ...,
            description=SpecDoc.CATEGORY_SET_IDS,
        ),
    ]


class DeleteSemanticCategorySpec(_WithOrgAndProj):
    """Specification model for deleting a semantic category."""

    category_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_ID,
        ),
    ]


# --- Semantic Tag API Models ---


class AddSemanticTagSpec(_WithOrgAndProj):
    """Specification model for adding a tag to a category."""

    category_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CATEGORY_ID,
        ),
    ]
    tag_name: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.TAG_NAME,
        ),
    ]
    tag_description: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.TAG_DESCRIPTION,
        ),
    ]


class AddSemanticTagResponse(BaseModel):
    """Response model for adding a tag."""

    tag_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.TAG_ID,
        ),
    ]


class DeleteSemanticTagSpec(_WithOrgAndProj):
    """Specification model for deleting a tag."""

    tag_id: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.TAG_ID,
        ),
    ]


# --- Episodic Memory Configuration Models ---


class GetEpisodicMemoryConfigSpec(_WithOrgAndProj):
    """Specification model for getting episodic memory configuration."""


class EpisodicMemoryConfigEntry(BaseModel):
    """Response model for episodic memory configuration."""

    enabled: Annotated[
        bool,
        Field(
            ...,
            description=SpecDoc.EPISODIC_ENABLED,
        ),
    ]
    long_term_memory_enabled: Annotated[
        bool,
        Field(
            ...,
            description=SpecDoc.EPISODIC_LTM_ENABLED,
        ),
    ]
    short_term_memory_enabled: Annotated[
        bool,
        Field(
            ...,
            description=SpecDoc.EPISODIC_STM_ENABLED,
        ),
    ]


class ConfigureEpisodicMemorySpec(_WithOrgAndProj):
    """Specification model for configuring episodic memory for a session."""

    enabled: Annotated[
        bool | None,
        Field(
            default=None,
            description=SpecDoc.EPISODIC_ENABLED,
        ),
    ]
    long_term_memory_enabled: Annotated[
        bool | None,
        Field(
            default=None,
            description=SpecDoc.EPISODIC_LTM_ENABLED,
        ),
    ]
    short_term_memory_enabled: Annotated[
        bool | None,
        Field(
            default=None,
            description=SpecDoc.EPISODIC_STM_ENABLED,
        ),
    ]


class GetShortTermMemoryConfigSpec(_WithOrgAndProj):
    """Specification model for getting short-term memory configuration."""


class ShortTermMemoryConfigEntry(BaseModel):
    """Response model for short-term memory configuration."""

    enabled: Annotated[
        bool,
        Field(
            ...,
            description=SpecDoc.EPISODIC_STM_ENABLED,
        ),
    ]


class ConfigureShortTermMemorySpec(_WithOrgAndProj):
    """Specification model for configuring short-term memory for a session."""

    enabled: Annotated[
        bool | None,
        Field(
            default=None,
            description=SpecDoc.EPISODIC_STM_ENABLED,
        ),
    ]


class GetLongTermMemoryConfigSpec(_WithOrgAndProj):
    """Specification model for getting long-term memory configuration."""


class LongTermMemoryConfigEntry(BaseModel):
    """Response model for long-term memory configuration."""

    enabled: Annotated[
        bool,
        Field(
            ...,
            description=SpecDoc.EPISODIC_LTM_ENABLED,
        ),
    ]


class ConfigureLongTermMemorySpec(_WithOrgAndProj):
    """Specification model for configuring long-term memory for a session."""

    enabled: Annotated[
        bool | None,
        Field(
            default=None,
            description=SpecDoc.EPISODIC_LTM_ENABLED,
        ),
    ]


class Version(BaseModel):
    """Model representing version information."""

    server_version: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.SERVER_VERSION,
            examples=Examples.SERVER_VERSION,
        ),
    ]
    client_version: Annotated[
        str,
        Field(
            ...,
            description=SpecDoc.CLIENT_VERSION,
            examples=Examples.CLIENT_VERSION,
        ),
    ]

    def __str__(self) -> str:
        """Return the version as a string."""
        return "\n".join(
            [
                f"server: {self.server_version}",
                f"client: {self.client_version}",
            ]
        )
