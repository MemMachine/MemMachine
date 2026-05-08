"""Episodic memory configuration and merge utilities."""

from typing import Annotated, Any, Literal, Self, cast

from pydantic import BaseModel, Discriminator, Field, Tag

from memmachine_server.common.configuration.mixin_confs import (
    MetricsFactoryIdMixin,
    YamlSerializableMixin,
)


def _long_term_memory_backend_discriminator(value: object) -> str:
    """
    Resolve the long-term-memory backend tag for the discriminated union.

    Parse-time default: a missing/None `backend` means the writer predates the
    discriminator, so deserialize as `"declarative"` (the legacy backend).
    Code that *creates* new configs is responsible for explicitly setting
    `backend="event"` if it wants the new default.
    """
    if isinstance(value, dict):
        backend = cast(dict[str, Any], value).get("backend")
    else:
        backend = getattr(value, "backend", None)
    return backend if isinstance(backend, str) else "declarative"


def merge_partial_configs[TFull: BaseModel, TPartial: BaseModel](
    primary: TPartial,
    fallback: TPartial,
    full_cls: type[TFull],
) -> TFull:
    """
    Merge partial Pydantic configs into a full configuration.

    - `primary` overrides `fallback`
    - Missing required fields (after merge) raise ValueError
    - Returns an instance of `full_cls`
    """
    data = {}

    for field in full_cls.model_fields:
        v1 = getattr(primary, field, None)
        v2 = getattr(fallback, field, None)

        if v1 is not None:
            data[field] = v1
        elif v2 is not None:
            data[field] = v2

    return full_cls(**data)


class ShortTermMemoryConf(BaseModel):
    """Configuration for short-term memory behavior."""

    session_key: str = Field(..., description="Session identifier", min_length=1)
    llm_model: str = Field(
        ...,
        description="ID of the language model to use for summarization",
    )
    summary_prompt_system: str = Field(
        ...,
        min_length=1,
        description="The system prompt for the summarization",
    )
    summary_prompt_user: str = Field(
        ...,
        min_length=1,
        description="The user prompt for the summarization",
    )
    message_capacity: int = Field(
        default=64000,
        gt=0,
        description="The maximum length of short-term memory",
    )


class ShortTermMemoryConfPartial(BaseModel):
    """Partial configuration for short-term memory."""

    session_key: str | None = Field(
        default=None,
        description="Session identifier",
        min_length=1,
    )
    llm_model: str | None = Field(
        default=None,
        description="ID of the language model to use for summarization",
    )
    summary_prompt_system: str | None = Field(
        default=None,
        min_length=1,
        description="The system prompt for the summarization",
    )
    summary_prompt_user: str | None = Field(
        default=None,
        min_length=1,
        description="The user prompt for the summarization",
    )
    message_capacity: int | None = Field(
        default=None,
        gt=0,
        description="The maximum length of short-term memory",
    )

    def merge(self, other: Self) -> ShortTermMemoryConf:
        """Merge with another partial into a complete short-term config."""
        return merge_partial_configs(self, other, ShortTermMemoryConf)


# Segmenter / Deriver sub-configurations for the event-backed long-term memory.


class PassthroughSegmenterConf(BaseModel):
    """One segment per block; no splitting."""

    type: Literal["passthrough"] = "passthrough"


class TextSegmenterConf(BaseModel):
    """Recursive-character text segmenter."""

    type: Literal["text"] = "text"
    max_chunk_length: int = Field(
        500,
        description="Max code-point length for text chunks",
    )


SegmenterConf = Annotated[
    PassthroughSegmenterConf | TextSegmenterConf,
    Field(discriminator="type"),
]


class WholeTextDeriverConf(BaseModel):
    """Whole-text deriver: one derivative per segment."""

    type: Literal["whole_text"] = "whole_text"


class SentenceTextDeriverConf(BaseModel):
    """Per-sentence text deriver."""

    type: Literal["sentence_text"] = "sentence_text"


DeriverConf = Annotated[
    WholeTextDeriverConf | SentenceTextDeriverConf,
    Field(discriminator="type"),
]


class DeclarativeLongTermMemoryConf(BaseModel):
    """Declarative-backend long-term memory (VectorGraphStore)."""

    backend: Literal["declarative"] = "declarative"
    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    vector_graph_store: str = Field(
        ...,
        description="ID of the VectorGraphStore instance for storing and retrieving memories",
    )
    embedder: str = Field(
        ...,
        description="ID of the Embedder instance for creating embeddings",
    )
    reranker: str = Field(
        ...,
        description="ID of the Reranker instance for reranking search results",
    )
    message_sentence_chunking: bool = Field(
        False,
        description="Whether to chunk message episodes into sentences for embedding",
    )


class EventLongTermMemoryConf(BaseModel):
    """Event-backend long-term memory (VectorStore + SegmentStore)."""

    backend: Literal["event"] = "event"
    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    vector_store: str = Field(
        ...,
        description="ID of the VectorStore instance backing the derivative index",
    )
    segment_store: str = Field(
        ...,
        description=(
            "ID of the SQL engine resource backing the segment store. "
            "The SegmentStore is constructed implicitly from the engine."
        ),
    )
    embedder: str = Field(
        ...,
        description="ID of the Embedder instance for creating embeddings",
    )
    reranker: str | None = Field(
        default=None,
        description=(
            "ID of the Reranker instance. If None, embedding similarity scores "
            "are used for ordering."
        ),
    )
    properties_schema: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "User-defined filterable properties and their type names "
            '(e.g. {"my_field": "str"}). Type names: bool, int, float, str, datetime.'
        ),
    )
    segmenter: SegmenterConf = Field(
        default_factory=PassthroughSegmenterConf,
        description="Segmenter sub-configuration (default: passthrough)",
    )
    deriver: DeriverConf = Field(
        default_factory=WholeTextDeriverConf,
        description="Deriver sub-configuration (default: whole_text)",
    )


LongTermMemoryConf = Annotated[
    Annotated[DeclarativeLongTermMemoryConf, Tag("declarative")]
    | Annotated[EventLongTermMemoryConf, Tag("event")],
    Discriminator(_long_term_memory_backend_discriminator),
]


class LongTermMemoryConfPartial(BaseModel):
    """
    Partial configuration for long-term memory.

    A flat partial that can describe either backend. `merge()` resolves the
    discriminator (None -> declarative for backwards compat) and produces the
    appropriate full conf variant.
    """

    backend: Literal["declarative", "event"] | None = Field(
        default=None,
        description=(
            "Long-term memory backend. None or 'declarative' uses the legacy "
            "VectorGraphStore-backed declarative memory. 'event' uses the "
            "VectorStore + SegmentStore event memory."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier",
    )

    # Declarative-backend fields.
    vector_graph_store: str | None = Field(
        default=None,
        description="ID of the VectorGraphStore (declarative backend only)",
    )
    message_sentence_chunking: bool | None = Field(
        default=None,
        description="Sentence-chunk message episodes (declarative backend only)",
    )

    # Event-backend fields.
    vector_store: str | None = Field(
        default=None,
        description="ID of the VectorStore (event backend only)",
    )
    segment_store: str | None = Field(
        default=None,
        description="ID of the SQL engine resource for the segment store (event backend only)",
    )
    properties_schema: dict[str, str] | None = Field(
        default=None,
        description="User-defined filterable properties (event backend only)",
    )
    segmenter: SegmenterConf | None = Field(
        default=None,
        description="Segmenter sub-configuration (event backend only)",
    )
    deriver: DeriverConf | None = Field(
        default=None,
        description="Deriver sub-configuration (event backend only)",
    )

    # Shared fields.
    embedder: str | None = Field(
        default=None,
        description="ID of the Embedder instance for creating embeddings",
    )
    reranker: str | None = Field(
        default=None,
        description="ID of the Reranker instance for reranking search results",
    )

    def merge(self, other: Self) -> LongTermMemoryConf:
        """Merge with another partial into a complete long-term config.

        Resolution rule for the backend discriminator:
        - if either side sets `backend` explicitly, that value wins (primary first).
        - if neither side sets `backend`, default to `declarative` (the legacy
          shape, for backwards compatibility with pre-discriminator configs).
          Callers that want event-memory should set `backend="event"`
          explicitly at creation time (e.g. wizard, project-creation API).
        """
        backend = self.backend if self.backend is not None else other.backend
        if backend is None:
            backend = "declarative"

        if backend == "declarative":
            return merge_partial_configs(self, other, DeclarativeLongTermMemoryConf)

        # Event backend: synthesize defaults for sub-configs that the flat partial
        # leaves None.
        merged = merge_partial_configs(
            _force_backend(self, "event"),
            _force_backend(other, "event"),
            EventLongTermMemoryConf,
        )
        return merged


def _force_backend(
    partial: LongTermMemoryConfPartial,
    backend: Literal["declarative", "event"],
) -> LongTermMemoryConfPartial:
    """Return a copy of the partial with `backend` explicitly set."""
    return partial.model_copy(update={"backend": backend})


class EpisodicMemoryConf(MetricsFactoryIdMixin, YamlSerializableMixin):
    """Configuration for episodic memory service."""

    session_key: str = Field(
        ...,
        min_length=1,
        description="The unique identifier for the session",
    )
    metrics_factory_id: str = Field(
        default="prometheus",
        description="ID of the metrics factory",
    )
    long_term_memory: LongTermMemoryConf | None = Field(
        default=None,
        description="The long-term memory configuration",
    )
    short_term_memory: ShortTermMemoryConf | None = Field(
        default=None,
        description="The short-term memory configuration",
    )
    long_term_memory_enabled: bool = Field(
        default=True,
        description="Whether the long-term memory is enabled",
    )
    short_term_memory_enabled: bool = Field(
        default=True,
        description="Whether the short-term memory is enabled",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the episodic memory is enabled",
    )


class EpisodicMemoryConfPartial(YamlSerializableMixin):
    """Partial configuration for episodic memory with nested sections."""

    session_key: str | None = Field(
        default=None,
        min_length=1,
        description="The unique identifier for the session",
    )
    metrics_factory_id: str | None = Field(
        default=None,
        description="ID of the metrics factory",
    )
    long_term_memory: LongTermMemoryConfPartial | None = Field(
        default=None,
        description="Partial configuration for long-term memory in episodic memory",
    )
    short_term_memory: ShortTermMemoryConfPartial | None = Field(
        default=None,
        description="Partial configuration for session memory in episodic memory",
    )
    long_term_memory_enabled: bool | None = Field(
        default=None,
        description="Whether the long-term memory is enabled",
    )
    short_term_memory_enabled: bool | None = Field(
        default=None,
        description="Whether the short-term memory is enabled",
    )
    enabled: bool | None = Field(
        default=True,
        description="Whether the episodic memory is enabled",
    )

    def merge(self, other: Self) -> EpisodicMemoryConf:
        """Merge scalar fields, then merge nested configuration blocks."""
        # ---- Step 1: merge scalar fields (this ignores nested configs) ----
        merged = merge_partial_configs(self, other, EpisodicMemoryConfPartial)

        # ---- Step 2: normalize partial nested configs ----
        # Convert None -> empty partial so merge() always works
        stm_self = self.short_term_memory or ShortTermMemoryConfPartial()
        stm_other = other.short_term_memory or ShortTermMemoryConfPartial()

        ltm_self = self.long_term_memory or LongTermMemoryConfPartial()
        ltm_other = other.long_term_memory or LongTermMemoryConfPartial()

        # ---- Step 3: perform merges using each component's own merge() method ----
        session_key = merged.session_key
        if session_key is None:
            raise ValueError("EpisodicMemoryConfPartial.merge() requires session_key")

        stm_self.session_key = session_key
        ltm_self.session_id = session_key
        stm_merged = stm_self.merge(stm_other)
        ltm_merged = ltm_self.merge(ltm_other)

        # ---- Step 4: update nested configuration in the base result ----
        return EpisodicMemoryConf(
            session_key=session_key,
            metrics_factory_id=merged.metrics_factory_id
            if merged.metrics_factory_id is not None
            else "prometheus",
            short_term_memory=stm_merged,
            long_term_memory=ltm_merged,
            long_term_memory_enabled=True
            if merged.long_term_memory_enabled is None and ltm_merged is not None
            else merged.long_term_memory_enabled,
            short_term_memory_enabled=True
            if merged.short_term_memory_enabled is None and stm_merged is not None
            else merged.short_term_memory_enabled,
            enabled=True if merged.enabled is None else merged.enabled,
        )
