"""Episodic memory configuration and merge utilities."""

from typing import Self, TypeVar

from pydantic import BaseModel, Field

from memmachine.common.configuration.mixin_confs import (
    MetricsFactoryIdMixin,
    YamlSerializableMixin,
)

TFull = TypeVar("TFull", bound=BaseModel)
TPartial = TypeVar("TPartial", bound=BaseModel)


def merge_partial_configs(
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


class LongTermMemoryConf(BaseModel):
    """Configuration for long-term memory backed by a vector store."""

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


class LongTermMemoryConfPartial(BaseModel):
    """Partial configuration for long-term memory."""

    session_id: str | None = Field(
        default=None,
        description="Session identifier",
    )
    vector_graph_store: str | None = Field(
        default=None,
        description="ID of the VectorGraphStore instance for storing and retrieving memories",
    )
    embedder: str | None = Field(
        default=None,
        description="ID of the Embedder instance for creating embeddings",
    )
    reranker: str | None = Field(
        default=None,
        description="ID of the Reranker instance for reranking search results",
    )

    def merge(self, other: Self) -> LongTermMemoryConf:
        """Merge with another partial into a complete long-term config."""
        return merge_partial_configs(self, other, LongTermMemoryConf)


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
