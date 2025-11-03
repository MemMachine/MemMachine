import string
from typing import Annotated

from pydantic import BaseModel, Field, InstanceOf, field_validator, model_validator

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.reranker import Reranker
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.session_manager_interface import SessionDataManager


class ShortTermMemoryParams(BaseModel):
    """
    Parameters for configuring the short-term memory.
    Attriutes:
        session_key (str): The unique identifier for the session.
        llm_model (LanguageModel): The language model to use for summarization.
        data_manager (SessionDataManager): The session data manager.
        summary_prompt_system (str): The system prompt for the summarization.
        summary_prompt_user (str): The user prompt for the summarization.
        message_capacity (int): The maximum number of messages to summarize.
        enabled (bool): Whether the short-term memory is enabled.
    """

    session_key: Annotated[str, Field(..., min_length=1)]
    llm_model: InstanceOf[LanguageModel] = Field(..., description="The language model to use for summarization")
    data_manager: InstanceOf[SessionDataManager] | None = Field(default=None, description="The session data manager")
    summary_prompt_system: Annotated[str, Field(..., min_length=1, description="The system prompt for the summarization")]
    summary_prompt_user: Annotated[str, Field(..., min_length=1, description="The user prompt for the summarization")]
    message_capacity: Annotated[int, Field(default=64000, gt=0, description="The maximum length of short-term memory")]
    enabled: bool = True

    @field_validator("summary_prompt_user")
    def validate_sumary_user_prompt(cls, v):
        fields = [fname for _, fname, _, _ in string.Formatter().parse(v) if fname]
        if len(fields) != 3:
            raise ValueError(f"Expect 3 fields in {v}")
        if "episodes" not in fields:
            raise ValueError(f"Expect 'episodes' in {v}")
        if "summary" not in fields:
            raise ValueError(f"Expect 'summary' in {v}")
        if "max_length" not in fields:
            raise ValueError(f"Expect 'max_length' in {v}")
        return v

class LongTermMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_id (str):
            Session identifier.
        max_chunk_length (int):
            Maximum length of a chunk in characters
            (default: 1000).
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
    """

    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    max_chunk_length: int = Field(
        1000,
        description="Maximum length of a chunk in characters.",
        gt=0,
    )
    vector_graph_store: InstanceOf[VectorGraphStore] = Field(
        ...,
        description="VectorGraphStore instance for storing and retrieving memories",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    enabled: bool = True


class EpisodicMemoryParams(BaseModel):
    """
    Parameters for configuring the EpisodicMemory.
    Attributes:
        session_key (str): The unique identifier for the session.
        metrics_factory (MetricsFactory): The metrics factory.
        short_term_memory (ShortTermMemoryParams): The short-term memory parameters.
        long_term_memory (LongTermMemoryParams): The long-term memory parameters.
        enabled (bool): Whether the episodic memory is enabled.
    """
    session_key: Annotated[str, Field(..., min_length=1, description="The unique identifier for the session")]
    metrics_factory: InstanceOf[MetricsFactory] = Field(..., description="The metrics factory")
    short_term_memory: ShortTermMemoryParams | None = Field(default=None, description="The short-term memory parameters")
    long_term_memory: LongTermMemoryParams | None = Field(default=None, description="The long-term memory parameters")
    enabled: bool = True

    @model_validator(mode="after")
    def validate_memory_params(self):
        if self.enabled is False:
            return self
        if self.short_term_memory is None and self.long_term_memory is None:
            raise ValueError(
                "At least one of short_term_memory or long_term_memory must be provided."
            )
        return self


class EpisodicMemoryManagerParam(BaseModel):
    """
    Parameters for configuring the EpisodicMemoryManager.
    Attributes:
        instance_cache_size (int): The maximum number of instances to cache.
        max_life_time (int): The maximum idle lifetime of an instance in seconds.
        session_storage (SessionDataManager): The session storage.
    """
    instance_cache_size: Annotated[int, Field(default=100, gt=0, description="The maximum number of instances to cache")]
    max_life_time: Annotated[int, Field(default=600, gt=0, description="The maximum idle lifetime of an instance in seconds")]
    session_storage: InstanceOf[SessionDataManager] = Field(..., description="Session storage")
