"""Configuration API specification models for request and response structures."""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from memmachine.common.api.doc import SpecDoc


class ResourceStatus(str, Enum):
    """Status of a resource."""

    READY = "ready"
    FAILED = "failed"
    PENDING = "pending"


class ResourceInfo(BaseModel):
    """Information about a configured resource."""

    name: Annotated[
        str,
        Field(..., description=SpecDoc.RESOURCE_NAME),
    ]
    provider: Annotated[
        str,
        Field(..., description=SpecDoc.RESOURCE_PROVIDER),
    ]
    status: Annotated[
        ResourceStatus,
        Field(..., description=SpecDoc.RESOURCE_STATUS),
    ]
    error: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.RESOURCE_ERROR_MSG),
    ]


class ResourcesStatus(BaseModel):
    """Status of all configured resources."""

    embedders: Annotated[
        list[ResourceInfo],
        Field(default_factory=list, description=SpecDoc.EMBEDDERS_STATUS),
    ]
    language_models: Annotated[
        list[ResourceInfo],
        Field(default_factory=list, description=SpecDoc.LANGUAGE_MODELS_STATUS),
    ]
    rerankers: Annotated[
        list[ResourceInfo],
        Field(default_factory=list, description=SpecDoc.RERANKERS_STATUS),
    ]
    databases: Annotated[
        list[ResourceInfo],
        Field(default_factory=list, description=SpecDoc.DATABASES_STATUS),
    ]


class LongTermMemoryConfigResponse(BaseModel):
    """Response model for long-term memory configuration."""

    embedder: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.LTM_CONFIG_EMBEDDER),
    ]
    reranker: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.LTM_CONFIG_RERANKER),
    ]
    vector_graph_store: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.LTM_CONFIG_VECTOR_GRAPH_STORE),
    ]
    enabled: Annotated[
        bool,
        Field(default=True, description=SpecDoc.LTM_CONFIG_ENABLED),
    ]


class ShortTermMemoryConfigResponse(BaseModel):
    """Response model for short-term memory configuration."""

    llm_model: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.STM_CONFIG_LLM_MODEL),
    ]
    message_capacity: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.STM_CONFIG_MESSAGE_CAPACITY),
    ]
    enabled: Annotated[
        bool,
        Field(default=True, description=SpecDoc.STM_CONFIG_ENABLED),
    ]


class EpisodicMemoryConfigResponse(BaseModel):
    """Response model for episodic memory configuration."""

    long_term_memory: Annotated[
        LongTermMemoryConfigResponse,
        Field(
            default_factory=LongTermMemoryConfigResponse, description=SpecDoc.LTM_CONFIG
        ),
    ]
    short_term_memory: Annotated[
        ShortTermMemoryConfigResponse,
        Field(
            default_factory=ShortTermMemoryConfigResponse,
            description=SpecDoc.STM_CONFIG,
        ),
    ]
    enabled: Annotated[
        bool,
        Field(default=True, description=SpecDoc.EPISODIC_ENABLED),
    ]


class SemanticMemoryConfigResponse(BaseModel):
    """Response model for semantic memory configuration."""

    enabled: Annotated[
        bool,
        Field(default=False, description=SpecDoc.SEMANTIC_ENABLED),
    ]
    database: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_DATABASE),
    ]
    llm_model: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_LLM_MODEL),
    ]
    embedding_model: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_EMBEDDING_MODEL),
    ]
    cluster_split_reranker: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_CLUSTER_SPLIT_RERANKER),
    ]
    cluster_similarity_threshold: Annotated[
        float | None,
        Field(default=None, description=SpecDoc.SEMANTIC_CLUSTER_SIMILARITY_THRESHOLD),
    ]
    cluster_max_time_gap_seconds: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.SEMANTIC_CLUSTER_MAX_TIME_GAP),
    ]
    ingestion_trigger_messages: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.SEMANTIC_INGESTION_MESSAGES),
    ]
    ingestion_trigger_age_seconds: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.SEMANTIC_INGESTION_AGE),
    ]
    cluster_idle_ttl_seconds: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.SEMANTIC_CLUSTER_IDLE_TTL),
    ]


class GetConfigResponse(BaseModel):
    """Response model for configuration retrieval."""

    resources: Annotated[
        ResourcesStatus,
        Field(..., description=SpecDoc.RESOURCES_STATUS),
    ]
    episodic_memory: Annotated[
        EpisodicMemoryConfigResponse,
        Field(
            default_factory=EpisodicMemoryConfigResponse,
            description=SpecDoc.EPISODIC_MEMORY_CONFIG,
        ),
    ]
    semantic_memory: Annotated[
        SemanticMemoryConfigResponse,
        Field(
            default_factory=SemanticMemoryConfigResponse,
            description=SpecDoc.SEMANTIC_MEMORY_CONFIG,
        ),
    ]


# --- Add Embedder Models ---


class AddOpenAIEmbedderConfig(BaseModel):
    """Configuration for adding an OpenAI embedder."""

    api_key: Annotated[
        str,
        Field(..., description=SpecDoc.API_KEY_OPENAI),
    ]
    model: Annotated[
        str,
        Field(default="text-embedding-3-small", description=SpecDoc.MODEL_NAME),
    ]
    dimensions: Annotated[
        int | None,
        Field(default=1536, description=SpecDoc.EMBEDDING_DIMENSIONS),
    ]
    base_url: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.API_BASE_URL),
    ]
    max_input_length: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.MAX_INPUT_LENGTH),
    ]
    max_retry_interval_seconds: Annotated[
        int,
        Field(default=120, description=SpecDoc.MAX_RETRY_INTERVAL),
    ]


class AddAmazonBedrockEmbedderConfig(BaseModel):
    """Configuration for adding an Amazon Bedrock embedder."""

    region: Annotated[
        str,
        Field(..., description=SpecDoc.AWS_REGION),
    ]
    model_id: Annotated[
        str,
        Field(
            default="amazon.titan-embed-text-v2:0",
            description=SpecDoc.MODEL_ID_BEDROCK,
        ),
    ]
    aws_access_key_id: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.AWS_ACCESS_KEY_ID),
    ]
    aws_secret_access_key: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.AWS_SECRET_ACCESS_KEY),
    ]
    aws_session_token: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.AWS_SESSION_TOKEN),
    ]
    max_input_length: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.MAX_INPUT_LENGTH),
    ]
    max_retry_interval_seconds: Annotated[
        int,
        Field(default=120, description=SpecDoc.MAX_RETRY_INTERVAL),
    ]


class AddSentenceTransformerEmbedderConfig(BaseModel):
    """Configuration for adding a SentenceTransformer embedder."""

    model: Annotated[
        str,
        Field(..., description=SpecDoc.SENTENCE_TRANSFORMER_MODEL),
    ]
    max_input_length: Annotated[
        int | None,
        Field(default=None, description=SpecDoc.MAX_INPUT_LENGTH),
    ]


class AddEmbedderSpec(BaseModel):
    """Specification for adding a new embedder."""

    name: Annotated[
        str,
        Field(..., description=SpecDoc.EMBEDDER_NAME),
    ]
    provider: Annotated[
        Literal["openai", "amazon-bedrock", "sentence-transformer"],
        Field(..., description=SpecDoc.EMBEDDER_PROVIDER_TYPE),
    ]
    config: Annotated[
        dict[str, Any],
        Field(..., description=SpecDoc.PROVIDER_CONFIG),
    ]


# --- Add Language Model Models ---


class AddOpenAIResponsesLanguageModelConfig(BaseModel):
    """Configuration for adding an OpenAI Responses language model."""

    api_key: Annotated[
        str,
        Field(..., description=SpecDoc.API_KEY_OPENAI),
    ]
    model: Annotated[
        str,
        Field(default="gpt-4o", description=SpecDoc.MODEL_NAME),
    ]
    base_url: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.API_BASE_URL),
    ]
    max_retry_interval_seconds: Annotated[
        int,
        Field(default=120, description=SpecDoc.MAX_RETRY_INTERVAL),
    ]


class AddOpenAIChatCompletionsLanguageModelConfig(BaseModel):
    """Configuration for adding an OpenAI Chat Completions language model."""

    api_key: Annotated[
        str,
        Field(..., description=SpecDoc.API_KEY_OPENAI),
    ]
    model: Annotated[
        str,
        Field(default="gpt-4o", description=SpecDoc.MODEL_NAME),
    ]
    base_url: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.API_BASE_URL),
    ]
    max_retry_interval_seconds: Annotated[
        int,
        Field(default=120, description=SpecDoc.MAX_RETRY_INTERVAL),
    ]


class AddAmazonBedrockLanguageModelConfig(BaseModel):
    """Configuration for adding an Amazon Bedrock language model."""

    region: Annotated[
        str,
        Field(..., description=SpecDoc.AWS_REGION),
    ]
    model_id: Annotated[
        str,
        Field(..., description=SpecDoc.MODEL_ID_BEDROCK),
    ]
    aws_access_key_id: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.AWS_ACCESS_KEY_ID),
    ]
    aws_secret_access_key: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.AWS_SECRET_ACCESS_KEY),
    ]
    aws_session_token: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.AWS_SESSION_TOKEN),
    ]
    max_retry_interval_seconds: Annotated[
        int,
        Field(default=120, description=SpecDoc.MAX_RETRY_INTERVAL),
    ]
    inference_config: Annotated[
        dict[str, Any] | None,
        Field(default=None, description=SpecDoc.INFERENCE_CONFIG),
    ]
    additional_model_request_fields: Annotated[
        dict[str, Any] | None,
        Field(default=None, description=SpecDoc.ADDITIONAL_MODEL_FIELDS),
    ]


class AddLanguageModelSpec(BaseModel):
    """Specification for adding a new language model."""

    name: Annotated[
        str,
        Field(..., description=SpecDoc.LM_NAME),
    ]
    provider: Annotated[
        Literal["openai-responses", "openai-chat-completions", "amazon-bedrock"],
        Field(..., description=SpecDoc.LM_PROVIDER_TYPE),
    ]
    config: Annotated[
        dict[str, Any],
        Field(..., description=SpecDoc.PROVIDER_CONFIG),
    ]


# --- Response Models ---


# --- Update Memory Configuration Models ---


class UpdateLongTermMemorySpec(BaseModel):
    """Partial update for long-term memory configuration."""

    embedder: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.LTM_EMBEDDER),
    ]
    reranker: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.LTM_RERANKER),
    ]
    vector_graph_store: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.LTM_VECTOR_GRAPH_STORE),
    ]


class UpdateShortTermMemorySpec(BaseModel):
    """Partial update for short-term memory configuration."""

    llm_model: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.STM_LLM_MODEL),
    ]
    message_capacity: Annotated[
        int | None,
        Field(default=None, gt=0, description=SpecDoc.STM_MESSAGE_CAPACITY),
    ]


class UpdateEpisodicMemorySpec(BaseModel):
    """Partial update for episodic memory configuration."""

    long_term_memory: Annotated[
        UpdateLongTermMemorySpec | None,
        Field(default=None, description=SpecDoc.EPISODIC_LTM_UPDATE),
    ]
    short_term_memory: Annotated[
        UpdateShortTermMemorySpec | None,
        Field(default=None, description=SpecDoc.EPISODIC_STM_UPDATE),
    ]
    long_term_memory_enabled: Annotated[
        bool | None,
        Field(default=None, description=SpecDoc.EPISODIC_LTM_ENABLED),
    ]
    short_term_memory_enabled: Annotated[
        bool | None,
        Field(default=None, description=SpecDoc.EPISODIC_STM_ENABLED),
    ]
    enabled: Annotated[
        bool | None,
        Field(default=None, description=SpecDoc.EPISODIC_ENABLED),
    ]


class UpdateSemanticMemorySpec(BaseModel):
    """Partial update for semantic memory configuration."""

    enabled: Annotated[
        bool | None,
        Field(default=None, description=SpecDoc.SEMANTIC_ENABLED),
    ]
    database: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_DATABASE),
    ]
    llm_model: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_LLM_MODEL),
    ]
    embedding_model: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_EMBEDDING_MODEL),
    ]
    cluster_split_reranker: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.SEMANTIC_CLUSTER_SPLIT_RERANKER),
    ]
    ingestion_trigger_messages: Annotated[
        int | None,
        Field(default=None, gt=0, description=SpecDoc.SEMANTIC_INGESTION_MESSAGES),
    ]
    ingestion_trigger_age_seconds: Annotated[
        int | None,
        Field(default=None, gt=0, description=SpecDoc.SEMANTIC_INGESTION_AGE),
    ]
    cluster_idle_ttl_seconds: Annotated[
        int | None,
        Field(default=None, gt=0, description=SpecDoc.SEMANTIC_CLUSTER_IDLE_TTL),
    ]
    cluster_similarity_threshold: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description=SpecDoc.SEMANTIC_CLUSTER_SIMILARITY_THRESHOLD,
        ),
    ]
    cluster_max_time_gap_seconds: Annotated[
        int | None,
        Field(default=None, gt=0, description=SpecDoc.SEMANTIC_CLUSTER_MAX_TIME_GAP),
    ]


class UpdateMemoryConfigSpec(BaseModel):
    """Specification for updating memory configuration."""

    episodic_memory: Annotated[
        UpdateEpisodicMemorySpec | None,
        Field(default=None, description=SpecDoc.UPDATE_EPISODIC_MEMORY),
    ]
    semantic_memory: Annotated[
        UpdateSemanticMemorySpec | None,
        Field(default=None, description=SpecDoc.UPDATE_SEMANTIC_MEMORY),
    ]


class UpdateMemoryConfigResponse(BaseModel):
    """Response model for memory configuration update."""

    success: Annotated[
        bool,
        Field(..., description=SpecDoc.OPERATION_SUCCESS),
    ]
    message: Annotated[
        str,
        Field(..., description=SpecDoc.STATUS_MESSAGE),
    ]


# --- Response Models ---


class UpdateResourceResponse(BaseModel):
    """Response model for resource update operations."""

    success: Annotated[
        bool,
        Field(..., description=SpecDoc.OPERATION_SUCCESS),
    ]
    status: Annotated[
        ResourceStatus,
        Field(..., description=SpecDoc.RESOURCE_STATUS_AFTER_OP),
    ]
    error: Annotated[
        str | None,
        Field(default=None, description=SpecDoc.OPERATION_ERROR),
    ]


class DeleteResourceResponse(BaseModel):
    """Response model for resource deletion operations."""

    success: Annotated[
        bool,
        Field(..., description=SpecDoc.DELETION_SUCCESS),
    ]
    message: Annotated[
        str,
        Field(..., description=SpecDoc.STATUS_MESSAGE),
    ]
