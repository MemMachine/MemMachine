"""Configuration models for embedder providers."""

from typing import Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, PrivateAttr, SecretStr, field_validator

from memmachine.common.configuration.metrics_conf import WithMetricsFactoryId
from memmachine.common.data_types import SimilarityMetric


class AmazonBedrockEmbedderConfig(BaseModel):
    """Configuration for AmazonBedrockEmbedder."""

    region: str = Field(
        ...,
        description="AWS region where Bedrock is hosted.",
    )
    aws_access_key_id: SecretStr | None = Field(
        ...,
        description="AWS access key ID for authentication.",
    )
    aws_secret_access_key: SecretStr | None = Field(
        ...,
        description="AWS secret access key for authentication.",
    )
    aws_session_token: SecretStr | None = Field(
        default=None,
        description="AWS session token for authentication (if applicable).",
    )
    model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="ID of the Bedrock model to use for embedding (e.g. 'amazon.titan-embed-text-v2:0').",
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE,
        description="Similarity metric to use",
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )


class OpenAIEmbedderConf(WithMetricsFactoryId):
    """Configuration for OpenAI embedding models."""

    model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="OpenAI Embeddings API-compatible model",
    )
    api_key: SecretStr = Field(
        ...,
        description="OpenAI Chat Completions API key for authentication",
        min_length=1,
    )
    dimensions: int | None = Field(
        default=1536,
        description="Dimensionality of the embeddings; must be provided if different from the default (1536)",
        gt=0,
    )
    base_url: str | None = Field(
        default=None,
        description=("OpenAI Embeddings API base URL"),
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls",
        gt=0,
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure the base URL includes a scheme and host."""
        if v is not None:
            parsed_url = urlparse(v)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid base URL: base_url={v}")
        return v


class SentenceTransformerEmbedderConfig(WithMetricsFactoryId):
    """Configuration for sentence-transformer based embedders."""

    model: str = Field(
        ...,
        min_length=1,
        description="The name of the sentence transformer model.",
    )


class EmbeddersConf(BaseModel):
    """Top-level embedder configuration mapping provider ids to configs."""

    amazon_bedrock: dict[str, AmazonBedrockEmbedderConfig] = {}
    openai: dict[str, OpenAIEmbedderConf] = {}
    sentence_transformer: dict[str, SentenceTransformerEmbedderConfig] = {}
    _saved_embedder_ids: set[str] = PrivateAttr(default_factory=set)

    def contains_embedder(self, embedder_id: str) -> bool:
        """Return if the embedder id is known."""
        return embedder_id in self._saved_embedder_ids

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        """Parse embedder config by provider and return the structured model."""
        embedder = input_dict.get("embedders", {})

        if isinstance(embedder, cls):
            return embedder

        amazon_bedrock_dict = {}
        openai_dict = {}
        sentence_transformer_dict = {}
        saved_embedder_ids = set(embedder.keys())

        for embedder_id, resource_definition in embedder.items():
            provider = resource_definition.get("provider")
            conf = resource_definition.get("config", {})
            if provider == "openai":
                openai_dict[embedder_id] = OpenAIEmbedderConf(**conf)
            elif provider == "amazon-bedrock":
                amazon_bedrock_dict[embedder_id] = AmazonBedrockEmbedderConfig(**conf)
            elif provider == "sentence-transformer":
                sentence_transformer_dict[embedder_id] = (
                    SentenceTransformerEmbedderConfig(**conf)
                )
            else:
                raise ValueError(
                    f"Unknown embedder provider '{provider}' for embedder id {embedder_id}",
                )
        ret = cls(
            amazon_bedrock=amazon_bedrock_dict,
            openai=openai_dict,
            sentence_transformer=sentence_transformer_dict,
        )
        ret._saved_embedder_ids = saved_embedder_ids
        return ret
