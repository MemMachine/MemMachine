"""Configuration models for embedder providers."""

from typing import ClassVar, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, field_validator

from memmachine.common.configuration.mixin_confs import (
    ApiKeyMixin,
    AWSCredentialsMixin,
    MetricsFactoryIdMixin,
    YamlSerializableMixin,
)
from memmachine.common.data_types import SimilarityMetric


class AmazonBedrockEmbedderConf(YamlSerializableMixin, AWSCredentialsMixin):
    """Configuration for AmazonBedrockEmbedder."""

    region: str = Field(
        ...,
        description="AWS region where Bedrock is hosted.",
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


class OpenAIEmbedderConf(MetricsFactoryIdMixin, YamlSerializableMixin, ApiKeyMixin):
    """Configuration for OpenAI embedding models."""

    model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="OpenAI Embeddings API-compatible model",
    )
    dimensions: int | None = Field(
        default=1536,
        description="Dimensionality of the embeddings; must be provided if different from the default (1536)",
        gt=0,
    )
    base_url: str | None = Field(
        default=None,
        description="OpenAI Embeddings API base URL",
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


class SentenceTransformerEmbedderConfig(MetricsFactoryIdMixin, YamlSerializableMixin):
    """Configuration for sentence-transformer based embedders."""

    model: str = Field(
        ...,
        min_length=1,
        description="The name of the sentence transformer model.",
    )


class EmbeddersConf(BaseModel):
    """Top-level embedder configuration mapping provider ids to configs."""

    amazon_bedrock: dict[str, AmazonBedrockEmbedderConf] = {}
    openai: dict[str, OpenAIEmbedderConf] = {}
    sentence_transformer: dict[str, SentenceTransformerEmbedderConfig] = {}

    def get_amazon_bedrock_embedder_name(self) -> str | None:
        """Return the first Amazon Bedrock embedder id, if any."""
        if self.amazon_bedrock:
            return next(iter(self.amazon_bedrock.keys()))
        return None

    def get_openai_embedder_name(self) -> str | None:
        """Return the first OpenAI embedder id, if any."""
        if self.openai:
            return next(iter(self.openai.keys()))
        return None

    def get_sentence_transformer_embedder_name(self) -> str | None:
        """Return the first Sentence Transformer embedder id, if any."""
        if self.sentence_transformer:
            return next(iter(self.sentence_transformer.keys()))
        return None

    def get_openai_embedder_conf(self, name: str) -> OpenAIEmbedderConf:
        """Return the OpenAI embedder config for the given name."""
        return self.openai[name]

    def get_amazon_bedrock_embedder_conf(self, name: str) -> AmazonBedrockEmbedderConf:
        """Return the Amazon Bedrock embedder config for the given name."""
        return self.amazon_bedrock[name]

    def contains_embedder(self, embedder_id: str) -> bool:
        """Return if the embedder id is known."""
        return (
            embedder_id in self.amazon_bedrock
            or embedder_id in self.openai
            or embedder_id in self.sentence_transformer
        )

    OPENAI_KEY: ClassVar[str] = "openai"
    BEDROCK_KEY: ClassVar[str] = "amazon-bedrock"
    SENTENCE_TRANSFORMER_KEY: ClassVar[str] = "sentence-transformer"
    PROVIDER_KEY: ClassVar[str] = "provider"
    CONFIG_KEY: ClassVar[str] = "config"

    def to_yaml_dict(self) -> dict:
        """Return the embedder configuration as a YAML-serializable dictionary."""
        embedders: dict[str, dict] = {}

        def add_embedder(name: str, provider: str, config: dict) -> None:
            embedders[name] = {
                self.PROVIDER_KEY: provider,
                self.CONFIG_KEY: config,
            }

        for embedder_id, cfg in self.openai.items():
            add_embedder(embedder_id, self.OPENAI_KEY, cfg.to_yaml_dict())

        for embedder_id, cfg in self.amazon_bedrock.items():
            add_embedder(embedder_id, self.BEDROCK_KEY, cfg.to_yaml_dict())

        for embedder_id, cfg in self.sentence_transformer.items():
            add_embedder(embedder_id, self.SENTENCE_TRANSFORMER_KEY, cfg.to_yaml_dict())

        # Final structure
        return embedders

    def to_yaml(self) -> str:
        data = {"embedders": self.to_yaml_dict()}
        return yaml.safe_dump(data, sort_keys=True)

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        """Parse embedder config by provider and return the structured model."""
        embedder = input_dict.get("embedders", {})

        if isinstance(embedder, cls):
            return embedder

        amazon_bedrock_dict = {}
        openai_dict = {}
        sentence_transformer_dict = {}

        for embedder_id, resource_definition in embedder.items():
            provider = resource_definition.get(cls.PROVIDER_KEY)
            conf = resource_definition.get(cls.CONFIG_KEY, {})
            if provider == cls.OPENAI_KEY:
                openai_dict[embedder_id] = OpenAIEmbedderConf(**conf)
            elif provider == cls.BEDROCK_KEY:
                amazon_bedrock_dict[embedder_id] = AmazonBedrockEmbedderConf(**conf)
            elif provider == cls.SENTENCE_TRANSFORMER_KEY:
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
        return ret
