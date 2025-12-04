"""Reranker configuration models."""

from typing import ClassVar, Self

import yaml
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from memmachine.common.configuration.mixin_confs import YamlSerializableMixin


class BM25RerankerConf(YamlSerializableMixin):
    """Parameters for BM25Reranker."""

    k1: float = Field(default=1.5, description="BM25 k1 parameter")
    b: float = Field(default=0.75, description="BM25 b parameter")
    epsilon: float = Field(default=0.25, description="BM25 epsilon parameter")
    tokenizer: str = Field(
        default="default",
        description="Tokenizer function to split text into tokens",
    )
    language: str = Field(
        default="english",
        description="Language for stop words in default tokenizer",
    )


class AmazonBedrockRerankerConf(YamlSerializableMixin):
    """Parameters for AmazonBedrockReranker."""

    model_id: str = Field(..., description="The Bedrock model ID to use for reranking")
    region: str = Field(
        ...,
        description="The AWS region of the Bedrock service",
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
        description="AWS session token for authentication.",
    )
    additional_model_request_fields: dict = Field(
        default_factory=dict,
        description="Additional fields to include in the Bedrock model request",
    )


class CrossEncoderRerankerConf(YamlSerializableMixin):
    """Parameters for CrossEncoderReranker."""

    model_name: str = Field(
        default="cross-encoder/qnli-electra-base",
        description="The cross-encoder model name to use for reranking",
        min_length=1,
    )


class EmbedderRerankerConf(YamlSerializableMixin):
    """Parameters for EmbedderReranker."""

    embedder_id: str = Field(
        ...,
        description="The embedder model resource id to use for reranking",
    )


class IdentityRerankerConf(YamlSerializableMixin):
    """Parameters for IdentityReranker."""


class RRFHybridRerankerConf(YamlSerializableMixin):
    """Parameters for RrfHybridReranker."""

    reranker_ids: list[str] = Field(
        ...,
        description="The IDs of the rerankers to combine in the hybrid",
        examples=["bm", "cross-encoder"],
    )
    k: int = Field(default=60, description="The k parameter for RRF scoring")


class RerankersConf(BaseModel):
    """Top-level configuration for available rerankers."""

    bm25: dict[str, BM25RerankerConf] = {}
    amazon_bedrock: dict[str, AmazonBedrockRerankerConf] = {}
    cross_encoder: dict[str, CrossEncoderRerankerConf] = {}
    embedder: dict[str, EmbedderRerankerConf] = {}
    identity: dict[str, IdentityRerankerConf] = {}
    rrf_hybrid: dict[str, RRFHybridRerankerConf] = {}

    _saved_reranker_ids: set[str] = PrivateAttr(default_factory=set)

    def contains_reranker(self, reranker_id: str) -> bool:
        """Check if a reranker ID is defined in the configuration."""
        return reranker_id in self._saved_reranker_ids

    BM25: ClassVar[str] = "bm25"
    CROSS_ENCODER: ClassVar[str] = "cross-encoder"
    EMBEDDER: ClassVar[str] = "embedder"
    IDENTITY: ClassVar[str] = "identity"
    RRF_HYBRID: ClassVar[str] = "rrf-hybrid"
    AMAZON_BEDROCK: ClassVar[str] = "amazon-bedrock"
    PROVIDER_KEY: ClassVar[str] = "provider"
    CONFIG_KEY: ClassVar[str] = "config"

    def to_yaml_dict(self) -> dict:
        """Serialize the reranker configuration to a YAML-compatible dictionary."""
        rerankers: dict[str, dict] = {}

        def add_reranker(name: str, provider: str, config: dict) -> None:
            """Add a reranker entry to the YAML dictionary."""
            rerankers[name] = {
                self.PROVIDER_KEY: provider,
                self.CONFIG_KEY: config,
            }

        for reranker_id, conf in self.bm25.items():
            add_reranker(reranker_id, self.BM25, conf.to_yaml_dict())

        for reranker_id, conf in self.amazon_bedrock.items():
            add_reranker(reranker_id, self.AMAZON_BEDROCK, conf.to_yaml_dict())

        for reranker_id, conf in self.cross_encoder.items():
            add_reranker(reranker_id, self.CROSS_ENCODER, conf.to_yaml_dict())

        for reranker_id, conf in self.embedder.items():
            add_reranker(reranker_id, self.EMBEDDER, conf.to_yaml_dict())

        for reranker_id, conf in self.identity.items():
            add_reranker(reranker_id, self.IDENTITY, conf.to_yaml_dict())

        for reranker_id, conf in self.rrf_hybrid.items():
            add_reranker(reranker_id, self.RRF_HYBRID, conf.to_yaml_dict())

        return rerankers

    def to_yaml(self) -> str:
        data = {"rerankers": self.to_yaml_dict()}
        return yaml.safe_dump(data, sort_keys=True)

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        """Parse reranker configuration from a raw mapping."""
        reranker = input_dict.get("rerankers", {})

        if isinstance(reranker, cls):
            return reranker

        bm25_dict = {}
        amazon_bedrock_dict = {}
        cross_encoder_dict = {}
        embedder_dict = {}
        identity_dict = {}
        rrf_hybrid_dict = {}
        saved_reranker_ids = set(reranker.keys())

        for reranker_id, value in reranker.items():
            provider = value.get(cls.PROVIDER_KEY)
            conf = value.get(cls.CONFIG_KEY, {})
            if provider == cls.BM25:
                bm25_dict[reranker_id] = BM25RerankerConf(**conf)
            elif provider == cls.AMAZON_BEDROCK:
                amazon_bedrock_dict[reranker_id] = AmazonBedrockRerankerConf(**conf)
            elif provider == cls.CROSS_ENCODER:
                cross_encoder_dict[reranker_id] = CrossEncoderRerankerConf(**conf)
            elif provider == cls.EMBEDDER:
                embedder_dict[reranker_id] = EmbedderRerankerConf(**conf)
            elif provider == cls.IDENTITY:
                identity_dict[reranker_id] = IdentityRerankerConf()
            elif provider == cls.RRF_HYBRID:
                rrf_hybrid_dict[reranker_id] = RRFHybridRerankerConf(**conf)
            else:
                raise ValueError(
                    f"Unknown reranker_type '{provider}' for reranker id '{reranker_id}'",
                )

        ret = cls(
            bm25=bm25_dict,
            amazon_bedrock=amazon_bedrock_dict,
            cross_encoder=cross_encoder_dict,
            embedder=embedder_dict,
            identity=identity_dict,
            rrf_hybrid=rrf_hybrid_dict,
        )
        ret._saved_reranker_ids = saved_reranker_ids
        return ret
