"""Reranker configuration models."""

from typing import Self

from pydantic import BaseModel, Field, SecretStr


class BM25RerankerConf(BaseModel):
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


class AmazonBedrockRerankerConf(BaseModel):
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


class CrossEncoderRerankerConf(BaseModel):
    """Parameters for CrossEncoderReranker."""

    model_name: str = Field(
        default="cross-encoder/qnli-electra-base",
        description="The cross-encoder model name to use for reranking",
        min_length=1,
    )


class EmbedderRerankerConf(BaseModel):
    """Parameters for EmbedderReranker."""

    embedder_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="The embedder model name to use for reranking",
    )


class IdentityRerankerConf(BaseModel):
    """Parameters for IdentityReranker."""


class RRFHybridRerankerConf(BaseModel):
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

    @classmethod
    def parse_reranker_conf(cls, input_dict: dict) -> Self:
        """Parse reranker configuration from a raw mapping."""
        reranker = input_dict.get("rerankers", {})

        bm25_dict = {}
        amazon_bedrock_dict = {}
        cross_encoder_dict = {}
        embedder_dict = {}
        identity_dict = {}
        rrf_hybrid_dict = {}

        for reranker_id, value in reranker.items():
            provider = value.get("provider")
            conf = value.get("config", {})
            if provider == "bm25":
                bm25_dict[reranker_id] = BM25RerankerConf(**conf)
            elif provider == "amazon-bedrock":
                amazon_bedrock_dict[reranker_id] = AmazonBedrockRerankerConf(**conf)
            elif provider == "cross-encoder":
                cross_encoder_dict[reranker_id] = CrossEncoderRerankerConf(**conf)
            elif provider == "embedder":
                embedder_dict[reranker_id] = EmbedderRerankerConf(**conf)
            elif provider == "identity":
                identity_dict[reranker_id] = IdentityRerankerConf()
            elif provider == "rrf-hybrid":
                rrf_hybrid_dict[reranker_id] = RRFHybridRerankerConf(**conf)
            else:
                raise ValueError(
                    f"Unknown reranker_type '{provider}' for reranker id '{reranker_id}'",
                )

        return cls(
            bm25=bm25_dict,
            amazon_bedrock=amazon_bedrock_dict,
            cross_encoder=cross_encoder_dict,
            embedder=embedder_dict,
            identity=identity_dict,
            rrf_hybrid=rrf_hybrid_dict,
        )
