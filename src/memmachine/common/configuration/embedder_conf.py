from typing import Self

from pydantic import BaseModel, Field, SecretStr

from memmachine.common.configuration.metrics_conf import WithMetricsFactoryId
from memmachine.common.data_types import SimilarityMetric


class AmazonBedrockEmbedderConfig(BaseModel):
    """
    Configuration for AmazonBedrockEmbedder.

    Attributes:
        region (str):
            AWS region where Bedrock is hosted.
        aws_access_key_id (SecretStr):
            AWS access key ID for authentication.
        aws_secret_access_key (SecretStr):
            AWS secret access key for authentication.
        model_id (str):
            ID of the Bedrock model to use for embedding
            (e.g. 'amazon.titan-embed-text-v2:0').
        similarity_metric (SimilarityMetric):
            Similarity metric to use for comparing embeddings
            (default: SimilarityMetric.COSINE).
        max_retry_interval_seconds (int, optional):
            Maximal retry interval in seconds
            (default: 120).
    """

    region: str = Field(
        "us-west-2",
        description="AWS region where Bedrock is hosted.",
    )
    aws_access_key_id: SecretStr = Field(
        default=SecretStr(""), description="AWS access key ID for authentication."
    )
    aws_secret_access_key: SecretStr = Field(
        default=SecretStr(""), description="AWS secret access key for authentication."
    )
    model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="ID of the Bedrock model to use for generation "
        "(e.g. 'openai.gpt-oss-20b-1:0').",
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Similarity metric to use"
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )


class OpenAIEmbedderConf(WithMetricsFactoryId):
    model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="OpenAPI embedding model",
    )
    api_key: SecretStr = Field(
        ...,
        description="API key for OpenAPI authentication",
        min_length=1,
    )
    dimensions: int | None = Field(
        default=1536,
        description="Dimensionality of the embeddings (default: model-specific).",
        gt=0,
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for OpenAPI API requests. If set to None, "
            "the environment variable OPENAI_BASE_URL will be used. "
            "(default: None)."
        ),
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )


class SentenceTransformerEmbedderConfig(WithMetricsFactoryId):
    model: str = Field(
        ..., min_length=1, description="The name of the sentence transformer model."
    )


class EmbedderConf(BaseModel):
    amazon_bedrock: dict[str, AmazonBedrockEmbedderConfig] = {}
    openai: dict[str, OpenAIEmbedderConf] = {}
    sentence_transformer: dict[str, SentenceTransformerEmbedderConfig] = {}

    @classmethod
    def parse_embedder_conf(cls, input_dict: dict) -> Self:
        embedder = input_dict
        if "embedder" in embedder:
            embedder = embedder["embedder"]

        amazon_bedrock_dict = {}
        openai_dict = {}
        sentence_transformer_dict = {}

        for embedder_id, value in embedder.items():
            name = value.get("provider")
            conf = value.get("config", {})
            if name == "openai":
                openai_dict[embedder_id] = OpenAIEmbedderConf(**conf)
            elif name == "amazon-bedrock":
                amazon_bedrock_dict[embedder_id] = AmazonBedrockEmbedderConfig(**conf)
            elif name == "sentence-transformer":
                sentence_transformer_dict[embedder_id] = (
                    SentenceTransformerEmbedderConfig(**conf)
                )
            else:
                raise ValueError(f"Unknown embedder: {name}")
        return cls(
            amazon_bedrock=amazon_bedrock_dict,
            openai=openai_dict,
            sentence_transformer=sentence_transformer_dict,
        )
