from typing import Any, Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator

from memmachine.common.configuration.metrics_conf import WithMetricsFactoryId
from memmachine.common.data_types import SimilarityMetric


class OpenAIModelConf(WithMetricsFactoryId):
    model: str = Field(default="gpt-5-nano", description="OpenAI model name")
    api_key: SecretStr = Field(
        ..., description="API key for OpenAPI authentication", min_length=1
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )


class AmazonBedrockConverseInferenceConfig(BaseModel):
    """
    Inference configuration for Amazon Bedrock Converse API.

    Attributes:
        max_tokens (int | None, optional):
            The maximum number of tokens to allow in the generated response.
            The default value is the maximum allowed value
            for the model that you are using.
        stop_sequences (list[str] | None, optional):
            A list of stop sequences that will stop response generation
            (default: None).
        temperature (float | None, optional):
            What sampling temperature to use, between 0 and 1.
            The default value is the default value
            for the model that you are using, applied when None
            (default: None).
        top_p (float | None, optional):
            The percentage of probability mass to consider for the next token
            (default: None).
    """

    max_tokens: int | None = Field(
        default=None,
        description=(
            "The maximum number of tokens to allow in the generated response. "
            "The default value is the maximum allowed value "
            "for the model that you are using, applied when None"
            "(default: None)."
        ),
        gt=0,
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description=(
            "A list of stop sequences that will stop response generation "
            "(default: None)."
        ),
    )
    temperature: float | None = Field(
        default=None,
        description=(
            "What sampling temperature to use, between 0 and 1. "
            "The default value is the default value "
            "for the model that you are using, applied when None "
            "(default: None)."
        ),
        ge=0.0,
        le=1.0,
    )
    top_p: float | None = Field(
        default=None,
        description=(
            "The percentage of probability mass to consider for the next token "
            "(default: None)."
        ),
        ge=0.0,
        le=1.0,
    )


class AwsBedrockModelConf(WithMetricsFactoryId):
    """
    Configuration for AmazonBedrockLanguageModel.

    Attributes:
        region (str):
            AWS region where Bedrock is hosted.
        aws_access_key_id (SecretStr):
            AWS access key ID for authentication.
        aws_secret_access_key (SecretStr):
            AWS secret access key for authentication.
        model_id (str):
            ID of the Bedrock model to use for generation
            (e.g. 'openai.gpt-oss-20b-1:0').
        inference_config (AmazonBedrockConverseInferenceConfig | None, optional):
            Inference configuration for the Bedrock Converse API
            (default: None).
        additional_model_request_fields (dict[str, Any] | None, optional):
            Keys are request fields for the model
            and values are values for those fields
            (default: None).
        max_retry_interval_seconds (int, optional):
            Maximal retry interval in seconds when retrying API calls
            (default: 120).
    """

    region: str = Field(
        default="us-east-1", description="AWS region where Bedrock is hosted."
    )
    aws_access_key_id: SecretStr = Field(
        default=SecretStr(""), description="AWS access key ID for authentication."
    )
    aws_secret_access_key: SecretStr = Field(
        default=SecretStr(""), description="AWS secret access key for authentication."
    )
    aws_session_token: SecretStr | None = Field(
        default=None,
        description="AWS session token for authentication.",
    )
    model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="ID of the Bedrock model to use for generation "
        "(e.g. 'openai.gpt-oss-20b-1:0').",
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Similarity metric to use"
    )
    inference_config: AmazonBedrockConverseInferenceConfig | None = Field(
        default=None,
        description="Inference configuration for the Bedrock Converse API.",
    )
    additional_model_request_fields: dict[str, Any] | None = Field(
        None,
        description=(
            "Keys are request fields for the model "
            "and values are values for those fields "
            "(default: None)."
        ),
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )


class OpenAICompatibleModelConf(WithMetricsFactoryId):
    model: str = Field(
        default="nomic-embed-text",
        min_length=1,
        description="OpenAI Compatible models such as Ollama, vLLM, SGLang",
    )
    api_key: SecretStr = Field(
        ...,
        description="API key for OpenAI Compatible models such as Ollama, vLLM, SGLang",
        min_length=1,
    )
    base_url: str = Field(
        ...,
        description="API base URL for OpenAI Compatible models such as Ollama, vLLM, SGLang",
        min_length=1,
        examples=["http://host.docker.internal:11434/v1"],
    )
    dimensions: int = Field(default=768, description="Embedding dimensions")
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if v is not None:
            parsed_url = urlparse(v)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid base URL: base_url={v}")
        return v


class LanguageModelConf(BaseModel):
    openai_confs: dict[str, OpenAIModelConf] = {}
    aws_bedrock_confs: dict[str, AwsBedrockModelConf] = {}
    openai_compatible_confs: dict[str, OpenAICompatibleModelConf] = {}

    @classmethod
    def parse_language_model_conf(cls, input_dict: dict) -> Self:
        lm = input_dict
        for key in ["language_model", "model", "language-model"]:
            if key in lm:
                lm = input_dict.get(key, {})
                break

        openai_dict, aws_bedrock_dict, openai_compatible_dict = {}, {}, {}
        for lm_id, conf in lm.items():
            vendor = conf.get("model_vendor").lower()
            if vendor == "openai":
                openai_dict[lm_id] = OpenAIModelConf(**conf)
            elif vendor == "amazon-bedrock":
                aws_bedrock_dict[lm_id] = AwsBedrockModelConf(**conf)
            elif vendor in ["vllm", "sglang", "openai-compatible"]:
                openai_compatible_dict[lm_id] = OpenAICompatibleModelConf(**conf)
            else:
                raise ValueError(
                    f"Unknown vendor_name '{lm_id}' for language model id '{lm_id}'"
                )

        return cls(
            openai_confs=openai_dict,
            aws_bedrock_confs=aws_bedrock_dict,
            openai_compatible_confs=openai_compatible_dict,
        )
