"""Language model configuration models."""

from typing import Any, Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator

from memmachine.common.configuration.metrics_conf import WithMetricsFactoryId
from memmachine.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockConverseInferenceConfig,
)


class OpenAIResponsesLanguageModelConf(WithMetricsFactoryId):
    """Configuration for OpenAI Responses-compatible models."""

    model: str = Field(
        default="gpt-5-nano",
        description="OpenAI Responses API-compatible model",
    )
    api_key: SecretStr = Field(
        ...,
        description="OpenAI Responses API key for authentication",
    )
    base_url: str | None = Field(
        default=None,
        description="OpenAI Responses API base URL",
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


class OpenAIChatCompletionsLanguageModelConf(WithMetricsFactoryId):
    """Configuration for OpenAI Chat Completions-compatible models."""

    model: str = Field(
        default="gpt-5-nano",
        min_length=1,
        description="OpenAI Chat Completions API-compatible model",
    )
    api_key: SecretStr = Field(
        ...,
        description="OpenAI Chat Completions API key for authentication",
    )
    base_url: str | None = Field(
        default=None,
        description="OpenAI Chat Completions API base URL",
        examples=["http://host.docker.internal:11434/v1"],
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


class AmazonBedrockLanguageModelConf(WithMetricsFactoryId):
    """
    Configuration for AmazonBedrockLanguageModel.

    Attributes:
        region (str): AWS region where Bedrock is hosted (default: 'us-east-1').
        aws_access_key_id (SecretStr | None): AWS access key ID.
        aws_secret_access_key (SecretStr | None): AWS secret access key.
        aws_session_token (SecretStr | None): AWS session token.
        model_id (str): ID of the Bedrock model to use for generation.
        inference_config (AmazonBedrockConverseInferenceConfig | None): Inference config.
        additional_model_request_fields (dict[str, Any] | None): Extra request fields.
        max_retry_interval_seconds (int): Max retry interval when retrying API calls.

    """

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
        description="AWS session token for authentication.",
    )
    model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="ID of the Bedrock model to use for generation (e.g. 'openai.gpt-oss-20b-1:0').",
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


class LanguageModelsConf(BaseModel):
    """Top-level language model configuration container."""

    openai_responses_language_model_confs: dict[
        str,
        OpenAIResponsesLanguageModelConf,
    ] = {}
    openai_chat_completions_language_model_confs: dict[
        str,
        OpenAIChatCompletionsLanguageModelConf,
    ] = {}
    amazon_bedrock_language_model_confs: dict[str, AmazonBedrockLanguageModelConf] = {}

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        """Parse language model config definitions into typed models."""
        lm = input_dict.get("language_models", {})

        if isinstance(lm, cls):
            return lm

        openai_dict, aws_bedrock_dict, openai_chat_completions_dict = {}, {}, {}

        for lm_id, resource_definition in lm.items():
            provider = resource_definition.get("provider")
            conf = resource_definition.get("config", {})
            if provider == "openai-responses":
                openai_dict[lm_id] = OpenAIResponsesLanguageModelConf(**conf)
            elif provider == "openai-chat-completions":
                openai_chat_completions_dict[lm_id] = (
                    OpenAIChatCompletionsLanguageModelConf(
                        **conf,
                    )
                )
            elif provider == "amazon-bedrock":
                aws_bedrock_dict[lm_id] = AmazonBedrockLanguageModelConf(**conf)
            else:
                raise ValueError(
                    f"Unknown language model provider '{provider}' for language model id '{lm_id}'",
                )

        return cls(
            openai_responses_language_model_confs=openai_dict,
            amazon_bedrock_language_model_confs=aws_bedrock_dict,
            openai_chat_completions_language_model_confs=openai_chat_completions_dict,
        )
