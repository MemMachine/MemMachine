"""Language model configuration models."""

from typing import Any, ClassVar, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator

from memmachine.common.configuration.mixin_confs import (
    ApiKeyMixin,
    AWSCredentialsMixin,
    MetricsFactoryIdMixin,
    YamlSerializableMixin,
)
from memmachine.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockConverseInferenceConfig,
)

DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434/v1"


class OpenAIResponsesLanguageModelConf(
    MetricsFactoryIdMixin, YamlSerializableMixin, ApiKeyMixin
):
    """Configuration for OpenAI Responses-compatible models."""

    model: str = Field(
        default="gpt-5-nano",
        description="OpenAI Responses API-compatible model",
    )
    api_key: SecretStr = Field(
        ...,
        description="OpenAI Responses API key for authentication, Can"
        "reference an environment variable using `$ENV` or `${ENV}` syntax ",
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


class OpenAIChatCompletionsLanguageModelConf(
    MetricsFactoryIdMixin, YamlSerializableMixin, ApiKeyMixin
):
    """Configuration for OpenAI Chat Completions-compatible models."""

    model: str = Field(
        default="gpt-5-nano",
        min_length=1,
        description="OpenAI Chat Completions API-compatible model",
    )
    base_url: str | None = Field(
        default=None,
        description="OpenAI Chat Completions API base URL",
        examples=[DEFAULT_OLLAMA_BASE_URL],
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


class AmazonBedrockLanguageModelConf(
    MetricsFactoryIdMixin, YamlSerializableMixin, AWSCredentialsMixin
):
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
    model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="ID of the Bedrock model to use for generation (e.g. 'openai.gpt-oss-20b-1:0').",
    )
    inference_config: AmazonBedrockConverseInferenceConfig | None = Field(
        default=None,
        description="Inference configuration for the Bedrock Converse API.",
    )
    additional_model_request_fields: dict[str, Any] | None = Field(
        default=None,
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

    def get_openai_responses_language_model_name(self) -> str | None:
        """Get the name of the first OpenAI Responses language model, if any."""
        if self.openai_responses_language_model_confs:
            return next(iter(self.openai_responses_language_model_confs))
        return None

    def get_openai_chat_completions_language_model_name(self) -> str | None:
        """Get the name of the first OpenAI Chat Completions language model, if any."""
        if self.openai_chat_completions_language_model_confs:
            return next(iter(self.openai_chat_completions_language_model_confs))
        return None

    def get_amazon_bedrock_language_model_name(self) -> str | None:
        """Get the name of the first Amazon Bedrock language model, if any."""
        if self.amazon_bedrock_language_model_confs:
            return next(iter(self.amazon_bedrock_language_model_confs))
        return None

    def get_openai_responses_language_model_conf(
        self, name: str
    ) -> OpenAIResponsesLanguageModelConf:
        """Get OpenAI Responses language model configuration by name."""
        return self.openai_responses_language_model_confs[name]

    def get_openai_chat_completions_language_model_conf(
        self, name: str
    ) -> OpenAIChatCompletionsLanguageModelConf:
        """Get OpenAI Chat Completions language model configuration by name."""
        return self.openai_chat_completions_language_model_confs[name]

    def get_amazon_bedrock_language_model_conf(
        self, name: str
    ) -> AmazonBedrockLanguageModelConf:
        """Get Amazon Bedrock language model configuration by name."""
        return self.amazon_bedrock_language_model_confs[name]

    OPENAI_RESPONSE: ClassVar[str] = "openai-responses"
    OPEN_CHAT_COMPLETION: ClassVar[str] = "openai-chat-completions"
    AMAZON_BEDROCK: ClassVar[str] = "amazon-bedrock"
    PROVIDER_KEY: ClassVar[str] = "provider"
    CONFIG_KEY: ClassVar[str] = "config"

    def to_yaml_dict(self) -> dict:
        """Serialize language model configurations to a YAML-compatible dictionary."""
        language_models: dict[str, dict] = {}

        def add_language_model(name: str, provider: str, config: dict) -> None:
            language_models[name] = {
                self.PROVIDER_KEY: provider,
                self.CONFIG_KEY: config,
            }

        for lm_id, cfg in self.openai_responses_language_model_confs.items():
            add_language_model(lm_id, self.OPENAI_RESPONSE, cfg.to_yaml_dict())

        for lm_id, cfg in self.openai_chat_completions_language_model_confs.items():
            add_language_model(lm_id, self.OPEN_CHAT_COMPLETION, cfg.to_yaml_dict())

        for lm_id, cfg in self.amazon_bedrock_language_model_confs.items():
            add_language_model(lm_id, self.AMAZON_BEDROCK, cfg.to_yaml_dict())

        return language_models

    def to_yaml(self) -> str:
        data = {"language_models": self.to_yaml_dict()}
        return yaml.safe_dump(data, sort_keys=True)

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
