"""Language model configuration models."""

from typing import Any, ClassVar, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator

from memmachine_server.common.configuration.mixin_confs import (
    ApiKeyMixin,
    AWSCredentialsMixin,
    MetricsFactoryIdMixin,
    YamlSerializableMixin,
)
from memmachine_server.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockConverseInferenceConfig,
)

DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434/v1"


def _clean_empty_lm_config(conf: dict) -> dict:
    """Remove empty strings and None values from config."""
    cleaned: dict = {}
    for key, value in (conf or {}).items():
        if isinstance(value, str) and value.strip() == "":
            continue
        if value is None:
            continue
        cleaned[key] = value
    return cleaned


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


class LiteLLMLanguageModelConf(
    MetricsFactoryIdMixin, YamlSerializableMixin, ApiKeyMixin
):
    """Configuration for LiteLLM-backed language models.

    LiteLLM routes a single `model` spec (e.g. ``anthropic/claude-sonnet-4-6``)
    to whichever underlying provider the prefix selects, picking up that
    backing's standard env vars (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
    ``AWS_*``, ...) at call time. Set ``api_base`` to point at a LiteLLM
    proxy server for centralized credential management.
    """

    model: str = Field(
        ...,
        min_length=1,
        description=(
            "LiteLLM model spec, e.g. 'anthropic/claude-sonnet-4-6', "
            "'openai/gpt-4o', 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0'."
        ),
    )
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Optional explicit API key. Most users leave this unset and let "
            "LiteLLM resolve credentials from each backing's standard env var "
            "(ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) at call time. Set this "
            "only when routing through a LiteLLM proxy."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Optional base URL. Set to a LiteLLM proxy endpoint "
            "(e.g. 'http://localhost:4000') for proxy mode."
        ),
    )
    api_version: str | None = Field(
        default=None,
        description="Optional API version (Azure-style endpoints).",
    )
    drop_params: bool = Field(
        default=True,
        description=(
            "Forward `drop_params=True` to litellm.acompletion so unsupported "
            "kwargs are stripped per backing rather than raising."
        ),
    )
    extra_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra kwargs forwarded verbatim to litellm.acompletion.",
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Max retry interval in seconds when retrying API calls.",
        gt=0,
    )

    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, v: str | None) -> str | None:
        """Ensure the base URL includes a scheme and host when provided."""
        if v is not None:
            parsed_url = urlparse(v)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid api_base URL: api_base={v}")
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
    litellm_language_model_confs: dict[str, LiteLLMLanguageModelConf] = {}

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

    def get_litellm_language_model_name(self) -> str | None:
        """Get the name of the first LiteLLM language model, if any."""
        if self.litellm_language_model_confs:
            return next(iter(self.litellm_language_model_confs))
        return None

    def get_litellm_language_model_conf(self, name: str) -> "LiteLLMLanguageModelConf":
        """Get LiteLLM language model configuration by name."""
        return self.litellm_language_model_confs[name]

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

    def contains_language_model(self, language_model_id: str) -> bool:
        """Return whether the language model id is known."""
        return (
            language_model_id in self.openai_responses_language_model_confs
            or language_model_id in self.openai_chat_completions_language_model_confs
            or language_model_id in self.amazon_bedrock_language_model_confs
            or language_model_id in self.litellm_language_model_confs
        )

    OPENAI_RESPONSE: ClassVar[str] = "openai-responses"
    OPEN_CHAT_COMPLETION: ClassVar[str] = "openai-chat-completions"
    AMAZON_BEDROCK: ClassVar[str] = "amazon-bedrock"
    LITELLM: ClassVar[str] = "litellm"
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

        for lm_id, cfg in self.litellm_language_model_confs.items():
            add_language_model(lm_id, self.LITELLM, cfg.to_yaml_dict())

        return language_models

    def to_yaml(self) -> str:
        data = {"language_models": self.to_yaml_dict()}
        return yaml.safe_dump(data, sort_keys=True)

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        """Parse language model config definitions into typed models."""
        lm = input_dict.get("language_models", {})

        if lm is None:
            lm = {}

        if isinstance(lm, cls):
            return lm

        openai_dict, aws_bedrock_dict, openai_chat_completions_dict, litellm_dict = {}, {}, {}, {}

        for lm_id, resource_definition in lm.items():
            provider = resource_definition.get("provider")
            conf = _clean_empty_lm_config(resource_definition.get("config", {}))
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
            elif provider == "litellm":
                litellm_dict[lm_id] = LiteLLMLanguageModelConf(**conf)
            else:
                raise ValueError(
                    f"Unknown language model provider '{provider}' for language model id '{lm_id}'",
                )

        return cls(
            openai_responses_language_model_confs=openai_dict,
            amazon_bedrock_language_model_confs=aws_bedrock_dict,
            openai_chat_completions_language_model_confs=openai_chat_completions_dict,
            litellm_language_model_confs=litellm_dict,
        )
