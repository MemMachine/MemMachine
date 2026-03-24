"""Retrieval-agent configuration models."""

from enum import StrEnum
from typing import Literal

from pydantic import Field, SecretStr, field_validator

from memmachine_server.common.configuration.mixin_confs import (
    WithValueFromEnv,
    YamlSerializableMixin,
)


class RetrievalAgentSessionProvider(StrEnum):
    """Supported providers for retrieval-agent multi-turn execution."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RetrievalAgentConf(YamlSerializableMixin, WithValueFromEnv):
    """Configuration for top-level retrieval-agent orchestration."""

    llm_model: str | None = Field(
        default=None,
        description="Default language model used by retrieval-agent strategies.",
    )
    reranker: str | None = Field(
        default=None,
        description="Default reranker used by retrieval-agent strategies.",
    )
    agent_session_provider: RetrievalAgentSessionProvider = Field(
        default=RetrievalAgentSessionProvider.OPENAI,
        description="Provider runtime used for retrieval-agent multi-turn sessions.",
    )
    agent_session_timeout_seconds: int = Field(
        default=180,
        gt=0,
        description="Global timeout budget for each retrieval-agent session.",
    )
    agent_session_max_combined_calls: int = Field(
        default=10,
        gt=0,
        description=(
            "Combined call budget for top-level tool calls plus sub-agent tool calls "
            "within one retrieval session."
        ),
    )
    agent_session_log_raw_output: bool = Field(
        default=True,
        description="Whether to emit full provider raw output payloads to debug logs.",
    )
    agent_native_bundle_root: str | None = Field(
        default=None,
        description=(
            "Optional directory used to materialize markdown bundles for provider "
            "attachments. If omitted, the system temp directory is used."
        ),
    )
    agent_session_max_retry_interval_seconds: int = Field(
        default=120,
        gt=0,
        description="Retry backoff cap for provider session API retries.",
    )
    openai_native_agent_environment: Literal["local", "container_auto"] = Field(
        default="local",
        description="Shell environment type used for OpenAI provider attachments.",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-5",
        min_length=1,
        description="Anthropic model ID used when agent_session_provider=anthropic.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Anthropic API key used when agent_session_provider=anthropic. "
            "Can reference environment variables with $ENV or ${ENV} syntax."
        ),
    )
    anthropic_base_url: str | None = Field(
        default=None,
        description="Optional Anthropic API base URL.",
    )
    anthropic_max_output_tokens: int = Field(
        default=2048,
        gt=0,
        description="Max output tokens per Anthropic messages call in agent sessions.",
    )

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _resolve_anthropic_api_key(
        cls,
        value: SecretStr | str | None,
    ) -> SecretStr | None:
        if value is None:
            return None
        resolved = cls._resolve_env(value)
        if isinstance(resolved, SecretStr):
            return resolved
        if isinstance(resolved, str):
            return SecretStr(resolved)
        raise TypeError("anthropic_api_key must be a string, SecretStr, or null")
