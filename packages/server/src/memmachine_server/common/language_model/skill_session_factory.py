"""Factory helpers for provider-specific retrieval skill session runtimes."""

from __future__ import annotations

import os

from memmachine_server.common.configuration.retrieval_config import (
    RetrievalAgentConf,
    RetrievalSkillSessionProvider,
)
from memmachine_server.common.language_model.language_model import LanguageModel

from .skill_anthropic_session_language_model import (
    SkillAnthropicSessionLanguageModel,
    SkillAnthropicSessionLanguageModelParams,
)
from .skill_openai_session_language_model import (
    SkillLanguageModel,
    SkillOpenAISessionLanguageModelParams,
    SkillSessionModelProtocol,
)


def create_skill_session_model(
    *,
    model: LanguageModel,
    retrieval_conf: RetrievalAgentConf,
) -> SkillSessionModelProtocol:
    """Build the configured provider runtime for retrieval skill live sessions."""
    provider = retrieval_conf.skill_session_provider
    if provider == RetrievalSkillSessionProvider.OPENAI:
        # Keep existing OpenAI behavior while allowing explicit raw-output logging.
        from .openai_responses_language_model import OpenAIResponsesLanguageModel

        if not isinstance(model, OpenAIResponsesLanguageModel):
            raise TypeError(
                "OpenAI skill session provider requires OpenAIResponsesLanguageModel."
            )
        return SkillLanguageModel(
            SkillOpenAISessionLanguageModelParams(
                client=model.client,
                model=model.model_name,
                max_retry_interval_seconds=model.max_retry_interval_seconds,
                reasoning_effort=model.reasoning_effort,
                log_raw_output=retrieval_conf.skill_session_log_raw_output,
                native_skill_environment=retrieval_conf.openai_native_skill_environment,
            )
        )

    if provider == RetrievalSkillSessionProvider.ANTHROPIC:
        api_key = _resolve_anthropic_api_key(retrieval_conf)
        try:
            import anthropic
        except ImportError as err:
            raise RuntimeError(
                "Anthropic skill session provider requires the 'anthropic' package."
            ) from err

        client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=retrieval_conf.anthropic_base_url,
        )
        return SkillAnthropicSessionLanguageModel(
            SkillAnthropicSessionLanguageModelParams(
                client=client,
                model=retrieval_conf.anthropic_model,
                max_retry_interval_seconds=(
                    retrieval_conf.skill_session_max_retry_interval_seconds
                ),
                max_output_tokens=retrieval_conf.anthropic_max_output_tokens,
                log_raw_output=retrieval_conf.skill_session_log_raw_output,
            )
        )

    raise ValueError(f"Unsupported skill session provider: {provider}")


def _resolve_anthropic_api_key(conf: RetrievalAgentConf) -> str:
    secret = conf.anthropic_api_key
    if secret is not None:
        resolved = secret.get_secret_value().strip()
        if resolved:
            return resolved
    from_env = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if from_env:
        return from_env
    raise ValueError(
        "Anthropic skill session provider requires anthropic_api_key config or "
        "ANTHROPIC_API_KEY environment variable."
    )
