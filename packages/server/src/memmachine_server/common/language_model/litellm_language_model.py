"""LiteLLM-backed language model implementation.

Routes calls through the LiteLLM SDK (`litellm.acompletion`) so any of the
100+ providers LiteLLM supports (OpenAI, Anthropic, Bedrock, Vertex AI,
Cohere, Mistral, Groq, Perplexity, Together, Fireworks, Cerebras, ...) is
reachable through a single LanguageModel implementation. LiteLLM normalizes
every backing's response to OpenAI's chat-completions shape, so this class
inherits the parsing, streaming, and tool-call logic from
`OpenAIChatCompletionsLanguageModel` and only swaps the underlying request
function.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .openai_chat_completions_language_model import (
    OpenAIChatCompletionsLanguageModel,
)

logger = logging.getLogger(__name__)


class LiteLLMLanguageModelParams(BaseModel):
    """
    Parameters for LiteLLMLanguageModel.

    Attributes:
        model (str):
            The LiteLLM model spec, e.g. 'anthropic/claude-sonnet-4-6',
            'openai/gpt-4o', 'bedrock/anthropic.claude-3-5-sonnet-...'.
        api_key (str | None):
            Optional explicit API key. When set, takes precedence over
            LiteLLM's per-backing env-var resolution. Useful for proxy mode.
        api_base (str | None):
            Optional base URL. Set to a LiteLLM proxy address
            (e.g. 'http://localhost:4000') for proxy mode; leave None for
            direct (embedded) mode where each backing's standard env vars
            (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) are used.
        api_version (str | None):
            Optional API version (Azure-style endpoints).
        drop_params (bool):
            When True (default), LiteLLM strips kwargs the chosen backing
            doesn't accept rather than erroring. Mirrors the lenient
            behavior of MemMachine's existing OpenAI-compatible adapters.
        extra_kwargs (dict[str, Any] | None):
            Additional kwargs forwarded verbatim to `litellm.acompletion`,
            useful for routing-specific options (`metadata`, `tags`, ...).
        max_retry_interval_seconds (int):
            Max retry interval when retrying API calls (default: 120).
        metrics_factory (MetricsFactory | None):
            Optional metrics factory.
    """

    model: str = Field(
        ...,
        description=(
            "LiteLLM model spec (e.g. 'anthropic/claude-sonnet-4-6', "
            "'openai/gpt-4o', 'bedrock/anthropic.claude-3-5-sonnet-...')"
        ),
        min_length=1,
    )
    api_key: str | None = Field(
        default=None,
        description="Optional explicit API key (proxy mode or override).",
    )
    api_base: str | None = Field(
        default=None,
        description="Optional base URL (LiteLLM proxy or backing override).",
    )
    api_version: str | None = Field(
        default=None,
        description="Optional API version (Azure-style endpoints).",
    )
    drop_params: bool = Field(
        default=True,
        description=(
            "When True (default), LiteLLM strips kwargs the chosen backing "
            "doesn't accept rather than erroring."
        ),
    )
    extra_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra kwargs forwarded to litellm.acompletion.",
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Max retry interval when retrying API calls.",
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        default=None,
        description="Optional metrics factory for usage counters.",
    )


class LiteLLMLanguageModel(OpenAIChatCompletionsLanguageModel):
    """LanguageModel routing through the LiteLLM SDK.

    Reuses every parsing, streaming, tool-call, and metrics path from
    `OpenAIChatCompletionsLanguageModel`. Only the request function is
    swapped to call `litellm.acompletion(**args)` instead of
    `client.chat.completions.create(**args)`.
    """

    def __init__(self, params: LiteLLMLanguageModelParams) -> None:
        """Initialize the LiteLLM language model with the given params."""
        # Skip the parent `__init__` to avoid requiring an AsyncOpenAI
        # client (LiteLLM has its own routing). Re-create only the parent
        # state our overrides + inherited methods rely on.
        # Calling LanguageModel (the abstract grandparent) is enough.
        super(OpenAIChatCompletionsLanguageModel, self).__init__()

        self._client = None  # not used; kept for parent attribute parity
        self._model = params.model
        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        self._litellm_api_key = params.api_key
        self._litellm_api_base = params.api_base
        self._litellm_api_version = params.api_version
        self._litellm_drop_params = params.drop_params
        self._litellm_extra_kwargs = dict(params.extra_kwargs or {})

        self._tracker = OperationTracker(
            params.metrics_factory, prefix="language_model_litellm"
        )

        self._should_collect_metrics = False
        if params.metrics_factory is not None:
            self._should_collect_metrics = True
            self._input_tokens_usage_counter = params.metrics_factory.get_counter(
                "language_model_litellm_usage_input_tokens",
                "Number of input tokens used for LiteLLM language model",
            )
            self._output_tokens_usage_counter = params.metrics_factory.get_counter(
                "language_model_litellm_usage_output_tokens",
                "Number of output tokens used for LiteLLM language model",
            )
            self._total_tokens_usage_counter = params.metrics_factory.get_counter(
                "language_model_litellm_usage_total_tokens",
                "Number of tokens used for LiteLLM language model",
            )

    async def _request_chat_completion(
        self,
        args: dict[str, Any],
        max_attempts: int,
        generate_response_call_uuid: object,
    ) -> ChatCompletion | AsyncIterator[object] | object:
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMLanguageModel. "
                "Install it with: pip install memmachine-server[litellm]"
            ) from e

        merged_args: dict[str, Any] = dict(args)
        # MemMachine's parent OpenAI implementation passes `store=False`,
        # which only OpenAI's API itself recognizes; stripping unsupported
        # kwargs is exactly what LiteLLM's `drop_params` does.
        merged_args.setdefault("drop_params", self._litellm_drop_params)
        if self._litellm_api_key and "api_key" not in merged_args:
            merged_args["api_key"] = self._litellm_api_key
        if self._litellm_api_base and "api_base" not in merged_args:
            merged_args["api_base"] = self._litellm_api_base
        if self._litellm_api_version and "api_version" not in merged_args:
            merged_args["api_version"] = self._litellm_api_version
        for key, value in self._litellm_extra_kwargs.items():
            merged_args.setdefault(key, value)

        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            try:
                return await litellm.acompletion(**merged_args)
            except Exception as e:
                if not _is_retryable_litellm_error(e) or attempt >= max_attempts:
                    error_message = (
                        f"[call uuid: {generate_response_call_uuid}] "
                        "Giving up generating response "
                        f"after failed attempt {attempt} "
                        f"due to {type(e).__name__}: {e}"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from e

                logger.info(
                    "[call uuid: %s] "
                    "Retrying generating response in %d seconds "
                    "after failed attempt %d due to retryable %s...",
                    generate_response_call_uuid,
                    sleep_seconds,
                    attempt,
                    type(e).__name__,
                )
                await asyncio.sleep(sleep_seconds)
                sleep_seconds = min(sleep_seconds * 2, self._max_retry_interval_seconds)

        raise RuntimeError("unreachable")


def _is_retryable_litellm_error(exc: BaseException) -> bool:
    """Return True for transient LiteLLM errors worth retrying.

    LiteLLM raises subclasses of `litellm.exceptions.*` that mirror the
    OpenAI taxonomy (RateLimitError, APITimeoutError, APIConnectionError,
    InternalServerError). Lookup by name keeps this module import-cheap
    when LiteLLM isn't installed.
    """
    retryable_names = {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "Timeout",
        "ServiceUnavailableError",
    }
    return type(exc).__name__ in retryable_names
