"""Benchmark configuration for answer model and evaluation model.

Mirrors the MemMachine server ``configuration.yml`` language model format
but only exposes the two models used by the evaluation harness:

* **answer_model** -- generates final answers from retrieved memories.
* **evaluation_model** -- the LLM judge that scores correctness.

Supported providers (same as ``resources.language_models`` in the server):

* ``openai-responses`` -- OpenAI Responses API.
* ``openai-chat-completions`` -- OpenAI Chat Completions API (also covers
  Ollama and any OpenAI-compatible endpoint).
* ``amazon-bedrock`` -- AWS Bedrock Converse API.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment-variable substitution
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _resolve_env_vars(value: str) -> str:
    """Replace ``${VAR}`` / ``$VAR`` references with environment values."""

    def _replacer(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, match.group(0))

    return _ENV_RE.sub(_replacer, value)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Provider-agnostic model descriptor."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    # AWS / Bedrock fields
    region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    model_id: str | None = None


@dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration."""

    answer_model: ModelConfig
    evaluation_model: ModelConfig


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_model_section(section: dict[str, Any]) -> ModelConfig:
    provider = section["provider"]
    raw_config: dict[str, Any] = section.get("config", {})

    # Resolve env-var references in string values.
    config: dict[str, Any] = {}
    for key, value in raw_config.items():
        config[key] = _resolve_env_vars(value) if isinstance(value, str) else value

    if provider in ("openai-responses", "openai-chat-completions"):
        return ModelConfig(
            provider=provider,
            model=config.get("model", "gpt-5-mini"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
        )
    if provider == "amazon-bedrock":
        return ModelConfig(
            provider=provider,
            model=config.get("model_id", ""),
            model_id=config.get("model_id"),
            region=config.get("region", "us-east-1"),
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
            aws_session_token=config.get("aws_session_token"),
        )
    raise ValueError(f"Unknown provider: {provider}")


def load_benchmark_config(config_path: str | Path) -> BenchmarkConfig:
    """Load a ``benchmark_config.yml`` file and return a typed config."""
    import yaml  # deferred so the module can be imported without pyyaml

    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    return BenchmarkConfig(
        answer_model=_parse_model_section(raw["answer_model"]),
        evaluation_model=_parse_model_section(raw["evaluation_model"]),
    )


def default_benchmark_config() -> BenchmarkConfig:
    """Return a config that mirrors the previous hard-coded defaults.

    Uses ``OPENAI_API_KEY`` from the environment and the OpenAI Responses API.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    default = ModelConfig(
        provider="openai-responses",
        model="gpt-5-mini",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
    )
    return BenchmarkConfig(answer_model=default, evaluation_model=default)


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------


@dataclass
class GenerateResult:
    """Return value from :class:`LLMClient` generation methods."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Unified LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """Provider-agnostic LLM client with sync and async generation.

    Wraps the three providers supported by the MemMachine server config:

    * ``openai-responses`` -- ``client.responses.create``
    * ``openai-chat-completions`` -- ``client.chat.completions.create``
      (also covers **Ollama** and any OpenAI-compatible endpoint)
    * ``amazon-bedrock`` -- ``boto3 bedrock-runtime converse``
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.provider = config.provider
        self.model_name = config.model

        if config.provider in ("openai-responses", "openai-chat-completions"):
            import openai as _openai

            client_kwargs: dict[str, Any] = {}
            if config.api_key:
                client_kwargs["api_key"] = config.api_key
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            self._async_client = _openai.AsyncOpenAI(**client_kwargs)
            self._sync_client = _openai.OpenAI(**client_kwargs)

        elif config.provider == "amazon-bedrock":
            import boto3

            bedrock_kwargs: dict[str, Any] = {
                "region_name": config.region or "us-east-1",
            }
            if config.aws_access_key_id:
                bedrock_kwargs["aws_access_key_id"] = config.aws_access_key_id
            if config.aws_secret_access_key:
                bedrock_kwargs["aws_secret_access_key"] = config.aws_secret_access_key
            if config.aws_session_token:
                bedrock_kwargs["aws_session_token"] = config.aws_session_token
            self._bedrock_client = boto3.client("bedrock-runtime", **bedrock_kwargs)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    # -- async ----------------------------------------------------------

    async def agenerate(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 4096,
        top_p: float = 1,
        json_mode: bool = False,
    ) -> GenerateResult:
        """Async text generation."""
        if self.provider == "openai-responses":
            return await self._agenerate_openai_responses(
                prompt, max_output_tokens, top_p, json_mode
            )
        if self.provider == "openai-chat-completions":
            return await self._agenerate_chat_completions(
                prompt, max_output_tokens, top_p, json_mode
            )
        if self.provider == "amazon-bedrock":
            import asyncio

            return await asyncio.to_thread(
                self._bedrock_generate, prompt, max_output_tokens, top_p
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    # -- sync -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 4096,
        top_p: float = 1,
        json_mode: bool = False,
    ) -> GenerateResult:
        """Synchronous text generation (used by the LLM judge)."""
        if self.provider == "openai-responses":
            return self._generate_openai_responses(
                prompt, max_output_tokens, top_p, json_mode
            )
        if self.provider == "openai-chat-completions":
            return self._generate_chat_completions(
                prompt, max_output_tokens, top_p, json_mode
            )
        if self.provider == "amazon-bedrock":
            return self._bedrock_generate(prompt, max_output_tokens, top_p)
        raise ValueError(f"Unsupported provider: {self.provider}")

    # -- OpenAI Responses -----------------------------------------------

    async def _agenerate_openai_responses(
        self,
        prompt: str,
        max_output_tokens: int,
        top_p: float,
        json_mode: bool,
    ) -> GenerateResult:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "input": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        rsp = await self._async_client.responses.create(**kwargs)
        usage = getattr(rsp, "usage", None)
        return GenerateResult(
            text=rsp.output_text or "",
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

    def _generate_openai_responses(
        self,
        prompt: str,
        max_output_tokens: int,
        top_p: float,
        json_mode: bool,
    ) -> GenerateResult:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "input": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        rsp = self._sync_client.responses.create(**kwargs)
        usage = getattr(rsp, "usage", None)
        return GenerateResult(
            text=rsp.output_text or "",
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

    # -- OpenAI Chat Completions ----------------------------------------

    async def _agenerate_chat_completions(
        self,
        prompt: str,
        max_output_tokens: int,
        top_p: float,
        json_mode: bool,
    ) -> GenerateResult:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_output_tokens,
            "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        rsp = await self._async_client.chat.completions.create(**kwargs)
        usage = rsp.usage
        return GenerateResult(
            text=rsp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def _generate_chat_completions(
        self,
        prompt: str,
        max_output_tokens: int,
        top_p: float,
        json_mode: bool,
    ) -> GenerateResult:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_output_tokens,
            "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        rsp = self._sync_client.chat.completions.create(**kwargs)
        usage = rsp.usage
        return GenerateResult(
            text=rsp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    # -- Amazon Bedrock -------------------------------------------------

    def _bedrock_generate(
        self,
        prompt: str,
        max_output_tokens: int,
        top_p: float,
    ) -> GenerateResult:
        response = self._bedrock_client.converse(
            modelId=self._config.model_id,
            messages=[
                {"role": "user", "content": [{"text": prompt}]},
            ],
            inferenceConfig={
                "maxTokens": max_output_tokens,
                "topP": top_p,
            },
        )
        output_msg = response.get("output", {}).get("message", {})
        content = output_msg.get("content", [])
        text = content[0].get("text", "") if content else ""
        usage = response.get("usage", {})
        return GenerateResult(
            text=text,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )
