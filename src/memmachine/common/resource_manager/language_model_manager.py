"""Builder for LanguageModel instances."""

import asyncio
from asyncio import Lock

from pydantic import SecretStr

from memmachine.common.configuration.language_model_conf import LanguageModelsConf
from memmachine.common.language_model.language_model import LanguageModel


class LanguageModelManager:
    """Create and cache configured language model instances."""

    def __init__(self, conf: LanguageModelsConf) -> None:
        """Store configuration and initialize caches."""
        self._lock = Lock()
        self._language_models_lock: dict[str, Lock] = {}

        self.conf = conf
        self._language_models: dict[str, LanguageModel] = {}

    async def build_all(self) -> dict[str, LanguageModel]:
        """Build all configured language models and return the cache."""
        names = set()
        for name in self.conf.openai_responses_language_model_confs:
            names.add(name)
        for name in self.conf.amazon_bedrock_language_model_confs:
            names.add(name)
        for name in self.conf.openai_chat_completions_language_model_confs:
            names.add(name)

        await asyncio.gather(*[self.get_language_model(name) for name in names])

        return self._language_models

    async def get_language_model(self, name: str) -> LanguageModel:
        """Return a named language model, building it on first access."""
        if name in self._language_models:
            return self._language_models[name]

        if name not in self._language_models_lock:
            async with self._lock:
                self._language_models_lock.setdefault(name, Lock())

        async with self._language_models_lock[name]:
            if name in self._language_models:
                return self._language_models[name]

            llm_model = self._build_language_model(name)
            self._language_models[name] = llm_model
            return llm_model

    def _build_language_model(self, name: str) -> LanguageModel:
        """Construct a language model based on provider."""
        if name in self.conf.openai_responses_language_model_confs:
            return self._build_openai_responses_language_model(name)
        if name in self.conf.openai_chat_completions_language_model_confs:
            return self._build_openai_chat_completions_language_model(name)
        if name in self.conf.amazon_bedrock_language_model_confs:
            return self._build_amazon_bedrock_language_model(name)
        raise ValueError(f"Language model with name {name} not found.")

    def _build_openai_responses_language_model(self, name: str) -> LanguageModel:
        import openai

        from memmachine.common.language_model.openai_responses_language_model import (
            OpenAIResponsesLanguageModel,
            OpenAIResponsesLanguageModelParams,
        )

        conf = self.conf.openai_responses_language_model_confs[name]

        return OpenAIResponsesLanguageModel(
            OpenAIResponsesLanguageModelParams(
                client=openai.AsyncOpenAI(
                    api_key=conf.api_key.get_secret_value(),
                    base_url=conf.base_url,
                ),
                model=conf.model,
                max_retry_interval_seconds=conf.max_retry_interval_seconds,
                metrics_factory=conf.get_metrics_factory(),
                user_metrics_labels=conf.user_metrics_labels,
            ),
        )

    def _build_openai_chat_completions_language_model(self, name: str) -> LanguageModel:
        import openai

        from memmachine.common.language_model.openai_chat_completions_language_model import (
            OpenAIChatCompletionsLanguageModel,
            OpenAIChatCompletionsLanguageModelParams,
        )

        conf = self.conf.openai_chat_completions_language_model_confs[name]

        return OpenAIChatCompletionsLanguageModel(
            OpenAIChatCompletionsLanguageModelParams(
                client=openai.AsyncOpenAI(
                    api_key=conf.api_key.get_secret_value(),
                    base_url=conf.base_url,
                ),
                model=conf.model,
                max_retry_interval_seconds=conf.max_retry_interval_seconds,
                metrics_factory=conf.get_metrics_factory(),
                user_metrics_labels=conf.user_metrics_labels,
            ),
        )

    def _build_amazon_bedrock_language_model(self, name: str) -> LanguageModel:
        import boto3
        from botocore.config import Config

        from memmachine.common.language_model.amazon_bedrock_language_model import (
            AmazonBedrockLanguageModel,
            AmazonBedrockLanguageModelParams,
        )

        conf = self.conf.amazon_bedrock_language_model_confs[name]

        def _get_secret_value(secret: SecretStr | None) -> str | None:
            if secret is None:
                return None
            return secret.get_secret_value()

        client = boto3.client(
            "bedrock-runtime",
            region_name=conf.region,
            aws_access_key_id=_get_secret_value(conf.aws_access_key_id),
            aws_secret_access_key=_get_secret_value(conf.aws_secret_access_key),
            aws_session_token=_get_secret_value(conf.aws_session_token),
            config=Config(
                retries={
                    "total_max_attempts": 1,
                    "mode": "standard",
                },
            ),
        )

        return AmazonBedrockLanguageModel(
            AmazonBedrockLanguageModelParams(
                client=client,
                model_id=conf.model_id,
                inference_config=conf.inference_config,
                additional_model_request_fields=conf.additional_model_request_fields,
                max_retry_interval_seconds=conf.max_retry_interval_seconds,
                metrics_factory=conf.get_metrics_factory(),
                user_metrics_labels=conf.user_metrics_labels,
            ),
        )
