"""Builder for LanguageModel instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import SecretStr

from memmachine.common.configuration.language_model_conf import LanguageModelsConf
from memmachine.common.errors import InvalidLanguageModelError
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.resource_manager.base_manager import BaseResourceManager

if TYPE_CHECKING:
    from memmachine.common.configuration.language_model_conf import (
        AmazonBedrockLanguageModelConf,
        OpenAIChatCompletionsLanguageModelConf,
        OpenAIResponsesLanguageModelConf,
    )

logger = logging.getLogger(__name__)


class LanguageModelManager(BaseResourceManager[LanguageModel]):
    """Create and cache configured language model instances."""

    def __init__(self, conf: LanguageModelsConf) -> None:
        """Store configuration and initialize caches."""
        super().__init__()
        self.conf = conf
        # Alias for backward compatibility
        self._language_models = self._resources

    @property
    def _resource_type_name(self) -> str:
        return "language model"

    def _is_configured(self, name: str) -> bool:
        """Check if a language model is configured."""
        return (
            name in self.conf.openai_responses_language_model_confs
            or name in self.conf.openai_chat_completions_language_model_confs
            or name in self.conf.amazon_bedrock_language_model_confs
        )

    def _get_not_found_error(self, name: str) -> Exception:
        """Return InvalidLanguageModelError for unknown language models."""
        return InvalidLanguageModelError(f"Language model with name {name} not found.")

    def get_all_names(self) -> set[str]:
        """Return all configured language model names."""
        names = set()
        names.update(self.conf.openai_responses_language_model_confs)
        names.update(self.conf.openai_chat_completions_language_model_confs)
        names.update(self.conf.amazon_bedrock_language_model_confs)
        return names

    async def build_all(self) -> dict[str, LanguageModel]:
        """Build all configured language models and return the cache."""
        return await self.build_all_with_error_tracking(self.get_all_names())

    async def get_language_model(
        self, name: str, validate: bool = False
    ) -> LanguageModel:
        """Return a named language model, building it on first access."""
        return await self._get_resource_with_locking(name, validate=validate)

    async def _build_resource(self, name: str, validate: bool = False) -> LanguageModel:
        """Build a language model by name."""
        return await self._build_language_model(name, validate=validate)

    def remove_language_model(self, name: str) -> bool:
        """
        Remove a language model from the manager.

        Returns True if the model was removed, False if it wasn't found.
        """
        removed = self._remove_from_cache(name)
        # Also remove from config
        if name in self.conf.openai_responses_language_model_confs:
            del self.conf.openai_responses_language_model_confs[name]
            removed = True
        if name in self.conf.openai_chat_completions_language_model_confs:
            del self.conf.openai_chat_completions_language_model_confs[name]
            removed = True
        if name in self.conf.amazon_bedrock_language_model_confs:
            del self.conf.amazon_bedrock_language_model_confs[name]
            removed = True
        return removed

    def add_language_model_config(
        self,
        name: str,
        provider: str,
        config: OpenAIResponsesLanguageModelConf
        | OpenAIChatCompletionsLanguageModelConf
        | AmazonBedrockLanguageModelConf,
    ) -> None:
        """
        Add a new language model configuration at runtime.

        Args:
            name: The name/id for the language model.
            provider: The provider type ('openai-responses', 'openai-chat-completions', 'amazon-bedrock').
            config: The provider-specific configuration object.

        """
        # Clear any previous errors for this name
        self.clear_build_error(name)

        if provider == "openai-responses":
            from memmachine.common.configuration.language_model_conf import (
                OpenAIResponsesLanguageModelConf,
            )

            if not isinstance(config, OpenAIResponsesLanguageModelConf):
                raise ValueError(
                    "Expected OpenAIResponsesLanguageModelConf for provider 'openai-responses'"
                )
            self.conf.openai_responses_language_model_confs[name] = config
        elif provider == "openai-chat-completions":
            from memmachine.common.configuration.language_model_conf import (
                OpenAIChatCompletionsLanguageModelConf,
            )

            if not isinstance(config, OpenAIChatCompletionsLanguageModelConf):
                raise ValueError(
                    "Expected OpenAIChatCompletionsLanguageModelConf for provider 'openai-chat-completions'"
                )
            self.conf.openai_chat_completions_language_model_confs[name] = config
        elif provider == "amazon-bedrock":
            from memmachine.common.configuration.language_model_conf import (
                AmazonBedrockLanguageModelConf,
            )

            if not isinstance(config, AmazonBedrockLanguageModelConf):
                raise ValueError(
                    "Expected AmazonBedrockLanguageModelConf for provider 'amazon-bedrock'"
                )
            self.conf.amazon_bedrock_language_model_confs[name] = config
        else:
            raise ValueError(f"Unknown language model provider: {provider}")

    @staticmethod
    async def _validate_language_model(
        name: str, language_model: LanguageModel
    ) -> None:
        """Validate that the language model is working."""
        try:
            logger.info("Validating language model '%s' ...", name)
            _ = await language_model.generate_response(
                system_prompt="a",
                user_prompt="b",
            )
            logger.info("Language model '%s' is valid.", name)
        except Exception as e:
            raise InvalidLanguageModelError(
                f"language model '{name}' is invalid. {e}"
            ) from e

    async def _build_language_model(
        self, name: str, validate: bool = False
    ) -> LanguageModel:
        """Construct a language model based on provider."""
        ret: LanguageModel | None = None
        if name in self.conf.openai_responses_language_model_confs:
            ret = self._build_openai_responses_language_model(name)
        if name in self.conf.openai_chat_completions_language_model_confs:
            ret = self._build_openai_chat_completions_language_model(name)
        if name in self.conf.amazon_bedrock_language_model_confs:
            ret = self._build_amazon_bedrock_language_model(name)
        if ret is None:
            raise InvalidLanguageModelError(
                f"Language model with name {name} not found."
            )
        if validate:
            await self._validate_language_model(name, ret)
        return ret

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
