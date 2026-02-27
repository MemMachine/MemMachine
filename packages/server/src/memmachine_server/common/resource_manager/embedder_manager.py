"""Manager for building and caching embedder instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memmachine_server.common.embedder import Embedder
from memmachine_server.common.errors import InvalidEmbedderError
from memmachine_server.common.resource_manager.base_manager import BaseResourceManager

if TYPE_CHECKING:
    from memmachine_server.common.configuration.embedder_conf import (
        AmazonBedrockEmbedderConf,
        EmbeddersConf,
        OpenAIEmbedderConf,
        SentenceTransformerEmbedderConf,
    )

logger = logging.getLogger(__name__)


class EmbedderManager(BaseResourceManager[Embedder]):
    """Create and cache embedders defined in configuration."""

    def __init__(self, conf: EmbeddersConf) -> None:
        """Store embedder configuration and initialize caches."""
        super().__init__()
        self.conf = conf
        # Alias for backward compatibility
        self._embedders = self._resources

    @property
    def _resource_type_name(self) -> str:
        return "embedder"

    def _is_configured(self, name: str) -> bool:
        """Check if an embedder is configured."""
        return self.conf.contains_embedder(name)

    def _get_not_found_error(self, name: str) -> Exception:
        """Return InvalidEmbedderError for unknown embedders."""
        return InvalidEmbedderError(name)

    def get_all_names(self) -> set[str]:
        """Return all configured embedder names."""
        names = set()
        names.update(self.conf.amazon_bedrock)
        names.update(self.conf.openai)
        names.update(self.conf.sentence_transformer)
        return names

    async def build_all(self) -> dict[str, Embedder]:
        """Trigger lazy initialization of all embedders concurrently."""
        return await self.build_all_with_error_tracking(self.get_all_names())

    async def get_embedder(self, name: str, validate: bool = False) -> Embedder:
        """Return a named embedder, building it on first access."""
        return await self._get_resource_with_locking(name, validate=validate)

    async def _build_resource(self, name: str, validate: bool = False) -> Embedder:
        """Build an embedder by name."""
        return await self._build_embedder(name, validate=validate)

    def remove_embedder(self, name: str) -> bool:
        """
        Remove an embedder from the manager.

        Returns True if the embedder was removed, False if it wasn't found.
        """
        removed = self._remove_from_cache(name)
        # Also remove from config
        if name in self.conf.openai:
            del self.conf.openai[name]
            removed = True
        if name in self.conf.amazon_bedrock:
            del self.conf.amazon_bedrock[name]
            removed = True
        if name in self.conf.sentence_transformer:
            del self.conf.sentence_transformer[name]
            removed = True
        return removed

    def add_embedder_config(
        self,
        name: str,
        provider: str,
        config: AmazonBedrockEmbedderConf
        | OpenAIEmbedderConf
        | SentenceTransformerEmbedderConf,
    ) -> None:
        """
        Add a new embedder configuration at runtime.

        Args:
            name: The name/id for the embedder.
            provider: The provider type ('openai', 'amazon-bedrock', 'sentence-transformer').
            config: The provider-specific configuration object.

        """
        # Clear any previous errors for this name
        self.clear_build_error(name)

        if provider == "openai":
            from memmachine_server.common.configuration.embedder_conf import (
                OpenAIEmbedderConf,
            )

            if not isinstance(config, OpenAIEmbedderConf):
                raise ValueError("Expected OpenAIEmbedderConf for provider 'openai'")
            self.conf.openai[name] = config
        elif provider == "amazon-bedrock":
            from memmachine_server.common.configuration.embedder_conf import (
                AmazonBedrockEmbedderConf,
            )

            if not isinstance(config, AmazonBedrockEmbedderConf):
                raise ValueError(
                    "Expected AmazonBedrockEmbedderConf for provider 'amazon-bedrock'"
                )
            self.conf.amazon_bedrock[name] = config
        elif provider == "sentence-transformer":
            from memmachine_server.common.configuration.embedder_conf import (
                SentenceTransformerEmbedderConf,
            )

            if not isinstance(config, SentenceTransformerEmbedderConf):
                raise ValueError(
                    "Expected SentenceTransformerEmbedderConf for provider 'sentence-transformer'"
                )
            self.conf.sentence_transformer[name] = config
        else:
            raise ValueError(f"Unknown embedder provider: {provider}")

    @staticmethod
    async def _validate_embedder(name: str, embedder: Embedder) -> None:
        """Validate that the embedder is working."""
        try:
            logger.info("Validating embedder '%s' is working.", name)
            _ = await embedder.search_embed(["a"])
            logger.info("Embedder '%s' is valid.", name)
        except Exception as e:
            raise InvalidEmbedderError(f"embedder '{name}' is invalid. {e}") from e

    async def _build_embedder(self, name: str, validate: bool) -> Embedder:
        """Construct an embedder based on provider."""
        ret: Embedder | None = None
        if name in self.conf.amazon_bedrock:
            ret = self._build_amazon_bedrock_embedders(name)
        if name in self.conf.openai:
            ret = self._build_openai_embedders(name)
        if name in self.conf.sentence_transformer:
            ret = self._build_sentence_transformer_embedders(name)
        if ret is None:
            raise InvalidEmbedderError(f"Embedder with name {name} not found.")
        if validate:
            await self._validate_embedder(name, ret)
        return ret

    def _build_amazon_bedrock_embedders(self, name: str) -> Embedder:
        conf = self.conf.amazon_bedrock[name]

        from botocore.config import Config
        from langchain_aws import BedrockEmbeddings

        from memmachine_server.common.embedder.amazon_bedrock_embedder import (
            AmazonBedrockEmbedder,
            AmazonBedrockEmbedderParams,
        )

        client = BedrockEmbeddings(
            region_name=conf.region,
            aws_access_key_id=conf.aws_access_key_id,
            aws_secret_access_key=conf.aws_secret_access_key,
            aws_session_token=conf.aws_session_token,
            model_id=conf.model_id,
            config=Config(
                retries={
                    "total_max_attempts": 1,
                    "mode": "standard",
                },
            ),
        )
        params = AmazonBedrockEmbedderParams(
            client=client,
            model_id=conf.model_id,
            similarity_metric=conf.similarity_metric,
            max_input_length=conf.max_input_length,
            max_retry_interval_seconds=conf.max_retry_interval_seconds,
        )
        return AmazonBedrockEmbedder(params)

    def _build_openai_embedders(self, name: str) -> Embedder:
        conf = self.conf.openai[name]

        import openai

        from memmachine_server.common.embedder.openai_embedder import (
            OpenAIEmbedder,
            OpenAIEmbedderParams,
        )

        dimensions = conf.dimensions or 1536

        params = OpenAIEmbedderParams(
            client=openai.AsyncOpenAI(
                api_key=conf.api_key.get_secret_value(),
                base_url=conf.base_url,
            ),
            model=conf.model,
            dimensions=dimensions,
            max_input_length=conf.max_input_length,
            max_retry_interval_seconds=conf.max_retry_interval_seconds,
            metrics_factory=conf.get_metrics_factory(),
            user_metrics_labels=conf.user_metrics_labels,
        )
        return OpenAIEmbedder(params)

    def _build_sentence_transformer_embedders(self, name: str) -> Embedder:
        conf = self.conf.sentence_transformer[name]

        from sentence_transformers import SentenceTransformer

        from memmachine_server.common.embedder.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderParams,
        )

        model_name = conf.model
        sentence_transformer = SentenceTransformer(model_name)

        params = SentenceTransformerEmbedderParams(
            model_name=model_name,
            sentence_transformer=sentence_transformer,
            max_input_length=conf.max_input_length,
        )
        return SentenceTransformerEmbedder(params)
