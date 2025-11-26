"""Manager for building and caching embedder instances."""

import asyncio
from asyncio import Lock

from memmachine.common.configuration.embedder_conf import EmbeddersConf
from memmachine.common.embedder import Embedder


class EmbedderManager:
    """Create and cache embedders defined in configuration."""

    def __init__(self, conf: EmbeddersConf) -> None:
        """Store embedder configuration and initialize caches."""
        self.conf = conf
        self._embedders: dict[str, Embedder] = {}

        # Lock to protect creation of per-embedder locks
        self._lock = Lock()
        self._embedders_lock: dict[str, Lock] = {}

    async def build_all(self) -> dict[str, Embedder]:
        """Trigger lazy initialization of all embedders concurrently."""
        names = set()
        names.update(self.conf.amazon_bedrock)
        names.update(self.conf.openai)
        names.update(self.conf.sentence_transformer)

        # Lazy initialization happens inside get_embedder
        await asyncio.gather(*[self.get_embedder(name) for name in names])

        return self._embedders

    async def get_embedder(self, name: str) -> Embedder:
        """Return a named embedder, building it on first access."""
        # Return cached if already built
        if name in self._embedders:
            return self._embedders[name]

        # Ensure a lock exists for this embedder
        if name not in self._embedders_lock:
            async with self._lock:
                self._embedders_lock.setdefault(name, Lock())

        async with self._embedders_lock[name]:
            # Double-checked locking
            if name in self._embedders:
                return self._embedders[name]

            # Lazy build happens here
            embedder = self._build_embedder(name)
            self._embedders[name] = embedder
            return embedder

    def _build_embedder(self, name: str) -> Embedder:
        """Construct an embedder based on provider."""
        if name in self.conf.amazon_bedrock:
            return self._build_amazon_bedrock_embedders(name)
        if name in self.conf.openai:
            return self._build_openai_embedders(name)
        if name in self.conf.sentence_transformer:
            return self._build_sentence_transformer_embedders(name)
        raise ValueError(f"Embedder with name {name} not found.")

    def _build_amazon_bedrock_embedders(self, name: str) -> Embedder:
        conf = self.conf.amazon_bedrock[name]

        from botocore.config import Config
        from langchain_aws import BedrockEmbeddings

        from memmachine.common.embedder.amazon_bedrock_embedder import (
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
            max_retry_interval_seconds=conf.max_retry_interval_seconds,
        )
        return AmazonBedrockEmbedder(params)

    def _build_openai_embedders(self, name: str) -> Embedder:
        conf = self.conf.openai[name]

        import openai

        from memmachine.common.embedder.openai_embedder import (
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
            max_retry_interval_seconds=conf.max_retry_interval_seconds,
            metrics_factory=conf.get_metrics_factory(),
            user_metrics_labels=conf.user_metrics_labels,
        )
        return OpenAIEmbedder(params)

    def _build_sentence_transformer_embedders(self, name: str) -> Embedder:
        conf = self.conf.sentence_transformer[name]

        from sentence_transformers import SentenceTransformer

        from memmachine.common.embedder.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderParams,
        )

        model_name = conf.model
        sentence_transformer = SentenceTransformer(model_name)

        params = SentenceTransformerEmbedderParams(
            model_name=model_name,
            sentence_transformer=sentence_transformer,
        )
        return SentenceTransformerEmbedder(params)
