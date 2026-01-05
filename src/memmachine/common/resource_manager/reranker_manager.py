"""Manager for constructing and caching reranker instances."""

import asyncio
import logging
import re
from asyncio import Lock
from collections import defaultdict
from collections.abc import Callable
from typing import Protocol

import boto3
from pydantic import InstanceOf, SecretStr
from typing_extensions import runtime_checkable

from memmachine.common.configuration.reranker_conf import RerankersConf
from memmachine.common.embedder import Embedder
from memmachine.common.errors import InvalidRerankerError
from memmachine.common.reranker import Reranker

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbedderFactory(Protocol):
    """Protocol for retrieving embedder instances."""

    async def get_embedder(self, name: str) -> Embedder:
        """Return an embedder by name."""
        raise NotImplementedError


class RerankerManager:
    """Create and cache configured rerankers."""

    def __init__(
        self,
        conf: RerankersConf,
        embedder_factory: InstanceOf[EmbedderFactory],
    ) -> None:
        """Store configuration and factories, initializing caches."""
        self.conf = conf
        self._rerankers: dict[str, Reranker] = {}

        self._embedder_factory: EmbedderFactory = embedder_factory
        self._lock: Lock = Lock()
        self._rerankers_lock: dict[str, Lock] = defaultdict(Lock)

    async def build_all(self) -> dict[str, Reranker]:
        """Build all configured rerankers and return the cache."""
        names = [
            name
            for keys in [
                self.conf.bm25.keys(),
                self.conf.cohere.keys(),
                self.conf.cross_encoder.keys(),
                self.conf.amazon_bedrock.keys(),
                self.conf.embedder.keys(),
                self.conf.identity.keys(),
                self.conf.rrf_hybrid.keys(),
            ]
            for name in keys
        ]
        tasks = [self.get_reranker(name) for name in names]
        await asyncio.gather(*tasks)
        return self._rerankers

    @property
    def num_of_rerankers(self) -> int:
        """Return the number of cached rerankers."""
        return len(self._rerankers)

    def has_reranker(self, name: str) -> bool:
        """Check if a reranker with the given name exists."""
        return name in self._rerankers

    async def get_reranker(self, name: str, validate: bool = False) -> Reranker:
        """Return a named reranker, building it on first access."""
        if name in self._rerankers:
            return self._rerankers[name]

        if name not in self._rerankers_lock:
            async with self._lock:
                self._rerankers_lock.setdefault(name, Lock())

        async with self._rerankers_lock[name]:
            if name in self._rerankers:
                return self._rerankers[name]

            reranker = await self._build_reranker(name, validate=validate)
            self._rerankers[name] = reranker
            return reranker

    @staticmethod
    async def _validate_reranker(name: str, reranker: Reranker) -> None:
        """Validate that the reranker is working."""
        try:
            logger.info("Validating reranker %s is working.", name)
            _ = await reranker.rerank("a", ["a", "b"])
            logger.info("Reranker %s is working.", name)
        except Exception as e:
            raise InvalidRerankerError(f"reranker '{name}' is invalid. {e}") from e

    async def _build_reranker(self, name: str, validate: bool = False) -> Reranker:
        """Create a reranker based on provider-specific configuration."""
        ret: Reranker | None = None
        if name in self.conf.bm25:
            ret = await self._build_bm25_reranker(name)
        if name in self.conf.cohere:
            ret = await self._build_cohere_reranker(name)
        if name in self.conf.cross_encoder:
            ret = await self._build_cross_encoder_reranker(name)
        if name in self.conf.amazon_bedrock:
            ret = await self._build_amazon_bedrock_reranker(name)
        if name in self.conf.embedder:
            ret = await self._build_embedder_reranker(name)
        if name in self.conf.identity:
            ret = await self._build_identity_reranker(name)
        if name in self.conf.rrf_hybrid:
            ret = await self._build_rrf_hybrid_reranker(name)
        if ret is None:
            raise InvalidRerankerError(f"Reranker with name {name} not found.")
        if validate:
            await self._validate_reranker(name, ret)
        return ret

    async def _build_bm25_reranker(self, name: str) -> Reranker:
        from memmachine.common.reranker.bm25_reranker import (
            BM25Reranker,
            BM25RerankerParams,
        )

        def get_tokenizer(name: str, language: str) -> Callable[[str], list[str]]:
            if name == "default":
                from nltk import word_tokenize
                from nltk.corpus import stopwords

                stop_words = stopwords.words(language)

                def _default_tokenize(text: str) -> list[str]:
                    """Tokenize text by normalizing and filtering stop words."""
                    alphanumeric_text = re.sub(r"\W+", " ", text)
                    lower_text = alphanumeric_text.lower()
                    words = word_tokenize(lower_text, language)
                    tokens = [word for word in words if word and word not in stop_words]
                    return tokens

                return _default_tokenize
            if name == "simple":
                return lambda text: re.sub(r"\W+", " ", text).lower().split()
            raise ValueError(f"Unknown tokenizer: {name}")

        conf = self.conf.bm25[name]
        self._rerankers[name] = BM25Reranker(
            BM25RerankerParams(
                k1=conf.k1,
                b=conf.b,
                epsilon=conf.epsilon,
                tokenize=get_tokenizer(conf.tokenizer, conf.language),
            ),
        )
        return self._rerankers[name]

    async def _build_cohere_reranker(self, name: str) -> Reranker:
        from cohere import ClientV2

        from memmachine.common.reranker.cohere_reranker import (
            CohereReranker,
            CohereRerankerParams,
        )

        conf = self.conf.cohere[name]

        cohere_api_key = conf.cohere_key.get_secret_value() if conf.cohere_key else None
        client = ClientV2(api_key=cohere_api_key)
        params = CohereRerankerParams(
            client=client,
            model=conf.model,
        )
        self._rerankers[name] = CohereReranker(params)
        return self._rerankers[name]

    async def _build_cross_encoder_reranker(self, name: str) -> Reranker:
        from sentence_transformers import CrossEncoder

        from memmachine.common.reranker.cross_encoder_reranker import (
            CrossEncoderReranker,
            CrossEncoderRerankerParams,
        )

        conf = self.conf.cross_encoder[name]

        cross_encoder = CrossEncoder(conf.model_name)
        self._rerankers[name] = CrossEncoderReranker(
            CrossEncoderRerankerParams(
                cross_encoder=cross_encoder, max_input_length=conf.max_input_length
            ),
        )
        return self._rerankers[name]

    async def _build_amazon_bedrock_reranker(self, name: str) -> Reranker:
        from memmachine.common.reranker.amazon_bedrock_reranker import (
            AmazonBedrockReranker,
            AmazonBedrockRerankerParams,
        )

        conf = self.conf.amazon_bedrock[name]

        def _get_secret_value(secret: SecretStr | None) -> str | None:
            if secret is None:
                return None
            return secret.get_secret_value()

        client = boto3.client(
            "bedrock-agent-runtime",
            region_name=conf.region,
            aws_access_key_id=_get_secret_value(conf.aws_access_key_id),
            aws_secret_access_key=_get_secret_value(conf.aws_secret_access_key),
            aws_session_token=_get_secret_value(conf.aws_session_token),
        )
        params = AmazonBedrockRerankerParams(
            client=client,
            region=conf.region,
            model_id=conf.model_id,
            additional_model_request_fields=conf.additional_model_request_fields,
            metrics_factory=conf.get_metrics_factory(),
            user_metrics_labels=conf.user_metrics_labels,
        )
        self._rerankers[name] = AmazonBedrockReranker(params)
        return self._rerankers[name]

    async def _build_embedder_reranker(self, name: str) -> Reranker:
        from memmachine.common.reranker.embedder_reranker import (
            EmbedderReranker,
            EmbedderRerankerParams,
        )

        conf = self.conf.embedder[name]
        embedder = await self._embedder_factory.get_embedder(conf.embedder_id)
        params = EmbedderRerankerParams(embedder=embedder)
        self._rerankers[name] = EmbedderReranker(params)
        return self._rerankers[name]

    async def _build_identity_reranker(self, name: str) -> Reranker:
        from memmachine.common.reranker.identity_reranker import IdentityReranker

        self._rerankers[name] = IdentityReranker()
        return self._rerankers[name]

    async def _build_rrf_hybrid_reranker(self, name: str) -> Reranker:
        """Build an RRF hybrid reranker by combining existing rerankers."""
        from memmachine.common.reranker.rrf_hybrid_reranker import (
            RRFHybridReranker,
            RRFHybridRerankerParams,
        )

        conf = self.conf.rrf_hybrid[name]
        rerankers = []
        for reranker_id in conf.reranker_ids:
            try:
                reranker = await self.get_reranker(reranker_id)
                rerankers.append(reranker)
            except Exception as e:
                raise ValueError(
                    f"Failed to get reranker with id {reranker_id} for "
                    f"RRFHybridReranker {name}: {e}",
                ) from e
        params = RRFHybridRerankerParams(rerankers=rerankers, k=conf.k)
        self._rerankers[name] = RRFHybridReranker(params)
        return self._rerankers[name]
