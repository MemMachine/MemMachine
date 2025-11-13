from sentence_transformers import SentenceTransformer

from memmachine.common.configuration.embedder_conf import EmbedderConf
from memmachine.common.embedder import Embedder


class EmbedderMgr:
    def __init__(self, conf: EmbedderConf):
        self.conf = conf
        self._embedders: dict[str, Embedder] = {}
        self._sentence_transformers: dict[str, SentenceTransformer] = {}

    def build_all(self) -> dict[str, Embedder]:
        self._build_openai_embedders()
        self._build_amazon_bedrock_embedders()
        self._build_sentence_transformer_embedders()
        return self.embedders

    @property
    def embedders(self) -> dict[str, Embedder]:
        return self._embedders

    def get_embedder(self, name: str) -> Embedder:
        if name not in self._embedders:
            raise ValueError(f"Embedder '{name}' not found.")
        return self._embedders[name]

    def _build_amazon_bedrock_embedders(self):
        for name, conf in self.conf.amazon_bedrock.items():
            import botocore
            from langchain_aws import BedrockEmbeddings

            from ..embedder.amazon_bedrock_embedder import (
                AmazonBedrockEmbedder,
                AmazonBedrockEmbedderParams,
            )

            client = BedrockEmbeddings(
                region_name=conf.region,
                aws_access_key_id=conf.aws_access_key_id,
                aws_secret_access_key=conf.aws_secret_access_key,
                aws_session_token=conf.aws_session_token,
                model_id=conf.model_id,
                config=botocore.config.Config(
                    retries={
                        "total_max_attempts": 1,
                        "mode": "standard",
                    }
                ),
            )
            params = AmazonBedrockEmbedderParams(
                client=client,
                model_id=conf.model_id,
                similarity_metric=conf.similarity_metric,
                max_retry_interval_seconds=conf.max_retry_interval_seconds,
            )
            self._embedders[name] = AmazonBedrockEmbedder(params)

    def _build_openai_embedders(self):
        for name, conf in self.conf.openai.items():
            import openai

            from ..embedder.openai_embedder import OpenAIEmbedder, OpenAIEmbedderParams

            params = OpenAIEmbedderParams(
                client=openai.AsyncOpenAI(
                    api_key=conf.api_key,
                    base_url=conf.base_url,
                ),
                model=conf.model,
                dimensions=conf.dimensions,
                max_retry_interval_seconds=conf.max_retry_interval_seconds,
                metrics_factory=conf.get_metrics_factory(),
                user_metrics_labels=conf.user_metrics_labels,
            )
            self._embedders[name] = OpenAIEmbedder(params)

    def _build_sentence_transformer_embedders(self):
        for name, conf in self.conf.sentence_transformer.items():
            from ..embedder.sentence_transformer_embedder import (
                SentenceTransformerEmbedder,
                SentenceTransformerEmbedderParams,
            )

            model_name = conf.model
            if model_name not in self._sentence_transformers:
                self._sentence_transformers[model_name] = SentenceTransformer(
                    model_name
                )
            params = SentenceTransformerEmbedderParams(
                model_name=model_name,
                sentence_transformer=self._sentence_transformers[model_name],
            )
            self._embedders[name] = SentenceTransformerEmbedder(params)
