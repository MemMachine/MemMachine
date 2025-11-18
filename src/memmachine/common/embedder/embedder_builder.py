"""
Builder for Embedder instances.
"""

from typing import Any

from memmachine.common.builder import Builder
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .data_types import SimilarityMetric
from .embedder import Embedder


class EmbedderBuilder(Builder):
    """
    Builder for Embedder instances.
    """

    # Do not type hint with SentenceTransformer to avoid importing it unnecessarily
    # Long-term solution is to refactor to use dependency injection
    _embedders: dict[str, Any] = {}

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids: set[str] = set()

        match name:
            case "openai":
                if "metrics_factory_id" in config:
                    dependency_ids.add(config["metrics_factory_id"])

        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> Embedder:
        match name:
            case "amazon-bedrock":
                import botocore
                from langchain_aws import BedrockEmbeddings

                from .amazon_bedrock_embedder import (
                    AmazonBedrockEmbedder,
                    AmazonBedrockEmbedderParams,
                )

                region = config.get("region", "us-west-2")
                aws_access_key_id = config.get("aws_access_key_id")
                aws_secret_access_key = config.get("aws_secret_access_key")
                aws_session_token = config.get("aws_session_token")
                model_id = config["model_id"]

                client = BedrockEmbeddings(
                    region_name=region,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    model_id=model_id,
                    config=botocore.config.Config(
                        retries={
                            "total_max_attempts": 1,
                            "mode": "standard",
                        }
                    ),
                )

                return AmazonBedrockEmbedder(
                    AmazonBedrockEmbedderParams(
                        client=client,
                        model_id=model_id,
                        similarity_metric=SimilarityMetric(
                            config.get("similarity_metric", "cosine")
                        ),
                        max_retry_interval_seconds=config.get(
                            "max_retry_interval_seconds", 120
                        ),
                    )
                )
            case "openai":
                import openai

                from .openai_embedder import OpenAIEmbedder, OpenAIEmbedderParams

                injected_metrics_factory_id = config.get("metrics_factory_id")
                if injected_metrics_factory_id is None:
                    injected_metrics_factory = None
                elif not isinstance(injected_metrics_factory_id, str):
                    raise TypeError("metrics_factory_id must be a string if provided")
                else:
                    injected_metrics_factory = injections.get(
                        injected_metrics_factory_id
                    )
                    if injected_metrics_factory is None:
                        raise ValueError(
                            "MetricsFactory with id "
                            f"{injected_metrics_factory_id} "
                            "not found in injections"
                        )
                    elif not isinstance(injected_metrics_factory, MetricsFactory):
                        raise TypeError(
                            "Injected dependency with id "
                            f"{injected_metrics_factory_id} "
                            "is not a MetricsFactory"
                        )

                return OpenAIEmbedder(
                    OpenAIEmbedderParams(
                        client=openai.AsyncOpenAI(
                            api_key=config["api_key"],
                            base_url=config.get("base_url"),
                        ),
                        model=config.get("model", "text-embedding-3-small"),
                        dimensions=config.get("dimensions", 1536),
                        max_retry_interval_seconds=config.get(
                            "max_retry_interval_seconds", 120
                        ),
                        metrics_factory=injected_metrics_factory,
                        user_metrics_labels=config.get("user_metrics_labels", {}),
                    )
                )
            case "sentence-transformer":
                from sentence_transformers import SentenceTransformer

                from .sentence_transformer_embedder import (
                    SentenceTransformerEmbedder,
                    SentenceTransformerEmbedderParams,
                )

                model_name = config.get("model")
                if model_name is None:
                    raise ValueError(
                        "'model' must be provided for sentence-transformer"
                    )
                if not isinstance(model_name, str):
                    raise TypeError("model_name must be a string")

                if model_name not in EmbedderBuilder._embedders:
                    EmbedderBuilder._embedders["model_name"] = SentenceTransformer(
                        model_name
                    )

                embedder = EmbedderBuilder._embedders["model_name"]

                return SentenceTransformerEmbedder(
                    SentenceTransformerEmbedderParams(
                        model_name=model_name, sentence_transformer=embedder
                    )
                )
            case _:
                raise ValueError(f"Unknown Embedder name: {name}")
