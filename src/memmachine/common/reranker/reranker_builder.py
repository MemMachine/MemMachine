"""
Builder for Reranker instances.
"""

from typing import Any

from memmachine.common.builder import Builder

from .reranker import Reranker


class RerankerBuilder(Builder):
    """
    Builder for Reranker instances.
    """

    _rerankers: dict[str, Any] = {}

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()

        match name:
            case "bm25" | "cross-encoder" | "identity":
                pass
            case "embedder":
                dependency_ids.add(config["embedder_id"])
            case "rrf-hybrid":
                dependency_ids.update(config["reranker_ids"])

        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> Reranker:
        match name:
            case "amazon-bedrock":
                import boto3

                from .amazon_bedrock_reranker import (
                    AmazonBedrockReranker,
                    AmazonBedrockRerankerParams,
                )

                region = config.get("region", "us-west-2")

                client = boto3.client(
                    "bedrock-agent-runtime",
                    region_name=region,
                    aws_access_key_id=config["aws_access_key_id"],
                    aws_secret_access_key=config["aws_secret_access_key"],
                )

                return AmazonBedrockReranker(
                    AmazonBedrockRerankerParams(
                        client=client,
                        region=region,
                        model_id=config["model_id"],
                        additional_model_request_fields=config.get(
                            "additional_model_request_fields", {}
                        ),
                    )
                )
            case "bm25":
                import re

                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize

                from .bm25_reranker import BM25Reranker, BM25RerankerParams

                language = config.get("language", "english")
                stop_words = stopwords.words(language)

                def default_tokenize(text: str) -> list[str]:
                    """
                    Preprocess the input text
                    by removing non-alphanumeric characters,
                    converting to lowercase,
                    word-tokenizing,
                    and removing stop words.

                    Args:
                        text (str): The input text to preprocess.

                    Returns:
                        list[str]: A list of tokens for use in BM25 scoring.
                    """
                    alphanumeric_text = re.sub(r"\W+", " ", text)
                    lower_text = alphanumeric_text.lower()
                    words = word_tokenize(lower_text, language)
                    tokens = [word for word in words if word and word not in stop_words]
                    return tokens

                return BM25Reranker(
                    BM25RerankerParams(
                        tokenize=default_tokenize,
                    )
                )
            case "cross-encoder":
                try:
                    from .cross_encoder_reranker import (
                        CrossEncoderReranker,
                        CrossEncoderRerankerParams,
                    )
                except ImportError as e:
                    raise ValueError(
                        "sentence-transformers is required "
                        "for CrossEncoderReranker. "
                        "Please install it with "
                        "`pip install sentence-transformers`, "
                        "or by including GPU dependencies with "
                        "`pip install memmachine[gpu]`."
                    ) from e

                from sentence_transformers import CrossEncoder

                model_name = config.get("model_name")
                if model_name is None:
                    raise ValueError(
                        "model_name must be provided for CrossEncoderReranker"
                    )
                if not isinstance(model_name, str):
                    raise ValueError("model_name must be a string")

                if model_name not in RerankerBuilder._rerankers:
                    RerankerBuilder._rerankers[model_name] = CrossEncoder(model_name)

                reranker = RerankerBuilder._rerankers[model_name]

                return CrossEncoderReranker(
                    CrossEncoderRerankerParams(cross_encoder=reranker)
                )
            case "embedder":
                from .embedder_reranker import EmbedderReranker, EmbedderRerankerParams

                embedder = injections[config["embedder_id"]]
                return EmbedderReranker(
                    EmbedderRerankerParams(
                        embedder=embedder,
                    )
                )
            case "identity":
                from .identity_reranker import IdentityReranker

                return IdentityReranker()
            case "rrf-hybrid":
                from .rrf_hybrid_reranker import (
                    RRFHybridReranker,
                    RRFHybridRerankerParams,
                )

                rerankers = [
                    injections[reranker_id] for reranker_id in config["reranker_ids"]
                ]
                k = config.get("k", 60)

                return RRFHybridReranker(
                    RRFHybridRerankerParams(
                        rerankers=rerankers,
                        k=k,
                    )
                )
            case _:
                raise ValueError(f"Unknown Reranker name: {name}")
