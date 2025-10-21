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
                from .amazon_bedrock_reranker import (
                    AmazonBedrockReranker,
                    AmazonBedrockRerankerConfig,
                )

                return AmazonBedrockReranker(AmazonBedrockRerankerConfig(**config))
            case "bm25":
                from .bm25_reranker import BM25Reranker

                return BM25Reranker(config)
            case "cross-encoder":
                try:
                    from .cross_encoder_reranker import CrossEncoderReranker
                except ImportError as e:
                    raise ValueError(
                        "sentence-transformers is required "
                        "for CrossEncoderReranker. "
                        "Please install it with "
                        "`pip install sentence-transformers`, "
                        "or by including GPU dependencies with "
                        "`pip install memmachine[gpu]`."
                    ) from e

                return CrossEncoderReranker(config)
            case "embedder":
                from .embedder_reranker import EmbedderReranker

                return EmbedderReranker(
                    {
                        "embedder": injections[config["embedder_id"]],
                    }
                )
            case "identity":
                from .identity_reranker import IdentityReranker

                return IdentityReranker()
            case "rrf-hybrid":
                from .rrf_hybrid_reranker import RRFHybridReranker

                return RRFHybridReranker(
                    {
                        "rerankers": [
                            injections[reranker_id]
                            for reranker_id in config["reranker_ids"]
                        ],
                        "k": config.get("k", 60),
                    }
                )
            case _:
                raise ValueError(f"Unknown Reranker name: {name}")
