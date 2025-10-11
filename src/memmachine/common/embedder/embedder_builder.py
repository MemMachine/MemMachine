"""
Builder for Embedder instances.
"""

import logging
from typing import Any

from memmachine.common.builder import Builder
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .embedder import Embedder


class EmbedderBuilder(Builder):
    """
    Builder for Embedder instances.
    """

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
        logging.info(f"Building Embedder with name: {name} and config: {config}")
        model = config.get("model_name", "text-embedding-3-small")
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
        match name:
            case "openai":
                from .openai_embedder import OpenAIEmbedder
                return OpenAIEmbedder(
                    {
                        "model_name": model,
                        "api_key": config["api_key"],
                        "dimensions": config.get("dimensions"),
                        "metrics_factory": injected_metrics_factory,
                        "user_metrics_labels": config.get("user_metrics_labels", {}),
                    }
                )
            case "openai-compatible":
                from .openai_embedder import OpenAIEmbedder
                return OpenAIEmbedder(
                    {
                        "model_name": model,
                        "api_key": config["api_key"],
                        "metrics_factory": injected_metrics_factory,
                        "user_metrics_labels": config.get("user_metrics_labels", {}),
                        "base_url": config.get("base_url"),
                    }
                )
            case _:
                raise ValueError(f"Unknown Embedder name: {name}")
