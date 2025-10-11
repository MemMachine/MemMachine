"""
Builder for LanguageModel instances.
"""

from typing import Any

from memmachine.common.builder import Builder
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .language_model import LanguageModel


class LanguageModelBuilder(Builder):
    """
    Builder for LanguageModel instances.
    """

    @staticmethod
    def get_dependency_ids(variant: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()

        match variant:
            case "openai":
                if "metrics_factory_id" in config:
                    dependency_ids.add(config["metrics_factory_id"])

        return dependency_ids

    @staticmethod
    def build(
        variant: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> LanguageModel:
        def get_metrics_factory(config: dict[str, Any]):
            injected_metrics_factory_id = config.get("metrics_factory_id")
            if injected_metrics_factory_id is None:
                injected_metrics_factory = None
            elif not isinstance(injected_metrics_factory_id, str):
                raise TypeError("metrics_factory_id must be a string if provided")
            else:
                injected_metrics_factory = injections.get(injected_metrics_factory_id)
                if injected_metrics_factory is None:
                    raise ValueError(
                        "MetricsFactory with id "
                        f"{injected_metrics_factory_id} "
                        "not found in injections"
                    )
                if not isinstance(injected_metrics_factory, MetricsFactory):
                    raise TypeError(
                        "Injected dependency with id "
                        f"{injected_metrics_factory_id} "
                        "is not a MetricsFactory"
                    )
            return injected_metrics_factory

        match variant:
            case "openai":
                from .openai_language_model import OpenAILanguageModel

                return OpenAILanguageModel(
                    {
                        key: value
                        for key, value in config.items()
                        if key != "metrics_factory_id"
                    }
                    | {
                        "metrics_factory": get_metrics_factory(config),
                    }
                )

            case "vllm" | "sglang" | "openai-compatible":
                from .openai_compatible_language_model import (
                    OpenAICompatibleLanguageModel,
                )

                return OpenAICompatibleLanguageModel(
                    {
                        key: value
                        for key, value in config.items()
                        if key != "metrics_factory_id"
                    }
                    | {
                        "metrics_factory": get_metrics_factory(config),
                    }
                )

            case _:
                raise ValueError(f"Unknown LanguageModel variant: {variant}")
