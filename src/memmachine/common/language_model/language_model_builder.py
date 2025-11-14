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
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()

        match name:
            case "openai":
                if "metrics_factory_id" in config:
                    dependency_ids.add(config["metrics_factory_id"])

        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
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

        match name:
            case "openai":
                import openai

                from .openai_responses_language_model import (
                    OpenAIResponsesLanguageModel,
                    OpenAIResponsesLanguageModelParams,
                )

                return OpenAIResponsesLanguageModel(
                    OpenAIResponsesLanguageModelParams(
                        client=openai.AsyncOpenAI(
                            api_key=config["api_key"],
                            base_url=config.get("base_url"),
                            max_retries=0,
                        ),
                        model=config["model"],
                        max_retry_interval_seconds=config.get(
                            "max_retry_interval_seconds", 120
                        ),
                        metrics_factory=get_metrics_factory(config),
                        user_metrics_labels=config.get("user_metrics_labels", {}),
                    )
                )

            case "vllm" | "sglang" | "openai-compatible":
                import openai

                from .openai_chat_completions_language_model import (
                    OpenAIChatCompletionsLanguageModel,
                    OpenAIChatCompletionsLanguageModelParams,
                )

                return OpenAIChatCompletionsLanguageModel(
                    OpenAIChatCompletionsLanguageModelParams(
                        client=openai.AsyncOpenAI(
                            api_key=config["api_key"],
                            base_url=config.get("base_url"),
                            max_retries=0,
                        ),
                        model=config["model"],
                        max_retry_interval_seconds=config.get(
                            "max_retry_interval_seconds", 120
                        ),
                        metrics_factory=get_metrics_factory(config),
                        user_metrics_labels=config.get("user_metrics_labels", {}),
                    )
                )

            case "amazon-bedrock":
                import boto3
                import botocore

                from .amazon_bedrock_language_model import (
                    AmazonBedrockConverseInferenceConfig,
                    AmazonBedrockLanguageModel,
                    AmazonBedrockLanguageModelParams,
                )

                region = config.get("region", "us-west-2")

                client = boto3.client(
                    "bedrock-runtime",
                    region_name=region,
                    aws_access_key_id=config.get("aws_access_key_id"),
                    aws_secret_access_key=config.get("aws_secret_access_key"),
                    aws_session_token=config.get("aws_session_token"),
                    config=botocore.config.Config(
                        retries={
                            "total_max_attempts": 1,
                            "mode": "standard",
                        }
                    ),
                )

                return AmazonBedrockLanguageModel(
                    AmazonBedrockLanguageModelParams(
                        client=client,
                        model_id=config["model_id"],
                        inference_config=AmazonBedrockConverseInferenceConfig(
                            **config.get("inference_config", {})
                        ),
                        additional_model_request_fields=config.get(
                            "additional_model_request_fields"
                        ),
                        max_retry_interval_seconds=config.get(
                            "max_retry_interval_seconds", 120
                        ),
                        metrics_factory=get_metrics_factory(config),
                        user_metrics_labels=config.get("user_metrics_labels", {}),
                    )
                )

            case _:
                raise ValueError(f"Unknown LanguageModel name: {name}")
