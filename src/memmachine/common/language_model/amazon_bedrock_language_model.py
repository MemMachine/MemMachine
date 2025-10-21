"""
Amazon Bedrock-based language model implementation.
"""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import boto3
import botocore
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.metrics_factory import MetricsFactory

from .language_model import LanguageModel

logger = logging.getLogger(__name__)


class AmazonBedrockConverseInferenceConfig(BaseModel):
    """
    Inference configuration for Amazon Bedrock Converse API.

    Attributes:
        max_tokens (int | None, optional):
            The maximum number of tokens to allow in the generated response.
            The default value is the maximum allowed value
            for the model that you are using.
        stop_sequences (list[str] | None, optional):
            A list of stop sequences that will stop response generation
            (default: None).
        temperature (float | None, optional):
            What sampling temperature to use, between 0 and 1.
            The default value is the default value
            for the model that you are using, applied when None
            (default: None).
        top_p (float | None, optional):
            The percentage of probability mass to consider for the next token
            (default: None).
    """

    max_tokens: int | None = Field(
        None,
        description=(
            "The maximum number of tokens to allow in the generated response. "
            "The default value is the maximum allowed value "
            "for the model that you are using, applied when None"
            "(default: None)."
        ),
        gt=0,
    )
    stop_sequences: list[str] | None = Field(
        None,
        description=(
            "A list of stop sequences that will stop response generation "
            "(default: None)."
        ),
    )
    temperature: float | None = Field(
        None,
        description=(
            "What sampling temperature to use, between 0 and 1. "
            "The default value is the default value "
            "for the model that you are using, applied when None "
            "(default: None)."
        ),
        ge=0.0,
        le=1.0,
    )
    top_p: float | None = Field(
        None,
        description=(
            "The percentage of probability mass to consider for the next token "
            "(default: None)."
        ),
        ge=0.0,
        le=1.0,
    )


class AmazonBedrockLanguageModelConfig(BaseModel):
    """
    Configuration for AmazonBedrockLanguageModel.

    Attributes:
        region (str):
            AWS region where Bedrock is hosted.
        aws_access_key_id (SecretStr):
            AWS access key ID for authentication.
        aws_secret_access_key (SecretStr):
            AWS secret access key for authentication.
        model_id (str):
            ID of the Bedrock model to use for generation
            (e.g. 'openai.gpt-oss-20b-1:0').
        inference_config (AmazonBedrockConverseInferenceConfig | None, optional):
            Inference configuration for the Bedrock Converse API
            (default: None).
        additional_model_request_fields (dict[str, Any] | None, optional):
            Keys are request fields for the model
            and values are values for those fields
            (default: None).
        max_retry_interval_seconds (int, optional):
            Maximal retry interval in seconds when retrying API calls
            (default: 120).
        metrics_factory (MetricsFactory | None, optional):
            An instance of MetricsFactory
            for collecting usage metrics
            (default: None).
        user_metrics_labels (dict[str, str], optional):
            Labels to attach to the collected metrics
            (default: None).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    region: str = Field(
        "us-west-2",
        description="AWS region where Bedrock is hosted.",
    )
    aws_access_key_id: SecretStr = Field(
        description=("AWS access key ID for authentication."),
    )
    aws_secret_access_key: SecretStr = Field(
        description=("AWS secret access key for authentication."),
    )
    model_id: str = Field(
        description=(
            "ID of the Bedrock model to use for generation "
            "(e.g. 'openai.gpt-oss-20b-1:0')."
        ),
    )
    inference_config: AmazonBedrockConverseInferenceConfig | None = Field(
        None,
        description=(
            "Inference configuration for the Bedrock Converse API (default: None)."
        ),
    )
    additional_model_request_fields: dict[str, Any] | None = Field(
        None,
        description=(
            "Keys are request fields for the model "
            "and values are values for those fields "
            "(default: None)."
        ),
    )
    max_retry_interval_seconds: int = Field(
        120,
        description=(
            "Maximal retry interval in seconds when retrying API calls (default: 120)."
        ),
        gt=0,
    )
    metrics_factory: MetricsFactory | None = Field(
        None,
        description=(
            "An instance of MetricsFactory "
            "for collecting usage metrics "
            "(default: None)."
        ),
    )
    user_metrics_labels: dict[str, str] | None = Field(
        None,
        description="Labels to attach to the collected metrics (default: None).",
    )


class AmazonBedrockLanguageModel(LanguageModel):
    """
    Language model that uses Amazon Bedrock models
    to generate responses based on prompts and tools.
    """

    def __init__(self, config: AmazonBedrockLanguageModelConfig):
        """
        Initialize an AmazonBedrockLanguageModel
        with the provided configuration.
        See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

        Args:
            config (AmazonBedrockLanguageModelConfig):
                Configuration for the language model.
        """
        super().__init__()

        region = config.region
        aws_access_key_id = config.aws_access_key_id
        aws_secret_access_key = config.aws_secret_access_key
        self._model_id = config.model_id

        self._inference_config = (
            {
                key: value
                for key, value in {
                    "maxTokens": config.inference_config.max_tokens,
                    "stopSequences": config.inference_config.stop_sequences,
                    "temperature": config.inference_config.temperature,
                    "topP": config.inference_config.top_p,
                }.items()
                if value is not None
            }
            if config.inference_config is not None
            else None
        )

        self._additional_model_request_fields = config.additional_model_request_fields

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_access_key_id.get_secret_value(),
            aws_secret_access_key=aws_secret_access_key.get_secret_value(),
            config=botocore.config.Config(
                retries={
                    "total_max_attempts": 1,
                    "mode": "standard",
                }
            ),
        )

        self._max_retry_interval_seconds = config.max_retry_interval_seconds

        metrics_factory = config.metrics_factory

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = config.user_metrics_labels or {}
            if not isinstance(self._user_metrics_labels, dict):
                raise TypeError("user_metrics_labels must be a dictionary")
            label_names = self._user_metrics_labels.keys()

            self._input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_amazon_bedrock_usage_input_tokens",
                "Number of input tokens used for Amazon Bedrock language model",
                label_names=label_names,
            )
            self._output_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_amazon_bedrock_usage_output_tokens",
                "Number of output tokens used for Amazon Bedrock language model",
                label_names=label_names,
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_amazon_bedrock_usage_total_tokens",
                "Number of tokens used for Amazon Bedrock language model",
                label_names=label_names,
            )
            self._cache_read_input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_amazon_bedrock_usage_cache_read_input_tokens",
                "Number of cache read input tokens used for Amazon Bedrock language model",
                label_names=label_names,
            )
            self._cache_write_input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_amazon_bedrock_usage_cache_write_input_tokens",
                "Number of cache write input tokens used for Amazon Bedrock language model",
                label_names=label_names,
            )

            self._latency_summary = metrics_factory.get_summary(
                "language_model_amazon_bedrock_latency_seconds",
                "Latency in seconds for Amazon Bedrock language model requests",
                label_names=label_names,
            )

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        converse_kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "system": [{"text": system_prompt or "."}],
            "messages": [{"role": "user", "content": [{"text": user_prompt or "."}]}],
        }

        if self._inference_config is not None:
            converse_kwargs["inferenceConfig"] = self._inference_config

        if self._additional_model_request_fields is not None:
            converse_kwargs["additionalModelRequestFields"] = (
                self._additional_model_request_fields
            )

        if tools is not None and len(tools) > 0:
            tool_config: dict[str, Any] = {
                "tools": AmazonBedrockLanguageModel._format_tools(tools)
            }
            if tool_choice is not None:
                tool_config["toolChoice"] = (
                    AmazonBedrockLanguageModel._format_tool_choice(tool_choice)
                )
            converse_kwargs["toolConfig"] = tool_config

        generate_response_call_uuid = uuid4()

        start_time = time.monotonic()

        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            logger.debug(
                "[call uuid: %s] Attempting to generate response using %s Amazon Bedrock model: "
                "on attempt %d with max attempts %d",
                generate_response_call_uuid,
                self._model_id,
                attempt,
                max_attempts,
            )

            try:
                response = await asyncio.to_thread(
                    self._client.converse,
                    **converse_kwargs,
                )
                break
            except Exception as e:
                # Exception may be retried.
                if attempt >= max_attempts:
                    error_message = (
                        f"[call uuid: {generate_response_call_uuid}] "
                        "Giving up generating response "
                        f"after failed attempt {attempt} "
                        f"due to assumed retryable {type(e).__name__}: "
                        f"max attempts {max_attempts} reached"
                    )
                    logger.error(error_message)
                    raise ExternalServiceAPIError(error_message)

                logger.info(
                    "[call uuid: %s] "
                    "Retrying generating response in %d seconds "
                    "after failed attempt %d due to assumed retryable %s...",
                    generate_response_call_uuid,
                    sleep_seconds,
                    attempt,
                    type(e).__name__,
                )
                await asyncio.sleep(sleep_seconds)
                sleep_seconds *= 2
                sleep_seconds = min(sleep_seconds, self._max_retry_interval_seconds)
                continue

        end_time = time.monotonic()

        if self._collect_metrics:
            if (response_usage := response.get("usage")) is not None:
                self._input_tokens_usage_counter.increment(
                    value=response_usage.get("inputTokens", 0),
                    labels=self._user_metrics_labels,
                )
                self._output_tokens_usage_counter.increment(
                    value=response_usage.get("outputTokens", 0),
                    labels=self._user_metrics_labels,
                )
                self._total_tokens_usage_counter.increment(
                    value=response_usage.get("totalTokens", 0),
                    labels=self._user_metrics_labels,
                )
                self._cache_read_input_tokens_usage_counter.increment(
                    response_usage.get("cacheReadInputTokens", 0),
                    labels=self._user_metrics_labels,
                )
                self._cache_read_input_tokens_usage_counter.increment(
                    response_usage.get("cacheWriteInputTokens", 0),
                    labels=self._user_metrics_labels,
                )

            self._latency_summary.observe(
                value=end_time - start_time,
                labels=self._user_metrics_labels,
            )

        text_block_strings = []
        function_calls_arguments = []

        content_blocks = response["output"]["message"]["content"]
        for content_block in content_blocks:
            if "text" in content_block:
                text_block = content_block["text"]
                text_block_strings.append(text_block)

            elif "toolUse" in content_block:
                tool_use_block = content_block["toolUse"]
                function_calls_arguments.append(
                    {
                        "call_id": tool_use_block["toolUseId"],
                        "function": {
                            "name": tool_use_block["name"],
                            "arguments": tool_use_block["input"],
                        },
                    }
                )
            else:
                logger.info(
                    "[call uuid: %s] "
                    "Ignoring unsupported content block type in response: "
                    "Received block with keys %s",
                    generate_response_call_uuid,
                    list(content_block.keys()),
                )

        # This approach is similar to how OpenAI handles multiple text blocks.
        output_text = "\n".join(text_block_strings)

        return (
            output_text,
            function_calls_arguments,
        )

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
        bedrock_tools = []
        for tool in tools:
            if "toolSpec" in tool:
                # Assume tool already in correct format.
                bedrock_tools.append(tool)
            else:
                # Convert from OpenAI format.
                bedrock_tools.append(
                    {
                        "toolSpec": {
                            "name": tool["name"],
                            "description": tool.get("description") or tool["name"],
                            "inputSchema": {"json": tool["parameters"]}
                            if "parameters" in tool
                            else {},
                        }
                    }
                )

        return bedrock_tools

    @staticmethod
    def _format_tool_choice(
        tool_choice: str | dict[str, str],
    ) -> dict[str, dict[str, str]]:
        if isinstance(tool_choice, dict):
            # Convert from OpenAI format.
            if tool_choice.get("type") == "function" and "name" in tool_choice:
                return {"tool": {"name": tool_choice["name"]}}
            else:
                raise ValueError(
                    "Tool choice must be in OpenAI format "
                    "with 'type' field equal to 'function' and 'name' specified",
                )

        # tool_choice should be a string here.
        match tool_choice:
            case "any" | "required":
                return {"any": {}}
            case "auto":
                return {"auto": {}}
            case _:
                return {"tool": {"name": tool_choice}}
