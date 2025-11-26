"""Amazon Bedrock-based language model implementation."""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import uuid4

import instructor
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.metrics_factory import MetricsFactory

from .language_model import LanguageModel

T = TypeVar("T")

logger = logging.getLogger(__name__)


class AmazonBedrockConverseInferenceConfig(BaseModel):
    """
    Inference configuration for Amazon Bedrock Converse API.

    Attributes:
        max_tokens (int | None):
            The maximum number of tokens to allow in the generated response.
            If None, uses the maximum allowed value
            for the model that you are using
            (default: None).
        stop_sequences (list[str] | None):
            A list of stop sequences that will stop response generation
            (default: None).
        temperature (float | None):
            What sampling temperature to use, between 0 and 1.
            The default value is the default value
            for the model that you are using, applied when None
            (default: None).
        top_p (float | None):
            The percentage of probability mass to consider for the next token
            (default: None).

    """

    max_tokens: int | None = Field(
        None,
        description=(
            "The maximum number of tokens to allow in the generated response. "
            "If None, uses the maximum allowed value "
            "for the model that you are using"
        ),
        gt=0,
    )
    stop_sequences: list[str] | None = Field(
        None,
        description="A list of stop sequences that will stop response generation",
    )
    temperature: float | None = Field(
        None,
        description=(
            "What sampling temperature to use, between 0 and 1. "
            "The default value is the default value "
            "for the model that you are using, applied when None"
        ),
        ge=0.0,
        le=1.0,
    )
    top_p: float | None = Field(
        None,
        description=(
            "The percentage of probability mass to consider for the next token"
        ),
        ge=0.0,
        le=1.0,
    )


class AmazonBedrockLanguageModelParams(BaseModel):
    """
    Parameters for AmazonBedrockLanguageModel.

    Attributes:
        client (Any):
            Boto3 Bedrock Runtime client
            to use for making API calls.
        model_id (str):
            ID of the Bedrock model to use for generation
            (e.g. 'openai.gpt-oss-20b-1:0').
        inference_config (AmazonBedrockConverseInferenceConfig | None):
            Inference configuration for the Bedrock Converse API
            (default: None).
        additional_model_request_fields (dict[str, Any] | None):
            Keys are request fields for the model
            and values are values for those fields
            (default: None).
        max_retry_interval_seconds (int):
            Maximal retry interval in seconds when retrying API calls
            (default: 120).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory
            for collecting usage metrics
            (default: None).
        user_metrics_labels (dict[str, str]):
            Labels to attach to the collected metrics
            (default: {}).

    """

    client: Any = Field(
        ...,
        description=(
            "Boto3 Agents for Amazon Bedrock Runtime client to use for making API calls"
        ),
    )

    model_id: str = Field(
        ...,
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
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description=(
            "An instance of MetricsFactory "
            "for collecting usage metrics "
            "(default: None)."
        ),
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels to attach to the collected metrics (default: None).",
    )


class AmazonBedrockLanguageModel(LanguageModel):
    """Language model that uses Amazon Bedrock models to generate responses."""

    def __init__(self, params: AmazonBedrockLanguageModelParams) -> None:
        """
        Initialize with Bedrock client parameters.

        See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html.

        Args:
            params (AmazonBedrockLanguageModelParams): Parameters for the language model.

        """
        super().__init__()

        self._client = params.client

        self._model_id = params.model_id

        self._inference_config = (
            {
                key: value
                for key, value in {
                    "maxTokens": params.inference_config.max_tokens,
                    "stopSequences": params.inference_config.stop_sequences,
                    "temperature": params.inference_config.temperature,
                    "topP": params.inference_config.top_p,
                }.items()
                if value is not None
            }
            if params.inference_config is not None
            else None
        )

        self._additional_model_request_fields = params.additional_model_request_fields
        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        metrics_factory = params.metrics_factory

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
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

    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T:
        """Generate a structured response parsed into the given Pydantic model."""
        from_bedrock: Callable[..., Any] | None = getattr(
            instructor, "from_bedrock", None
        )
        if from_bedrock is None:
            msg = "instructor.from_bedrock is not available"
            logger.error(msg)
            raise AttributeError(msg)

        client = from_bedrock(self._client, async_client=True)

        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        converse_kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "system": [{"text": system_prompt or "."}],
            "messages": [{"role": "user", "content": [{"text": user_prompt or "."}]}],
            "response_model": output_format,
            "max_retries": max_attempts,
        }

        if self._inference_config is not None:
            converse_kwargs["inferenceConfig"] = self._inference_config

        if self._additional_model_request_fields is not None:
            converse_kwargs["additionalModelRequestFields"] = (
                self._additional_model_request_fields
            )

        start_time = time.monotonic()

        response = await client.chat.completions.create(**converse_kwargs)

        end_time = time.monotonic()

        self._collect_metrics(response, start_time, end_time)

        return response

    async def generate_response(  # noqa: C901
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        """Generate a raw text response (and optional tool call) from Bedrock."""
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
                "tools": AmazonBedrockLanguageModel._format_tools(tools),
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
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from e

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

        self._collect_metrics(response, start_time, end_time)

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
                    },
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

    def _collect_metrics(
        self,
        response: dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> None:
        if self._should_collect_metrics:
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
                        },
                    },
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
