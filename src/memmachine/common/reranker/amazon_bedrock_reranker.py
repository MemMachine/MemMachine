"""Amazon Bedrock-based reranker implementation."""

import asyncio
import logging
import time
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.metrics_factory import MetricsFactory

from .reranker import Reranker

logger = logging.getLogger(__name__)


class AmazonBedrockRerankerParams(BaseModel):
    """Parameters for AmazonBedrockReranker."""

    client: Any = Field(
        ...,
        description=(
            "Boto3 Agents for Amazon Bedrock Runtime client to use for making API calls"
        ),
    )
    region: str = Field(
        ...,
        description="AWS region where the Bedrock model is hosted",
    )
    model_id: str = Field(
        ...,
        description=(
            "ID of the Bedrock model to use for reranking "
            "(e.g. 'amazon.rerank-v1:0', 'cohere.rerank-v3-5:0')"
        ),
    )
    additional_model_request_fields: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keys are request fields for the model "
            "and values are values for those fields"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels to attach to the collected metrics",
    )


class AmazonBedrockReranker(Reranker):
    """Reranker that uses Amazon Bedrock models to score candidate relevance."""

    def __init__(self, params: AmazonBedrockRerankerParams) -> None:
        """
        Initialize the Bedrock reranker with client parameters.

        See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html.

        Args:
            params (AmazonBedrockRerankerParams):
                Configuration for the reranker.

        """
        super().__init__()

        self._client = params.client

        additional_model_request_fields = params.additional_model_request_fields

        self._model_id = params.model_id
        model_arn = (
            f"arn:aws:bedrock:{params.region}::foundation-model/{self._model_id}"
        )

        self._model_configuration = {
            "additionalModelRequestFields": additional_model_request_fields,
            "modelArn": model_arn,
        }

        metrics_factory = params.metrics_factory

        self._score_call_counter = None
        self._score_latency_summary = None

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
            label_names = self._user_metrics_labels.keys()

            self._score_call_counter = metrics_factory.get_counter(
                name="amazon_bedrock_reranker_score_calls",
                description="Number of calls to score in AmazonBedrockReranker",
                label_names=label_names,
            )
            self._score_latency_summary = metrics_factory.get_summary(
                name="amazon_bedrock_reranker_score_latency_seconds",
                description="Latency in seconds for score in AmazonBedrockReranker",
            )

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates for a query using the Bedrock reranker."""
        rerank_kwargs = {
            "queries": [
                {
                    "textQuery": {"text": AmazonBedrockReranker._sanitize_query(query)},
                    "type": "TEXT",
                },
            ],
            "rerankingConfiguration": {
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": self._model_configuration,
                    "numberOfResults": len(candidates),
                },
                "type": "BEDROCK_RERANKING_MODEL",
            },
            "sources": [
                {
                    "inlineDocumentSource": {
                        "textDocument": {
                            "text": AmazonBedrockReranker._sanitize_document(candidate)
                        },
                        "type": "TEXT",
                    },
                    "type": "INLINE",
                }
                for candidate in candidates
            ],
        }

        score_call_uuid = uuid4()

        start_time = time.monotonic()

        results: list = []
        next_token = ""
        while len(results) < len(candidates) and next_token is not None:
            if len(results) == 0:
                logger.debug(
                    "[call uuid: %s] "
                    "Scoring %d candidates for query using %s Amazon Bedrock model",
                    score_call_uuid,
                    len(candidates),
                    self._model_id,
                )
            else:
                logger.debug(
                    "[call uuid: %s] Retrieving next batch of scoring results",
                    score_call_uuid,
                )

            try:
                response = await asyncio.to_thread(
                    self._client.rerank,
                    **rerank_kwargs,
                )
            except Exception as e:
                if len(results) == 0:
                    error_message = (
                        f"[call uuid: {score_call_uuid}] "
                        "Failed to score candidates "
                        f"due to {type(e).__name__}"
                    )
                else:
                    error_message = (
                        f"[call uuid: {score_call_uuid}] "
                        "Failed to retrieve next batch of scoring results "
                        f"due to {type(e).__name__}"
                    )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from e

            next_token = response.get("nextToken")
            rerank_kwargs["nextToken"] = next_token

            batch_results = response["results"]
            logger.debug(
                "[call uuid: %s] Received %d %s scores in batch",
                score_call_uuid,
                len(batch_results),
                "initial" if len(results) == 0 else "additional",
            )

            results += batch_results

        if len(results) != len(candidates):
            error_message = (
                f"Expected {len(candidates)} total scores, but got {len(results)}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message)

        end_time = time.monotonic()

        logger.debug(
            "[call uuid: %s] Scoring completed in %.3f seconds",
            score_call_uuid,
            end_time - start_time,
        )

        scores = [0.0] * len(candidates)
        for result in results:
            scores[result["index"]] = result["relevanceScore"]

        if self._should_collect_metrics:
            cast(MetricsFactory.Counter, self._score_call_counter).increment(
                labels=self._user_metrics_labels
            )
            cast(MetricsFactory.Summary, self._score_latency_summary).observe(
                end_time - start_time, labels=self._user_metrics_labels
            )

        return scores

    @staticmethod
    def _sanitize_query(query: str) -> str:
        # Not documented in AWS documentation, but the query length is limited at 9000 UTF-16 code units.
        query_length_limit = 9000
        return (
            query.encode("utf-16-le")[:query_length_limit].decode(
                "utf-16-le", errors="ignore"
            )
            if query
            else "."
        )

    @staticmethod
    def _sanitize_document(document: str) -> str:
        # Text must be between 1 and 32000 characters, inclusive.
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RerankTextDocument.html
        document_length_limit = 32000
        return document[:document_length_limit] if document else "."
