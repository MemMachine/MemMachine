"""
Amazon Bedrock-based reranker implementation.
"""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from memmachine.common.data_types import ExternalServiceAPIError

from .reranker import Reranker

logger = logging.getLogger(__name__)


class AmazonBedrockRerankerParams(BaseModel):
    """
    Parameters for AmazonBedrockReranker.

    Attributes:
        client (Any):
            Boto3 Agents for Amazon Bedrock Runtime client
            to use for making API calls.
        region (str):
            AWS region where the Bedrock model is hosted.
        model_id (str):
            ID of the Bedrock model to use for reranking
            (e.g. 'amazon.rerank-v1:0', 'cohere.rerank-v3-5:0').
        additional_model_request_fields (dict[str, Any], optional):
            Keys are request fields for the model
            and values are values for those fields
            (default: {}).
    """

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


class AmazonBedrockReranker(Reranker):
    """
    Reranker that uses Amazon Bedrock models
    to score relevance of candidates to a query.
    """

    def __init__(self, params: AmazonBedrockRerankerParams):
        """
        Initialize an AmazonBedrockReranker
        with the provided parameters.
        See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html

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

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        rerank_kwargs = {
            "queries": [
                {
                    "textQuery": {"text": query},
                    "type": "TEXT",
                }
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
                        "textDocument": {"text": candidate},
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
                logger.error(error_message)
                raise ExternalServiceAPIError(error_message)

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
            logger.error(error_message)
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

        return scores
