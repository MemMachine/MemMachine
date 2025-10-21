"""
Amazon Bedrock-based reranker implementation.
"""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import boto3
from pydantic import BaseModel, Field, SecretStr

from memmachine.common.data_types import ExternalServiceAPIError

from .reranker import Reranker

logger = logging.getLogger(__name__)


class AmazonBedrockRerankerConfig(BaseModel):
    """
    Configuration for AmazonBedrockReranker.

    Attributes:
        region (str):
            AWS region where Bedrock is hosted.
        aws_access_key_id (SecretStr):
            AWS access key ID for authentication.
        aws_secret_access_key (SecretStr):
            AWS secret access key for authentication.
        model_id (str):
            ID of the Bedrock model to use for reranking
            (e.g. 'amazon.rerank-v1:0', 'cohere.rerank-v3-5:0').
        additional_model_request_fields (dict[str, Any], optional):
            Keys are request fields for the model
            and values are values for those fields
            (default: {}).
    """

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
            "ID of the Bedrock model to use for reranking "
            "(e.g. 'amazon.rerank-v1:0', 'cohere.rerank-v3-5:0')."
        ),
    )
    additional_model_request_fields: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keys are request fields for the model "
            "and values are values for those fields."
        ),
    )


class AmazonBedrockReranker(Reranker):
    """
    Reranker that uses Amazon Bedrock models
    to score relevance of candidates to a query.
    """

    def __init__(self, config: AmazonBedrockRerankerConfig):
        """
        Initialize an AmazonBedrockReranker
        with the provided configuration.
        See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html

        Args:
            config (AmazonBedrockRerankerConfig):
                Configuration for the reranker.
        """
        super().__init__()

        region = config.region
        aws_access_key_id = config.aws_access_key_id
        aws_secret_access_key = config.aws_secret_access_key
        additional_model_request_fields = config.additional_model_request_fields

        self._model_id = config.model_id

        self._client = boto3.client(
            "bedrock-agent-runtime",
            region_name=region,
            aws_access_key_id=aws_access_key_id.get_secret_value(),
            aws_secret_access_key=aws_secret_access_key.get_secret_value(),
        )

        model_arn = f"arn:aws:bedrock:{region}::foundation-model/{self._model_id}"

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
