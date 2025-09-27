"""
Amazon Bedrock-based reranker implementation.
"""

import asyncio
import os
from typing import Any

import boto3

from .reranker import Reranker


class AmazonBedrockReranker(Reranker):
    """
    Reranker that uses Amazon Bedrock models
    to score relevance of candidates to a query.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize an AmazonBedrockReranker
        with the provided configuration.
        See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - region (str, optional):
                    AWS region where Bedrock is hosted
                    (default: "us-west-2").
                - aws_access_key_id (str, optional):
                    AWS access key ID for authentication.
                    If not provided, will use the value
                    of the AWS_ACCESS_KEY_ID environment variable.
                - aws_secret_access_key (str, optional):
                    AWS secret access key for authentication.
                    If not provided, will use the value
                    of the AWS_SECRET_ACCESS_KEY environment variable.
                - model_id (str):
                    ID of the Bedrock model to use for reranking.
                    (e.g. "amazon.rerank-v1:0", "cohere.rerank-v3-5:0")
                - additional_model_request_fields (dict, optional):
                    Keys are request fields for the model
                    and values are values for those fields.
        """
        super().__init__()

        region = config.get("region", "us-west-2")
        if not isinstance(region, str):
            raise TypeError("region must be a string")

        aws_access_key_id = config.get(
            "aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID")
        )
        if not isinstance(aws_access_key_id, str):
            raise TypeError("aws_access_key_id must be a string")

        aws_secret_access_key = config.get(
            "aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        if not isinstance(aws_secret_access_key, str):
            raise TypeError("aws_secret_access_key must be a string")

        self._client = boto3.client(
            "bedrock-agent-runtime",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        additional_model_request_fields = config.get(
            "additional_model_request_fields", {}
        )
        if not isinstance(additional_model_request_fields, dict):
            raise TypeError(
                "additional_model_request_fields must be a dictionary"
            )

        model_id = config.get("model_id")
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a string")

        model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"

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

        response = await asyncio.to_thread(
            self._client.rerank,
            **rerank_kwargs,
        )

        results = response["results"]
        while len(results) < len(candidates) and "nextToken" in response:
            next_token = response["nextToken"]

            response = await asyncio.to_thread(
                self._client.rerank,
                **rerank_kwargs,
                nextToken=next_token,
            )

            results += response["results"]

        scores = [0.0] * len(candidates)
        for result in results:
            scores[result["index"]] = result["relevanceScore"]

        return scores
