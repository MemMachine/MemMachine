"""
Amazon Bedrock-based embedder implementation.
"""

import asyncio
import os
from typing import Any

from langchain_aws import BedrockEmbeddings

from .embedder import Embedder


class AmazonBedrockEmbedder(Embedder):
    """
    Embedder that uses Amazon Bedrock models
    to generate embeddings for inputs and queries.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize an AmazonBedrockEmbedder
        with the provided configuration.

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
                    (e.g. "amazon.titan-embed-text-v2:0")
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

        model_id = config.get("model_id")
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a string")

        self._embeddings = BedrockEmbeddings(
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            model_id=model_id,
        )

    async def ingest_embed(self, inputs: list[Any]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(inputs)

    async def search_embed(self, queries: list[Any]) -> list[list[float]]:
        embed_queries_tasks = [
            self._embeddings.aembed_query(query) for query in queries
        ]
        return await asyncio.gather(*embed_queries_tasks)
