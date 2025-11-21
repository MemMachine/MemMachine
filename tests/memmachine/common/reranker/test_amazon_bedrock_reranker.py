import os

import boto3
import pytest

from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def boto3_client():
    return boto3.client(
        "bedrock-agent-runtime",
        region_name=os.environ.get("AWS_REGION", "us-west-2"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


@pytest.fixture
def reranker(boto3_client):
    return AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=boto3_client,
            region=os.environ.get("AWS_REGION", "us-west-2"),
            model_id="amazon.rerank-v1:0",
        )
    )


@pytest.mark.asyncio
async def test_rerank_sanity(reranker):
    query = "What is the capital of France?"
    candidates = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "Some apples are red.",
    ]

    scores = await reranker.score(query, candidates)

    assert len(scores) == len(candidates)
    assert scores[0] > scores[1] > scores[2]


@pytest.mark.asyncio
async def test_large_query(reranker):
    query = "👩‍💻" * 100000
    candidates = ["Candidate 1", "Candidate 2"]

    await reranker.rerank(query, candidates)


@pytest.mark.asyncio
async def test_large_document(reranker):
    query = "Query"
    candidates = ["👩‍💻" * 100000, "Candidate 2"]

    await reranker.rerank(query, candidates)
