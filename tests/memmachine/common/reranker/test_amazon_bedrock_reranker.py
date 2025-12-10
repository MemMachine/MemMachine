import pytest

from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def reranker(boto3_bedrock_agent_runtime_client, bedrock_integration_config):
    return AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=boto3_bedrock_agent_runtime_client,
            region=bedrock_integration_config["aws_region"],
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
