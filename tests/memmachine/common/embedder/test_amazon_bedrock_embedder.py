import pytest
from langchain_aws import BedrockEmbeddings

from memmachine.common.embedder.amazon_bedrock_embedder import (
    AmazonBedrockEmbedder,
    AmazonBedrockEmbedderParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def embedder(boto3_bedrock_runtime_client):
    return AmazonBedrockEmbedder(
        AmazonBedrockEmbedderParams(
            client=BedrockEmbeddings(
                client=boto3_bedrock_runtime_client,
                model_id="amazon.titan-embed-text-v2:0",
            ),
            model_id="amazon.titan-embed-text-v2:0",
            max_input_length=2000,
        )
    )


@pytest.fixture(
    params=[
        ["Are tomatoes fruits?", "Tomatoes are red."],
        ["Are tomatoes fruits?", "Tomatoes are red.", ""],
        ["."],
        [" "],
        [""],
        [],
    ],
)
def inputs(request):
    return request.param


@pytest.mark.asyncio
async def test_ingest_embed(embedder, inputs):
    embeddings = await embedder.ingest_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == embedder.dimensions for embedding in embeddings)


@pytest.mark.asyncio
async def test_search_embed(embedder, inputs):
    embeddings = await embedder.search_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == embedder.dimensions for embedding in embeddings)


@pytest.mark.asyncio
async def test_large_input(embedder):
    input_text = "👩‍💻" * 10000

    await embedder.ingest_embed([input_text])
    await embedder.search_embed([input_text])
