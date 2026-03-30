from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_embed(openai_embedder, inputs):
    embeddings = await openai_embedder.ingest_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == openai_embedder.dimensions for embedding in embeddings)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_embed(openai_embedder, inputs):
    embeddings = await openai_embedder.search_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == openai_embedder.dimensions for embedding in embeddings)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_input(openai_embedder):
    input_text = "👩‍💻" * 10000

    assert len(await openai_embedder.ingest_embed([input_text])) == 1
    assert len(await openai_embedder.search_embed([input_text])) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_many_inputs(openai_embedder):
    input_texts = ["Hello, world!"] * 10000
    assert len(await openai_embedder.ingest_embed(input_texts)) == 10000
    assert len(await openai_embedder.search_embed(input_texts)) == 10000


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_embed_retry_on_internal_server_error(mock_sleep):
    """Test retry logic on InternalServerError."""
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 256

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.total_tokens = 10

    mock_client.embeddings.create = AsyncMock(
        side_effect=[
            openai.InternalServerError(
                "internal server error", response=MagicMock(), body=None
            ),
            mock_response,
        ]
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=mock_client,
            model="test-model",
            dimensions=256,
        )
    )

    result = await embedder.ingest_embed(["test input"], max_attempts=2)

    assert len(result) == 1
    assert len(result[0]) == 256
    assert mock_client.embeddings.create.call_count == 2
    mock_sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_embed_fail_after_max_retries_on_internal_server_error(mock_sleep):
    """Test that ExternalServiceAPIError is raised after max_attempts on InternalServerError."""
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)
    mock_client.embeddings.create = AsyncMock(
        side_effect=openai.InternalServerError(
            "internal server error", response=MagicMock(), body=None
        ),
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=mock_client,
            model="test-model",
            dimensions=256,
        )
    )

    with pytest.raises(ExternalServiceAPIError, match=r"max attempts"):
        await embedder.ingest_embed(["test input"], max_attempts=3)

    assert mock_client.embeddings.create.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_await(1)
    mock_sleep.assert_any_await(2)
