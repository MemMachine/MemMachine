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


@pytest.mark.asyncio
async def test_embed_oversized_input_with_no_max_input_length():
    """Regression for #1298: text > 75,000 chars must not raise ValueError.

    When max_input_length is None, _embed() must fall back to
    max_total_input_length_per_request (75,000) as the chunking bound so that
    cluster_texts() never receives a text that exceeds its hard limit.
    Prior to the fix, this raised ValueError("Text length ... exceeds
    max_total_length_per_cluster ..."), which surfaced as HTTP 500.
    """
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 256

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.total_tokens = 10

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=mock_client,
            model="test-model",
            dimensions=256,
            max_input_length=None,  # default — reproduces the crash scenario
        )
    )

    # 80,000 chars exceeds the 75,000-char cluster hard limit by 5,000.
    long_text = "x" * 80_000

    # Before the fix this raised:
    #   ValueError: Text length 80000 exceeds max_total_length_per_cluster 75000
    result = await embedder.ingest_embed([long_text])

    assert len(result) == 1
    assert len(result[0]) == 256
    # The text must have been split: greedy chunks of 80,000 chars at limit
    # 75,000 produce chunks of 75,000 and 5,000, each in its own cluster.
    assert mock_client.embeddings.create.call_count >= 2


@pytest.mark.asyncio
async def test_embed_max_input_length_above_request_cap():
    """max_input_length above max_total_input_length_per_request must not raise.

    A configured chunk limit above the per-request cap can never be honored
    (no chunk larger than the cap fits in any request), so chunking clamps
    to the cap instead of handing cluster_texts() a chunk it rejects.
    """
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 256

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.total_tokens = 10

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=mock_client,
            model="test-model",
            dimensions=256,
            max_input_length=80_000,  # above the 75,000-char per-request cap
        )
    )

    long_text = "x" * 100_000

    result = await embedder.ingest_embed([long_text])

    assert len(result) == 1
    assert len(result[0]) == 256
    # Clamped to 75,000, the text splits into 75,000 + 25,000 chars, which
    # exceed one request's total together, so each chunk gets its own call.
    assert mock_client.embeddings.create.call_count == 2
