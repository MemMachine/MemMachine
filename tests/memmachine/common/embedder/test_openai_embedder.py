import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(
    params=[
        ["Are tomatoes fruits?", "Tomatoes are red."],
        ["Are tomatoes fruits?", "Tomatoes are red.", ""],
        [""],
        [],
    ],
)
def inputs(request):
    return request.param


@pytest.mark.asyncio
async def test_ingest_embed(openai_embedder, inputs):
    embeddings = await openai_embedder.ingest_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == openai_embedder.dimensions for embedding in embeddings)


@pytest.mark.asyncio
async def test_search_embed(openai_embedder, inputs):
    embeddings = await openai_embedder.search_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == openai_embedder.dimensions for embedding in embeddings)


@pytest.mark.asyncio
async def test_large_input(openai_embedder):
    input_text = "👩‍💻" * 10000

    await openai_embedder.ingest_embed([input_text])
    await openai_embedder.search_embed([input_text])
