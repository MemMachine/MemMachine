import pytest

pytest.importorskip("sentence_transformers")

from sentence_transformers import SentenceTransformer

from memmachine_server.common.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def embedder():
    return SentenceTransformerEmbedder(
        SentenceTransformerEmbedderParams(
            model_name="all-MiniLM-L6-v2",
            sentence_transformer=SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            max_input_length=2000,
        ),
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
    input_text = "👩‍💻" * 100000

    await embedder.ingest_embed([input_text])
    await embedder.search_embed([input_text])
