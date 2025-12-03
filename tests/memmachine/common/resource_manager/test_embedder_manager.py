import pytest
from pydantic import SecretStr

from memmachine.common.configuration import EmbeddersConf
from memmachine.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConfig,
    OpenAIEmbedderConf,
    SentenceTransformerEmbedderConfig,
)
from memmachine.common.embedder import Embedder
from memmachine.common.resource_manager.embedder_manager import EmbedderManager


@pytest.fixture
def mock_conf():
    conf = EmbeddersConf(
        amazon_bedrock={
            "aws_embedder_id": AmazonBedrockEmbedderConfig(
                model_id="amazon.embed-v1:0",
                aws_access_key_id=SecretStr("<AWS_ACCESS_KEY_ID>"),
                aws_secret_access_key=SecretStr("<AWS_SECRET_ACCESS_KEY>"),
                region="us-east-1",
            ),
        },
        openai={
            "openai_embedder_id": OpenAIEmbedderConf(
                model="text-embedding-ada-002",
                api_key=SecretStr("<OPENAI_API_KEY>"),
            ),
        },
        sentence_transformer={
            "sentence_transformer_id": SentenceTransformerEmbedderConfig(
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
        },
    )
    return conf


@pytest.mark.asyncio
async def test_build_amazon_bedrock_embedders(mock_conf):
    builder = EmbedderManager(mock_conf)
    await builder.build_all()

    assert "aws_embedder_id" in builder._embedders
    embedder = builder._embedders["aws_embedder_id"]
    assert isinstance(embedder, Embedder)


@pytest.mark.asyncio
async def test_build_openai_embedders(mock_conf):
    builder = EmbedderManager(mock_conf)
    await builder.build_all()

    assert "openai_embedder_id" in builder._embedders
    embedder = builder._embedders["openai_embedder_id"]
    assert isinstance(embedder, Embedder)


@pytest.mark.asyncio
async def test_build_sentence_transformer_embedders(mock_conf):
    builder = EmbedderManager(mock_conf)
    await builder.build_all()

    assert "sentence_transformer_id" in builder._embedders
    embedder = builder._embedders["sentence_transformer_id"]
    assert isinstance(embedder, Embedder)


@pytest.mark.asyncio
async def test_build_all(mock_conf):
    builder = EmbedderManager(mock_conf)
    all_embedders = await builder.build_all()

    assert "aws_embedder_id" in all_embedders
    assert "openai_embedder_id" in all_embedders
    assert "sentence_transformer_id" in all_embedders

    for embedder in all_embedders.values():
        assert isinstance(embedder, Embedder)
