from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from memmachine.common.configuration import EmbeddersConf
from memmachine.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConfig,
    OpenAIEmbedderConf,
)
from memmachine.common.data_types import SimilarityMetric


@pytest.fixture
def openai_embedder_conf() -> dict[str, Any]:
    return {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002",
            "api_key": "open-ai-key",
        },
    }


@pytest.fixture
def aws_embedder_conf() -> dict[str, Any]:
    return {
        "provider": "amazon-bedrock",
        "config": {
            "region": "us-west-2",
            "aws_access_key_id": "key-id",
            "aws_secret_access_key": "secret-key",
            "model_id": "amazon.titan-embed-text-v2:0",
            "similarity_metric": "cosine",
        },
    }


@pytest.fixture
def ollama_embedder_conf() -> dict[str, Any]:
    return {
        "provider": "openai",
        "config": {
            "model": "nomic-embed-text",
            "api_key": "empty",
            "base_url": "http://localhost:11434",
            "dimensions": 768,
        },
    }


@pytest.fixture
def embedder_conf(
    openai_embedder_conf,
    aws_embedder_conf,
    ollama_embedder_conf,
) -> dict[str, Any]:
    return {
        "embedders": {
            "openai_embedder": openai_embedder_conf,
            "aws_embedder_id": aws_embedder_conf,
            "ollama_embedder": ollama_embedder_conf,
        },
    }


def test_valid_open_ai_embedder_config(openai_embedder_conf):
    conf = OpenAIEmbedderConf(**openai_embedder_conf["config"])
    assert conf.model == "text-embedding-ada-002"
    assert conf.api_key == SecretStr("open-ai-key")


def test_valid_aws_bedrock_embedder_config(aws_embedder_conf):
    conf = AmazonBedrockEmbedderConfig(**aws_embedder_conf["config"])
    assert conf.region == "us-west-2"
    assert conf.aws_access_key_id == SecretStr("key-id")
    assert conf.aws_secret_access_key == SecretStr("secret-key")
    assert conf.model_id == "amazon.titan-embed-text-v2:0"
    assert conf.similarity_metric == SimilarityMetric.COSINE


def test_valid_ollama_embedder_config(ollama_embedder_conf):
    conf = OpenAIEmbedderConf(**ollama_embedder_conf["config"])
    assert conf.model == "nomic-embed-text"
    assert conf.api_key == SecretStr("empty")
    assert conf.base_url == "http://localhost:11434"
    assert conf.dimensions == 768


def test_full_embedder_conf(embedder_conf):
    conf = EmbeddersConf.parse(embedder_conf)
    assert len(conf.amazon_bedrock) > 0
    assert len(conf.openai) > 0
    assert conf.amazon_bedrock.get("aws_embedder_id") is not None
    assert conf.openai.get("openai_embedder") is not None
    assert conf.openai.get("ollama_embedder") is not None


def test_open_ai_embeder_without_key():
    conf_dict = {
        "model": "text-embedding-ada-002",
    }
    with pytest.raises(ValidationError) as exc_info:
        OpenAIEmbedderConf(**conf_dict)

    assert "missing" in str(exc_info.value)
