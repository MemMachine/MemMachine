from typing import Any

import pytest
import yaml
from pydantic import SecretStr

from memmachine.common.configuration import EmbeddersConf
from memmachine.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConf,
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
    conf = AmazonBedrockEmbedderConf(**aws_embedder_conf["config"])
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


def test_get_embedder_name(embedder_conf):
    conf = EmbeddersConf.parse(embedder_conf)
    assert conf.get_openai_embedder_name() == "openai_embedder"
    assert conf.get_amazon_bedrock_embedder_name() == "aws_embedder_id"
    assert conf.get_sentence_transformer_embedder_name() is None

    assert isinstance(
        conf.get_openai_embedder_conf("openai_embedder"), OpenAIEmbedderConf
    )
    assert isinstance(
        conf.get_amazon_bedrock_embedder_conf("aws_embedder_id"),
        AmazonBedrockEmbedderConf,
    )


def test_embedder_to_yaml(embedder_conf):
    conf = EmbeddersConf.parse(embedder_conf)
    yaml_str = conf.to_yaml()
    conf_cp = EmbeddersConf.parse(yaml.safe_load(yaml_str))
    assert conf_cp == conf
    assert len(conf_cp.openai) == len(conf.openai)
    assert len(conf_cp.amazon_bedrock) == len(conf.amazon_bedrock)


def test_open_ai_embeder_without_key():
    conf_dict = {
        "model": "text-embedding-ada-002",
    }
    conf = OpenAIEmbedderConf(**conf_dict)
    assert conf.api_key.get_secret_value() == ""


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ["MY_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
        monkeypatch.delenv(var, raising=False)


def test_read_aws_keys_from_env(monkeypatch, aws_embedder_conf):
    monkeypatch.setenv("MY_KEY_ID", "my-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "access-key")
    aws_embedder_conf["config"]["aws_access_key_id"] = "${MY_KEY_ID}"
    aws_embedder_conf["config"]["aws_secret_access_key"] = ""
    conf = AmazonBedrockEmbedderConf(**aws_embedder_conf["config"])
    assert conf.aws_access_key_id.get_secret_value() == "my-key-id"
    assert conf.aws_secret_access_key.get_secret_value() == "access-key"
    assert conf.aws_session_token is None
