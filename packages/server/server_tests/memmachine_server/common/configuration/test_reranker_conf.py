import pytest
import yaml
from pydantic import SecretStr, ValidationError

from memmachine_server.common.configuration.reranker_conf import (
    AmazonBedrockRerankerConf,
    BM25RerankerConf,
    CohereRerankerConf,
    CrossEncoderRerankerConf,
    EmbedderRerankerConf,
    IdentityRerankerConf,
    RerankersConf,
    RRFHybridRerankerConf,
)


@pytest.fixture
def bm25_reranker_conf() -> dict:
    return {
        "provider": "bm25",
        "config": {
            "language": "english",
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "tokenizer": "default",
        },
    }


@pytest.fixture
def identity_reranker_conf() -> dict:
    return {"provider": "identity"}


@pytest.fixture
def amazon_bedrock_reranker_conf() -> dict:
    return {
        "provider": "amazon-bedrock",
        "config": {
            "region": "us-west-2",
            "aws_access_key_id": "key-id",
            "aws_secret_access_key": "secret-key",
            "model_id": "amazon.rerank-v1:0",
        },
    }


@pytest.fixture
def cohere_reranker_conf() -> dict:
    return {
        "provider": "cohere",
        "config": {
            "cohere_key": "test-cohere-key",
            "model": "rerank-english-v3.0",
            "base_url": "http://localhost:8000",
        },
    }


@pytest.fixture
def cross_encoder_reranker_conf() -> dict:
    return {
        "provider": "cross-encoder",
        "config": {
            "model_name": "cross-encoder/qnli-electra-base",
        },
    }


@pytest.fixture
def embedder_reranker_conf() -> dict:
    return {
        "provider": "embedder",
        "config": {
            "embedder_id": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }


@pytest.fixture
def rrf_hybrid_reranker_conf(bm25_reranker_conf, identity_reranker_conf) -> dict:
    return {
        "provider": "rrf-hybrid",
        "config": {
            "reranker_ids": ["id_ranker_id", "bm_ranker_id"],
            "k": 60,
        },
    }


@pytest.fixture
def full_reranker_input(
    bm25_reranker_conf,
    identity_reranker_conf,
    amazon_bedrock_reranker_conf,
    cohere_reranker_conf,
    cross_encoder_reranker_conf,
    embedder_reranker_conf,
    rrf_hybrid_reranker_conf,
) -> dict:
    return {
        "rerankers": {
            "my_reranker_id": rrf_hybrid_reranker_conf,
            "id_ranker_id": identity_reranker_conf,
            "bm_ranker_id": bm25_reranker_conf,
            "aws_reranker_id": amazon_bedrock_reranker_conf,
            "cohere_reranker_id": cohere_reranker_conf,
            "cross_encoder_id": cross_encoder_reranker_conf,
            "embedder_id": embedder_reranker_conf,
        },
    }


def test_valid_bm25_reranker_conf(bm25_reranker_conf):
    conf = BM25RerankerConf(**bm25_reranker_conf["config"])
    assert conf.language == "english"
    assert conf.k1 == 1.5
    assert conf.b == 0.75
    assert conf.epsilon == 0.25
    assert conf.tokenizer == "default"


def test_valid_identity_reranker_conf(identity_reranker_conf):
    conf = IdentityRerankerConf()
    assert conf is not None  # IdentityRerankerConf has no fields


def test_valid_amazon_bedrock_reranker_conf(amazon_bedrock_reranker_conf):
    conf = AmazonBedrockRerankerConf(**amazon_bedrock_reranker_conf["config"])
    assert conf.region == "us-west-2"
    assert conf.aws_access_key_id == SecretStr("key-id")
    assert conf.aws_secret_access_key == SecretStr("secret-key")
    assert conf.model_id == "amazon.rerank-v1:0"


def test_valid_amazon_bedrock_reranker_conf_without_explicit_credentials(
    monkeypatch,
):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    conf = AmazonBedrockRerankerConf(
        region="us-west-2",
        model_id="amazon.rerank-v1:0",
    )

    assert conf.region == "us-west-2"
    assert conf.aws_access_key_id is None
    assert conf.aws_secret_access_key is None
    assert conf.aws_session_token is None
    assert conf.model_id == "amazon.rerank-v1:0"


def test_valid_amazon_bedrock_reranker_conf_uses_env_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env-session-token")

    conf = AmazonBedrockRerankerConf(
        region="us-west-2",
        model_id="amazon.rerank-v1:0",
    )

    assert conf.aws_access_key_id == SecretStr("env-key-id")
    assert conf.aws_secret_access_key == SecretStr("env-secret-key")
    assert conf.aws_session_token == SecretStr("env-session-token")


def test_valid_cohere_reranker_conf(cohere_reranker_conf):
    conf = CohereRerankerConf(**cohere_reranker_conf["config"])
    assert conf.cohere_key == SecretStr("test-cohere-key")
    assert conf.model == "rerank-english-v3.0"
    assert conf.base_url == "http://localhost:8000"


def test_valid_cross_encoder_reranker_conf(cross_encoder_reranker_conf):
    conf = CrossEncoderRerankerConf(**cross_encoder_reranker_conf["config"])
    assert conf.model_name == "cross-encoder/qnli-electra-base"


def test_valid_embedder_reranker_conf(embedder_reranker_conf):
    conf = EmbedderRerankerConf(**embedder_reranker_conf["config"])
    assert conf.embedder_id == "sentence-transformers/all-MiniLM-L6-v2"


def test_valid_rrf_hybrid_reranker_conf(rrf_hybrid_reranker_conf):
    conf = RRFHybridRerankerConf(**rrf_hybrid_reranker_conf["config"])
    assert conf.reranker_ids == ["id_ranker_id", "bm_ranker_id"]
    assert conf.k == 60


def test_full_reranker_conf(full_reranker_input):
    conf = RerankersConf.parse(full_reranker_input)
    assert "my_reranker_id" in conf.rrf_hybrid
    hybrid = conf.rrf_hybrid["my_reranker_id"]
    assert hybrid.reranker_ids == ["id_ranker_id", "bm_ranker_id"]

    assert "bm_ranker_id" in conf.bm25
    assert conf.bm25["bm_ranker_id"].k1 == 1.5

    assert "id_ranker_id" in conf.identity

    assert "aws_reranker_id" in conf.amazon_bedrock
    assert conf.amazon_bedrock["aws_reranker_id"].region == "us-west-2"
    assert "cohere_reranker_id" in conf.cohere
    assert conf.cohere["cohere_reranker_id"].base_url == "http://localhost:8000"

    assert "cross_encoder_id" in conf.cross_encoder
    assert (
        conf.cross_encoder["cross_encoder_id"].model_name
        == "cross-encoder/qnli-electra-base"
    )

    assert "embedder_id" in conf.embedder
    assert (
        conf.embedder["embedder_id"].embedder_id
        == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_serialize_deserialize_reranker_conf(full_reranker_input):
    conf = RerankersConf.parse(full_reranker_input)
    serialized = conf.to_yaml()
    conf_cp = RerankersConf.parse(yaml.safe_load(serialized))
    assert conf == conf_cp


def test_invalid_cohere_base_url():
    with pytest.raises(ValidationError, match="Invalid base URL"):
        CohereRerankerConf(
            cohere_key=SecretStr("test-cohere-key"),
            model="rerank-english-v3.0",
            base_url="localhost:8000",
        )
