import pytest
from pydantic import SecretStr

from memmachine.common.configuration.reranker_conf import (
    AmazonBedrockRerankerConf,
    BM25RerankerConf,
    CrossEncoderRerankerConf,
    RerankerConf,
    RRFHybridRerankerConf,
)
from memmachine.common.resource_mgr.reranker_mgr import RerankerMgr


@pytest.fixture
def mock_conf():
    conf = RerankerConf(
        rrf_hybrid={
            "my_reranker_id": RRFHybridRerankerConf(
                reranker_ids=["bm_ranker_id", "ce_ranker_id", "id_ranker_id"],
            )
        },
        identity={"id_ranker_id": {}},
        bm25={"bm_ranker_id": BM25RerankerConf(tokenize="simple")},
        cross_encoder={
            "ce_ranker_id": CrossEncoderRerankerConf(
                model_name="cross-encoder/qnli-electra-base",
            )
        },
        amazon_bedrock={
            "aws_reranker_id": AmazonBedrockRerankerConf(
                model_id="amazon.rerank-v1:0",
                aws_access_key_id=SecretStr("<AWS_ACCESS_KEY_ID>"),
                aws_secret_access_key=SecretStr("<AWS_SECRET_ACCESS_KEY>"),
                region="us-east-1",
            )
        },
    )
    return conf


def test_build_bm25_rerankers(mock_conf):
    builder = RerankerMgr(mock_conf)
    builder._build_bm25_rerankers()

    assert "bm_ranker_id" in builder.rerankers
    reranker = builder.rerankers["bm_ranker_id"]
    assert reranker is not None


def test_build_cross_encoder_rerankers(mock_conf):
    builder = RerankerMgr(mock_conf)
    builder._build_cross_encoder_rerankers()

    assert "ce_ranker_id" in builder.rerankers
    reranker = builder.rerankers["ce_ranker_id"]
    assert reranker is not None


def test_amazon_bedrock_rerankers(mock_conf):
    builder = RerankerMgr(mock_conf)
    builder._build_amazon_bedrock_rerankers()

    assert "aws_reranker_id" in builder.rerankers
    reranker = builder.rerankers["aws_reranker_id"]
    assert reranker is not None


def test_identity_rerankers(mock_conf):
    builder = RerankerMgr(mock_conf)
    builder._build_identity_rerankers()

    assert "id_ranker_id" in builder.rerankers
    reranker = builder.rerankers["id_ranker_id"]
    assert reranker is not None


def test_build_rrf_hybrid_rerankers(mock_conf):
    builder = RerankerMgr(mock_conf)
    builder.build_all()

    assert "my_reranker_id" in builder.rerankers
    reranker = builder.rerankers["my_reranker_id"]
    assert reranker is not None
