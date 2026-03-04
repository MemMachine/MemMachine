from typing import cast

import pytest
from pydantic import SecretStr

from memmachine_server.common.configuration.reranker_conf import (
    AmazonBedrockRerankerConf,
    BM25RerankerConf,
    CohereRerankerConf,
    CrossEncoderRerankerConf,
    IdentityRerankerConf,
    RerankersConf,
    RRFHybridRerankerConf,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.errors import InvalidRerankerError
from memmachine_server.common.resource_manager.reranker_manager import (
    EmbedderFactory,
    RerankerManager,
)
from server_tests.memmachine_server.conftest import requires_sentence_transformers


@pytest.fixture
def mock_conf():
    conf = RerankersConf(
        rrf_hybrid={
            "my_reranker_id": RRFHybridRerankerConf(
                reranker_ids=["bm_ranker_id", "ce_ranker_id", "id_ranker_id"],
            ),
        },
        identity={"id_ranker_id": IdentityRerankerConf()},
        bm25={"bm_ranker_id": BM25RerankerConf(tokenizer="simple")},
        cohere={
            "cohere_reranker_id": CohereRerankerConf(
                cohere_key=SecretStr("<COHERE_API_KEY>"),
                model="rerank-english-v3.0",
            ),
        },
        cross_encoder={
            "ce_ranker_id": CrossEncoderRerankerConf(
                model_name="cross-encoder/qnli-electra-base",
            ),
        },
        amazon_bedrock={
            "aws_reranker_id": AmazonBedrockRerankerConf(
                model_id="amazon.rerank-v1:0",
                aws_access_key_id=SecretStr("<AWS_ACCESS_KEY_ID>"),
                aws_secret_access_key=SecretStr("<AWS_SECRET_ACCESS_KEY>"),
                region="us-west-2",
            ),
        },
    )
    return conf


class FakeEmbedder(Embedder):
    async def ingest_embed(
        self,
        inputs: list[object],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return []

    async def search_embed(
        self,
        queries: list[object],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return []

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dimensions(self) -> int:
        return 0

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class FakeEmbedderFactory:
    async def get_embedder(self, _: str) -> Embedder:
        return FakeEmbedder()


@pytest.fixture
def reranker_manager(mock_conf):
    return RerankerManager(
        conf=mock_conf,
        embedder_factory=cast(EmbedderFactory, FakeEmbedderFactory()),
    )


@pytest.mark.asyncio
async def test_build_bm25_rerankers(reranker_manager):
    await reranker_manager.build_all()

    assert reranker_manager.has_reranker("bm_ranker_id")
    reranker = reranker_manager.get_reranker("bm_ranker_id")
    assert reranker is not None


@pytest.mark.asyncio
async def test_lazy_initialization(reranker_manager):
    assert reranker_manager.num_of_rerankers == 0

    await reranker_manager.get_reranker("bm_ranker_id")
    assert reranker_manager.has_reranker("bm_ranker_id")
    assert reranker_manager.num_of_rerankers == 1


@pytest.mark.asyncio
async def test_reranker_not_found(reranker_manager):
    with pytest.raises(
        InvalidRerankerError,
        match=r"Reranker with name unknown_reranker_id not found\.",
    ):
        await reranker_manager.get_reranker("unknown_reranker_id")


@pytest.mark.asyncio
async def test_build_cohere_rerankers(reranker_manager):
    await reranker_manager.build_all()

    assert reranker_manager.has_reranker("cohere_reranker_id")
    reranker = reranker_manager.get_reranker("cohere_reranker_id")
    assert reranker is not None


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_build_cross_encoder_rerankers(reranker_manager):
    await reranker_manager.build_all()

    assert reranker_manager.has_reranker("ce_ranker_id")
    reranker = reranker_manager.get_reranker("ce_ranker_id")
    assert reranker is not None


@pytest.mark.asyncio
async def test_amazon_bedrock_rerankers(reranker_manager):
    await reranker_manager.build_all()

    assert reranker_manager.has_reranker("aws_reranker_id")
    reranker = reranker_manager.get_reranker("aws_reranker_id")
    assert reranker is not None


@pytest.mark.asyncio
async def test_identity_rerankers(reranker_manager):
    await reranker_manager.build_all()

    assert reranker_manager.has_reranker("id_ranker_id")
    reranker = reranker_manager.get_reranker("id_ranker_id")
    assert reranker is not None


@requires_sentence_transformers
@pytest.mark.asyncio
async def test_build_rrf_hybrid_rerankers(reranker_manager):
    await reranker_manager.build_all()

    assert reranker_manager.has_reranker("my_reranker_id")
    reranker = reranker_manager.get_reranker("my_reranker_id")
    assert reranker is not None
