from typing import cast
from unittest.mock import AsyncMock

import cohere
import pytest
import yaml
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
from memmachine_server.common.reranker.cohere_reranker import CohereReranker
from memmachine_server.common.reranker.rrf_hybrid_reranker import RRFHybridReranker
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
                cohere_key=SecretStr("test-cohere-key"),
                model="rerank-english-v3.0",
                base_url="http://localhost:8000",
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


@pytest.mark.asyncio
async def test_build_cohere_reranker_passes_base_url(monkeypatch):
    captured_kwargs = {}
    fake_client = AsyncMock(spec=cohere.AsyncClientV2)

    def fake_client_v2(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setattr("cohere.AsyncClientV2", fake_client_v2)

    conf = RerankersConf(
        cohere={
            "cohere_reranker_id": CohereRerankerConf(
                cohere_key=SecretStr("test-cohere-key"),
                model="rerank-english-v3.0",
                base_url="http://localhost:8000",
            ),
        },
    )
    reranker_manager = RerankerManager(
        conf=conf,
        embedder_factory=cast(EmbedderFactory, FakeEmbedderFactory()),
    )

    reranker = await reranker_manager.get_reranker("cohere_reranker_id")

    assert isinstance(reranker, CohereReranker)
    assert captured_kwargs == {
        "api_key": "test-cohere-key",
        "base_url": "http://localhost:8000",
    }


@pytest.mark.asyncio
async def test_build_cohere_reranker_omits_base_url_when_not_configured(monkeypatch):
    captured_kwargs = {}
    fake_client = AsyncMock(spec=cohere.AsyncClientV2)

    def fake_client_v2(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setattr("cohere.AsyncClientV2", fake_client_v2)

    conf = RerankersConf(
        cohere={
            "cohere_reranker_id": CohereRerankerConf(
                cohere_key=SecretStr("test-cohere-key"),
                model="rerank-english-v3.0",
            ),
        },
    )
    reranker_manager = RerankerManager(
        conf=conf,
        embedder_factory=cast(EmbedderFactory, FakeEmbedderFactory()),
    )

    reranker = await reranker_manager.get_reranker("cohere_reranker_id")

    assert isinstance(reranker, CohereReranker)
    assert captured_kwargs == {"api_key": "test-cohere-key"}


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


@pytest.mark.parametrize(
    ("configured_k", "expected_k"), [(None, 60), (0, 0), (1, 1), (60, 60), ("0", 0)]
)
@pytest.mark.asyncio
async def test_build_rrf_from_yaml_preserves_valid_k(
    configured_k: int | str | None, expected_k: int
) -> None:
    hybrid_config: dict[str, object] = {"reranker_ids": ["identity"]}
    if configured_k is not None:
        hybrid_config["k"] = configured_k
    document = yaml.safe_dump(
        {
            "rerankers": {
                "hybrid": {"provider": "rrf-hybrid", "config": hybrid_config},
                "identity": {"provider": "identity"},
            }
        }
    )
    conf = RerankersConf.parse(yaml.safe_load(document))
    restored_conf = RerankersConf.parse(yaml.safe_load(conf.to_yaml()))
    assert conf.rrf_hybrid["hybrid"].k == expected_k
    assert restored_conf.rrf_hybrid["hybrid"].k == expected_k

    manager = RerankerManager(
        conf=restored_conf,
        embedder_factory=cast(EmbedderFactory, FakeEmbedderFactory()),
    )
    reranker = await manager.get_reranker("hybrid")
    assert isinstance(reranker, RRFHybridReranker)
    scores = await reranker.score("query", ["first", "second"])
    assert scores == pytest.approx([1 / (expected_k + 1), 1 / (expected_k + 2)])
