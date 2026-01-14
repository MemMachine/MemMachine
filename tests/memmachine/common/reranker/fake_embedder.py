from typing import Any

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder import Embedder


class FakeEmbedder(Embedder):
    def __init__(
        self,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size)

        self._similarity_metric = similarity_metric

    async def _ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [[float(len(_input)), -float(len(_input))] for _input in inputs]

    async def _search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [[float(len(query)), -float(len(query))] for query in queries]

    @property
    def model_id(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return self._similarity_metric
