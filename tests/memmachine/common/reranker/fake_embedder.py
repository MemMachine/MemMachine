from memmachine.common.embedder import Embedder, SimilarityMetric


class FakeEmbedder(Embedder):
    async def ingest_embed(
        self,
        inputs: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [[float(len(input)), -float(len(input))] for input in inputs]

    async def search_embed(
        self,
        queries: list[str],
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
        return SimilarityMetric.COSINE
