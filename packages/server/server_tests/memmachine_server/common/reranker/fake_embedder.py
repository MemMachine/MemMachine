from typing import Any

from memmachine_server.common.embedder import Embedder


class FakeEmbedder(Embedder):
    def __init__(
        self,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size)

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
