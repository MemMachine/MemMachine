import random

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeEntry, EpisodeStorage


class SpyEmbedder(Embedder):
    """Test double that records calls and produces deterministic embeddings."""

    def __init__(self) -> None:
        self.ingest_calls: list[list[str]] = []
        self.search_calls: list[list[str]] = []

    async def ingest_embed(
        self,
        inputs: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        self.ingest_calls.append(list(inputs))
        return [self._vector(text) for text in inputs]

    async def search_embed(
        self,
        queries: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        self.search_calls.append(list(queries))
        return [self._vector(text) for text in queries]

    @property
    def model_id(self) -> str:
        return "spy-embedder"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        score_alpha = 1.0 if "alpha" in lowered else -1.0
        score_beta = 1.0 if "beta" in lowered else -1.0
        return [score_alpha, score_beta]


class LengthEmbedder(Embedder):
    """Returns a random vector of length n"""

    def __init__(self, n: int) -> None:
        self.n = n

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        return [[random.random() for _ in range(self.n)] for _ in inputs]

    async def ingest_embed(
        self, inputs: list[str], max_attempts: int = 1
    ) -> list[list[float]]:
        return self._embed(inputs)

    async def search_embed(
        self, queries: list[str], max_attempts: int = 1
    ) -> list[list[float]]:
        return self._embed(queries)

    @property
    def model_id(self) -> str:
        return str(self.n)

    @property
    def dimensions(self) -> int:
        return self.n

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


async def add_history(history_storage: EpisodeStorage, content: str):
    episodes = await history_storage.add_episodes(
        session_key="session_id",
        episodes=[
            EpisodeEntry(
                content=content,
                producer_id="profile_id",
                producer_role="dev",
            )
        ],
    )
    return episodes[0].uid
