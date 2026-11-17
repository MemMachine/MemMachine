import random
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeEntry, EpisodeStorage
from memmachine_server.semantic_memory.semantic_memory import SemanticService
from memmachine_server.semantic_memory.semantic_model import SetIdT
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage


@dataclass(frozen=True)
class SessionData:
    """Org and project a semantic memory request acts for."""

    org_id: str
    project_id: str


class SpyEmbedder(Embedder):
    """Test double that records calls and produces deterministic embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.ingest_calls: list[list[str]] = []
        self.search_calls: list[list[str]] = []

    async def _ingest_embed(
        self,
        inputs: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        self.ingest_calls.append(list(inputs))
        return [self._vector(text) for text in inputs]

    async def _search_embed(
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
        super().__init__()
        self.n = n

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        return [[random.random() for _ in range(self.n)] for _ in inputs]

    async def _ingest_embed(
        self, inputs: list[str], max_attempts: int = 1
    ) -> list[list[float]]:
        return self._embed(inputs)

    async def _search_embed(
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


async def register_history(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    set_id: SetIdT,
    history_ids: Sequence[UUID],
) -> None:
    """Queue episodes for ingestion into a set, at their episode times."""
    for history_id in history_ids:
        episode = await episode_storage.get_episode(history_id)
        await semantic_storage.add_history_to_set(
            set_id=set_id,
            history_id=history_id,
            created_at=None if episode is None else episode.created_at,
        )


async def get_set_id(
    semantic_service: SemanticService,
    session_data: SessionData,
    *,
    is_org_level: bool = False,
    **set_metadata: str,
) -> SetIdT:
    """Return the id of the session's set with the given set metadata."""
    return await semantic_service.get_set_id(
        session_data=session_data,
        set_metadata_keys=list(set_metadata),
        set_metadata=set_metadata,
        is_org_level=is_org_level,
    )
