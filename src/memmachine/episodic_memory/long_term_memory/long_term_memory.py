from collections.abc import Iterable, Mapping
from typing import Any, cast

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder.embedder import Embedder
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store import VectorGraphStore

from ..data_types import ContentType, Episode
from ..declarative_memory import DeclarativeMemory, DeclarativeMemoryParams
from ..declarative_memory.data_types import (
    ContentType as DeclarativeMemoryContentType,
)
from ..declarative_memory.data_types import (
    Episode as DeclarativeMemoryEpisode,
)

content_type_to_declarative_memory_content_type_map = {
    ContentType.STRING: DeclarativeMemoryContentType.MESSAGE,
}

declarative_memory_content_type_to_content_type_map = {
    DeclarativeMemoryContentType.MESSAGE: ContentType.STRING,
    DeclarativeMemoryContentType.TEXT: ContentType.STRING,
}


class LongTermMemoryParams(BaseModel):
    """
    Parameters for LongTermMemory.

    Attributes:
        group_id (str):
            Group identifier.
        session_id (str):
            Session identifier.
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
    """

    group_id: str = Field(
        ...,
        description="Group identifier",
    )
    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    vector_graph_store: InstanceOf[VectorGraphStore] = Field(
        ...,
        description="VectorGraphStore instance for storing and retrieving memories",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )


class LongTermMemory:
    _shared_resources: dict[str, Any] = {}

    def __init__(self, params: LongTermMemoryParams):
        self._group_id = params.group_id
        # Note: Things look a bit weird during refactor...
        # Internal session_id is used for external group_id. This is intentional.
        self._declarative_memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=params.group_id,
                vector_graph_store=params.vector_graph_store,
                embedder=params.embedder,
                reranker=params.reranker,
            )
        )

    async def add_episodes(self, episodes: Iterable[Episode]):
        declarative_memory_episodes = [
            DeclarativeMemoryEpisode(
                uuid=episode.uuid,
                timestamp=episode.timestamp,
                source=episode.producer_id,
                content_type=content_type_to_declarative_memory_content_type_map[
                    episode.content_type
                ],
                content=episode.content,
                filterable_properties={
                    key: value
                    for key, value in {
                        "session_id": episode.session_id,
                        "producer_id": episode.producer_id,
                        "produced_for_id": episode.produced_for_id,
                    }.items()
                    if value is not None
                },
                user_metadata=episode.user_metadata,
            )
            for episode in episodes
        ]
        await self._declarative_memory.add_episodes(declarative_memory_episodes)

    async def search(
        self,
        query: str,
        num_episodes_limit: int,
        id_filter: Mapping[str, str] | None = None,
    ):
        if id_filter is None:
            id_filter = {}

        id_filter = {
            key: value for key, value in id_filter.items() if key != "group_id"
        }

        declarative_memory_episodes = await self._declarative_memory.search(
            query,
            max_num_episodes=num_episodes_limit,
            property_filter=dict(id_filter),
        )
        return [
            Episode(
                uuid=declarative_memory_episode.uuid,
                timestamp=declarative_memory_episode.timestamp,
                episode_type="",
                content_type=(
                    declarative_memory_content_type_to_content_type_map[
                        declarative_memory_episode.content_type
                    ]
                ),
                content=declarative_memory_episode.content,
                group_id=self._group_id,
                session_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "session_id", ""
                    ),
                ),
                producer_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "producer_id", ""
                    ),
                ),
                produced_for_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "produced_for_id", ""
                    ),
                ),
                user_metadata=declarative_memory_episode.user_metadata,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def clear(self):
        self._declarative_memory.delete_episodes(
            episode.uuid
            for episode in await self._declarative_memory.get_matching_episodes()
        )

    async def forget_session(self):
        await self._declarative_memory.delete_episodes(
            episode.uuid
            for episode in await self._declarative_memory.get_matching_episodes(
                property_filter={
                    "session_id": self._memory_context.session_id,
                }
            )
        )
