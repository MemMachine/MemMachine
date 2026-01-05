"""Long-term declarative memory coordination."""

from collections.abc import Iterable
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import FilterablePropertyValue
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import ContentType, Episode, EpisodeType
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.reranker import Reranker
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)
from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType as DeclarativeMemoryContentType,
)
from memmachine.episodic_memory.declarative_memory.data_types import (
    Episode as DeclarativeMemoryEpisode,
)


class LongTermMemoryParams(BaseModel):
    """
    Parameters for LongTermMemory.

    Attributes:
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
    message_sentence_chunking: bool = Field(
        False,
        description="Whether to chunk message episodes into sentences for embedding",
    )


class LongTermMemory:
    """High-level facade around the declarative memory store."""

    _FILTERABLE_METADATA_NONE_FLAG = "_filterable_metadata_none"

    def __init__(self, params: LongTermMemoryParams) -> None:
        """Wire up the declarative memory backing store."""
        self._declarative_memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=params.session_id,
                vector_graph_store=params.vector_graph_store,
                embedder=params.embedder,
                reranker=params.reranker,
                message_sentence_chunking=params.message_sentence_chunking,
            ),
        )

    async def add_episodes(self, episodes: Iterable[Episode]) -> None:
        declarative_memory_episodes = [
            DeclarativeMemoryEpisode(
                uid=episode.uid or str(uuid4()),
                timestamp=episode.created_at,
                source=episode.producer_id,
                content_type=LongTermMemory._declarative_memory_content_type_from_episode(
                    episode,
                ),
                content=episode.content,
                filterable_properties=cast(
                    dict[str, FilterablePropertyValue],
                    {
                        key: value
                        for key, value in {
                            "created_at": episode.created_at,
                            "session_key": episode.session_key,
                            "producer_id": episode.producer_id,
                            "producer_role": episode.producer_role,
                            "produced_for_id": episode.produced_for_id,
                            "sequence_num": episode.sequence_num,
                            "episode_type": episode.episode_type.value,
                            "content_type": episode.content_type.value,
                        }.items()
                        if value is not None
                    }
                    | (
                        {
                            LongTermMemory._mangle_filterable_metadata_key(key): value
                            for key, value in (
                                episode.filterable_metadata or {}
                            ).items()
                        }
                        if episode.filterable_metadata is not None
                        else {LongTermMemory._FILTERABLE_METADATA_NONE_FLAG: True}
                    ),
                ),
                user_metadata=episode.metadata,
            )
            for episode in episodes
        ]
        await self._declarative_memory.add_episodes(declarative_memory_episodes)

    async def search(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        score_threshold: float = -float("inf"),
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        scored_episodes = await self.search_scored(
            query,
            num_episodes_limit=num_episodes_limit,
            score_threshold=score_threshold,
            property_filter=property_filter,
        )
        return [episode for _, episode in scored_episodes]

    async def search_scored(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        score_threshold: float = -float("inf"),
        property_filter: FilterExpr | None = None,
    ) -> list[tuple[float, Episode]]:
        scored_declarative_memory_episodes = (
            await self._declarative_memory.search_scored(
                query,
                max_num_episodes=num_episodes_limit,
                property_filter=LongTermMemory._sanitize_property_filter(
                    property_filter
                ),
            )
        )
        return [
            (
                score,
                LongTermMemory._episode_from_declarative_memory_episode(
                    declarative_memory_episode,
                ),
            )
            for score, declarative_memory_episode in (
                scored_declarative_memory_episodes
            )
            if score >= score_threshold
        ]

    async def get_episodes(self, uids: Iterable[str]) -> list[Episode]:
        declarative_memory_episodes = await self._declarative_memory.get_episodes(uids)
        return [
            LongTermMemory._episode_from_declarative_memory_episode(
                declarative_memory_episode,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def get_matching_episodes(
        self,
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        declarative_memory_episodes = (
            await self._declarative_memory.get_matching_episodes(
                property_filter=LongTermMemory._sanitize_property_filter(
                    property_filter
                ),
            )
        )
        return [
            LongTermMemory._episode_from_declarative_memory_episode(
                declarative_memory_episode,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        await self._declarative_memory.delete_episodes(uids)

    async def delete_matching_episodes(
        self,
        property_filter: FilterExpr | None = None,
    ) -> None:
        await self._declarative_memory.delete_episodes(
            episode.uid
            for episode in await self._declarative_memory.get_matching_episodes(
                property_filter=LongTermMemory._sanitize_property_filter(
                    property_filter
                ),
            )
        )

    async def close(self) -> None:
        # Do nothing.
        pass

    @staticmethod
    def _declarative_memory_content_type_from_episode(
        episode: Episode,
    ) -> DeclarativeMemoryContentType:
        match episode.episode_type:
            case EpisodeType.MESSAGE:
                match episode.content_type:
                    case ContentType.STRING:
                        return DeclarativeMemoryContentType.MESSAGE
                    case _:
                        return DeclarativeMemoryContentType.TEXT
            case _:
                match episode.content_type:
                    case ContentType.STRING:
                        return DeclarativeMemoryContentType.TEXT
                    case _:
                        return DeclarativeMemoryContentType.TEXT

    @staticmethod
    def _episode_from_declarative_memory_episode(
        declarative_memory_episode: DeclarativeMemoryEpisode,
    ) -> Episode:
        return Episode(
            uid=declarative_memory_episode.uid,
            sequence_num=cast(
                "int",
                declarative_memory_episode.filterable_properties.get("sequence_num", 0),
            ),
            session_key=cast(
                "str",
                declarative_memory_episode.filterable_properties.get("session_key", ""),
            ),
            episode_type=EpisodeType(
                cast(
                    "str",
                    declarative_memory_episode.filterable_properties.get(
                        "episode_type",
                        "",
                    ),
                ),
            ),
            content_type=ContentType(
                cast(
                    "str",
                    declarative_memory_episode.filterable_properties.get(
                        "content_type",
                        "",
                    ),
                ),
            ),
            content=declarative_memory_episode.content,
            created_at=declarative_memory_episode.timestamp,
            producer_id=cast(
                "str",
                declarative_memory_episode.filterable_properties.get("producer_id", ""),
            ),
            producer_role=cast(
                "str",
                declarative_memory_episode.filterable_properties.get(
                    "producer_role",
                    "",
                ),
            ),
            produced_for_id=cast(
                "str | None",
                declarative_memory_episode.filterable_properties.get("produced_for_id"),
            ),
            filterable_metadata={
                LongTermMemory._demangle_filterable_metadata_key(key): value
                for key, value in declarative_memory_episode.filterable_properties.items()
                if LongTermMemory._is_mangled_filterable_metadata_key(key)
            }
            if LongTermMemory._FILTERABLE_METADATA_NONE_FLAG
            not in declarative_memory_episode.filterable_properties
            else None,
            metadata=declarative_memory_episode.user_metadata,
        )

    _MANGLE_FILTERABLE_METADATA_KEY_PREFIX = "metadata."

    @staticmethod
    def _mangle_filterable_metadata_key(key: str) -> str:
        return LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX + key

    @staticmethod
    def _demangle_filterable_metadata_key(mangled_key: str) -> str:
        return mangled_key.removeprefix(
            LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX
        )

    @staticmethod
    def _is_mangled_filterable_metadata_key(candidate_key: str) -> bool:
        return candidate_key.startswith(
            LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX
        )

    @staticmethod
    def _sanitize_property_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        if property_filter is None:
            return None

        return LongTermMemory._sanitize_filter_expr(property_filter)

    @staticmethod
    def _sanitize_filter_expr(expr: FilterExpr) -> FilterExpr:
        if isinstance(expr, FilterComparison):
            if expr.field.startswith("m."):
                sanitized_field = LongTermMemory._mangle_filterable_metadata_key(
                    expr.field.removeprefix("m.")
                )
            else:
                sanitized_field = expr.field
            return FilterComparison(
                field=sanitized_field,
                op=expr.op,
                value=expr.value,
            )
        if isinstance(expr, FilterAnd):
            return FilterAnd(
                left=LongTermMemory._sanitize_filter_expr(expr.left),
                right=LongTermMemory._sanitize_filter_expr(expr.right),
            )
        if isinstance(expr, FilterOr):
            return FilterOr(
                left=LongTermMemory._sanitize_filter_expr(expr.left),
                right=LongTermMemory._sanitize_filter_expr(expr.right),
            )
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
