"""Core MemMachine orchestration logic."""

import asyncio
import logging
from asyncio import Task
from collections.abc import Coroutine
from enum import Enum
from typing import Any, Final, Protocol, cast

from pydantic import BaseModel, InstanceOf, JsonValue

from memmachine.common.configuration import Configuration
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConf,
    LongTermMemoryConf,
    ShortTermMemoryConf,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
    parse_filter,
    to_property_filter,
)
from memmachine.common.resource_manager.resource_manager import ResourceManagerImpl
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.episode_store.episode_model import Episode, EpisodeEntry, EpisodeIdT
from memmachine.episodic_memory import EpisodicMemory
from memmachine.main.memmachine_errors import (
    DefaultEmbedderNotConfiguredError,
    DefaultRerankerNotConfiguredError,
    EmbedderNotFoundError,
    RerankerNotFoundError,
)
from memmachine.semantic_memory.semantic_model import FeatureIdT, SemanticFeature
from memmachine.semantic_memory.semantic_session_resource import IsolationType
from memmachine.server.api_v2.spec import AddMemoryResult

logger = logging.getLogger(__name__)


class SessionData(Protocol):
    """Protocol describing session-scoped metadata used by memories."""

    @property
    def session_key(self) -> str:
        """Unique session identifier."""
        raise NotImplementedError

    @property
    def user_profile_id(self) -> str | None:
        raise NotImplementedError

    @property
    def role_profile_id(self) -> str | None:
        raise NotImplementedError

    @property
    def session_id(self) -> str | None:
        raise NotImplementedError


class MemoryType(Enum):
    """MemMachine type."""

    Semantic = 0
    Episodic = 1


ALL_MEMORY_TYPES: Final[list[MemoryType]] = list(MemoryType)


class MemMachine:
    """MemMachine class."""

    def __init__(self, conf: Configuration, resources: ResourceManagerImpl) -> None:
        """Create a MemMachine using the provided configuration."""
        self._conf = conf
        self._resources = resources

    async def start(self) -> None:
        await self._resources.build()

        semantic_service = await self._resources.get_semantic_service()
        await semantic_service.start()

    async def stop(self) -> None:
        semantic_service = await self._resources.get_semantic_service()
        await semantic_service.stop()

        await self._resources.close()

    async def session_exists(self, session_key: str) -> bool:
        # Check if session exists
        try:
            session_info = await self.get_session(session_key=session_key)
        except RuntimeError:
            return False
        return session_info is not None

    def _with_default_episodic_memory_conf(
        self,
        *,
        embedder_name: str | None = None,
        reranker_name: str | None = None,
        session_key: str,
    ) -> EpisodicMemoryConf:
        # Get default prompts from config, with fallbacks
        short_term = self._conf.episodic_memory.short_term_memory
        summary_prompt_system = (
            short_term.summary_prompt_system
            if short_term and short_term.summary_prompt_system
            else "You are a helpful assistant."
        )
        summary_prompt_user = (
            short_term.summary_prompt_user
            if short_term and short_term.summary_prompt_user
            else "Based on the following episodes: {episodes}, and the previous summary: {summary}, please update the summary. Keep it under {max_length} characters."
        )

        # Get default embedder and reranker from config
        long_term = self._conf.episodic_memory.long_term_memory

        if (
            (embedder_name is None or embedder_name == "default")
            and long_term
            and long_term.embedder
        ):
            target_embedder = long_term.embedder
        elif embedder_name is not None:
            target_embedder = embedder_name
        else:
            raise DefaultEmbedderNotConfiguredError

        if (
            (reranker_name is None or reranker_name == "default")
            and long_term
            and long_term.reranker
        ):
            target_reranker = long_term.reranker
        elif reranker_name is not None:
            target_reranker = reranker_name
        else:
            raise DefaultRerankerNotConfiguredError

        if not self._conf.resources.rerankers.contains_reranker(target_reranker):
            raise RerankerNotFoundError(target_reranker)

        if not self._conf.resources.embedders.contains_embedder(target_embedder):
            raise EmbedderNotFoundError(target_embedder)

        target_vector_store = (
            long_term.vector_graph_store
            if long_term and long_term.vector_graph_store
            else "default_store"
        )

        target_short_llm_model = (
            short_term.llm_model if short_term and short_term.llm_model else "gpt-4.1"
        )

        return EpisodicMemoryConf(
            session_key=session_key,
            long_term_memory=LongTermMemoryConf(
                session_id=session_key,
                vector_graph_store=target_vector_store,
                embedder=target_embedder,
                reranker=target_reranker,
            ),
            short_term_memory=ShortTermMemoryConf(
                session_key=session_key,
                llm_model=target_short_llm_model,
                summary_prompt_system=summary_prompt_system,
                summary_prompt_user=summary_prompt_user,
            ),
            long_term_memory_enabled=True,
            short_term_memory_enabled=True,
            enabled=True,
        )

    async def create_session(
        self,
        session_key: str,
        *,
        description: str = "",
        embedder_name: str | None = None,
        reranker_name: str | None = None,
    ) -> SessionDataManager.SessionInfo:
        """Create a new session."""
        episodic_memory_conf = self._with_default_episodic_memory_conf(
            embedder_name=embedder_name,
            reranker_name=reranker_name,
            session_key=session_key,
        )

        session_data_manager = await self._resources.get_session_data_manager()
        await session_data_manager.create_new_session(
            session_key=session_key,
            configuration={},
            param=episodic_memory_conf,
            description=description,
            metadata={},
        )
        return await self.get_session(session_key=session_key)

    async def get_session(
        self, session_key: str
    ) -> SessionDataManager.SessionInfo | None:
        session_data_manager = await self._resources.get_session_data_manager()
        return await session_data_manager.get_session_info(session_key)

    async def delete_session(self, session_key: str) -> None:
        session_data_manager = await self._resources.get_session_data_manager()
        await session_data_manager.delete_session(session_key)

    async def search_sessions(
        self,
        search_filter: FilterExpr | None = None,
    ) -> list[str]:
        session_data_manager = await self._resources.get_session_data_manager()
        return await session_data_manager.get_sessions(
            filters=cast(dict[str, object] | None, to_property_filter(search_filter))
        )

    async def add_episodes(
        self,
        session_data: InstanceOf[SessionData],
        episode_entries: list[EpisodeEntry],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
    ) -> list[AddMemoryResult]:
        if not await self.session_exists(session_data.session_key):
            await self.create_session(
                session_data.session_key,
                description="",
                embedder_name="default",
                reranker_name="default",
            )

        episode_storage = await self._resources.get_episode_storage()
        episodes = await episode_storage.add_episodes(
            session_data.session_key,
            episode_entries,
        )
        ret = [AddMemoryResult(uid=episode.uid) for episode in episodes]

        semantic_manager = await self._resources.get_semantic_manager()
        semantic_session_data = (
            semantic_manager.simple_semantic_session_id_manager.generate_session_data(
                session_id=session_data.session_id,
            )
        )

        tasks = []

        if MemoryType.Episodic in target_memories:
            episodic_memory_manager = (
                await self._resources.get_episodic_memory_manager()
            )
            async with episodic_memory_manager.open_episodic_memory(
                session_data.session_key
            ) as episodic_session:
                tasks.append(episodic_session.add_memory_episodes(episodes))

        if MemoryType.Semantic in target_memories:
            episode_ids = [e.uid for e in episodes]
            semantic_session_manager = (
                await self._resources.get_semantic_session_manager()
            )
            tasks.append(
                semantic_session_manager.add_message(
                    memory_type=[IsolationType.SESSION],
                    episode_ids=episode_ids,
                    session_data=semantic_session_data,
                )
            )

        await asyncio.gather(*tasks)
        return ret

    class SearchResponse(BaseModel):
        """Aggregated search results across memory types."""

        episodic_memory: EpisodicMemory.QueryResponse | None = None
        semantic_memory: list[SemanticFeature] | None = None

    async def _search_episodic_memory(
        self,
        *,
        session_data: InstanceOf[SessionData],
        query: str,
        limit: int | None = None,
        search_filter: str | None = None,
    ) -> EpisodicMemory.QueryResponse | None:
        episodic_memory_manager = await self._resources.get_episodic_memory_manager()

        async with episodic_memory_manager.open_episodic_memory(
            session_data.session_key
        ) as episodic_session:
            response = await episodic_session.query_memory(
                query=query,
                limit=limit,
                property_filter=to_property_filter(parse_filter(search_filter)),
            )

        return response

    async def query_search(
        self,
        session_data: InstanceOf[SessionData],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
        query: str,
        limit: int
        | None = None,  # TODO: Define if limit is per memory or is global limit
        search_filter: str | None = None,
    ) -> SearchResponse:
        episodic_task: Task | None = None
        semantic_task: Task | None = None

        semantic_manager = await self._resources.get_semantic_manager()
        semantic_session_data = (
            semantic_manager.simple_semantic_session_id_manager.generate_session_data(
                session_id=session_data.session_id,
            )
        )

        if not await self.session_exists(session_data.session_key):
            raise RuntimeError(
                f"No session info found for session {session_data.session_key}"
            )

        if MemoryType.Episodic in target_memories:
            episodic_task = asyncio.create_task(
                self._search_episodic_memory(
                    session_data=session_data,
                    query=query,
                    limit=limit,
                    search_filter=search_filter,
                )
            )

        if MemoryType.Semantic in target_memories:
            semantic_session = await self._resources.get_semantic_session_manager()
            semantic_task = asyncio.create_task(
                semantic_session.search(
                    memory_type=[IsolationType.SESSION],
                    message=query,
                    session_data=semantic_session_data,
                    limit=limit,
                    search_filter=parse_filter(search_filter),
                )
            )

        return MemMachine.SearchResponse(
            episodic_memory=await episodic_task if episodic_task else None,
            semantic_memory=await semantic_task if semantic_task else None,
        )

    class ListResults(BaseModel):
        """Result payload for list-style memory queries."""

        episodic_memory: list[Episode] | None = None
        semantic_memory: list[SemanticFeature] | None = None

    async def list_search(
        self,
        session_data: InstanceOf[SessionData],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
        search_filter: str | None = None,
        limit: int | None = None,
    ) -> ListResults:
        search_filter_expr = parse_filter(search_filter) if search_filter else None

        episodic_task: Task | None = None
        semantic_task: Task | None = None

        if MemoryType.Episodic in target_memories:
            episode_storage = await self._resources.get_episode_storage()
            # TODO: modify episode store filter
            episodic_task = asyncio.create_task(
                episode_storage.get_episode_messages(
                    limit=limit,
                    session_keys=[session_data.session_key],
                    metadata=cast(
                        dict[str, JsonValue] | None,
                        to_property_filter(search_filter_expr),
                    ),
                )
            )

        if MemoryType.Semantic in target_memories:
            semantic_session = await self._resources.get_semantic_session_manager()
            semantic_task = asyncio.create_task(
                semantic_session.get_set_features(
                    session_data=session_data,
                    search_filter=search_filter_expr,
                    limit=limit,
                )
            )

        episodic_result = await episodic_task if episodic_task else None
        semantic_result = await semantic_task if semantic_task else None

        return MemMachine.ListResults(
            episodic_memory=episodic_result,
            semantic_memory=semantic_result,
        )

    async def delete_episodes(
        self,
        episode_ids: list[EpisodeIdT],
        session_data: InstanceOf[SessionData] | None = None,
    ) -> None:
        episode_storage = await self._resources.get_episode_storage()

        tasks: list[Coroutine[Any, Any, Any]] = [
            episode_storage.delete_episodes(episode_ids),
        ]

        if session_data is not None:
            episodic_memory_manager = (
                await self._resources.get_episodic_memory_manager()
            )
            async with episodic_memory_manager.open_episodic_memory(
                session_data.session_key
            ) as episodic_session:
                t = episodic_session.delete_episodes(episode_ids)
                tasks.append(t)

        await asyncio.gather(*tasks)

    async def delete_features(
        self,
        feature_ids: list[FeatureIdT],
    ) -> None:
        semantic_session = await self._resources.get_semantic_session_manager()
        await semantic_session.delete_features(feature_ids)

    async def delete_filtered(
        self,
        *,
        session_data: InstanceOf[SessionData],
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
        delete_filter: str | None = None,
    ) -> None:
        raise NotImplementedError
