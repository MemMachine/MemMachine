"""Core MemMachine orchestration logic."""

import asyncio
import logging
from asyncio import Task
from collections.abc import Coroutine
from typing import Any, Final, Protocol, cast

from pydantic import BaseModel, InstanceOf, ValidationError

from memmachine.common.api import MemoryType
from memmachine.common.configuration import Configuration
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConf,
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.episode_store import Episode, EpisodeEntry, EpisodeIdT
from memmachine.common.errors import ConfigurationError, SessionNotFoundError
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
    parse_filter,
    to_property_filter,
)
from memmachine.common.resource_manager.resource_manager import ResourceManagerImpl
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.episodic_memory import EpisodicMemory
from memmachine.semantic_memory.semantic_model import FeatureIdT, SemanticFeature
from memmachine.semantic_memory.semantic_session_manager import IsolationType

logger = logging.getLogger(__name__)


ALL_MEMORY_TYPES: Final[list[MemoryType]] = list(MemoryType)


class MemMachine:
    """MemMachine class."""

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

    def __init__(
        self, conf: Configuration, resources: ResourceManagerImpl | None = None
    ) -> None:
        """Create a MemMachine using the provided configuration."""
        self._conf = conf
        if resources is not None:
            self._resources = resources
        else:
            self._resources = ResourceManagerImpl(conf)
        self._initialize_default_episodic_configuration()
        self._started = False

    def _initialize_default_episodic_configuration(self) -> None:
        # initialize the default value for episodic memory configuration
        # Can not put the logic into the data type

        if self._conf.episodic_memory is None:
            self._conf.episodic_memory = EpisodicMemoryConfPartial()
            self._conf.episodic_memory.enabled = False
        if self._conf.episodic_memory.long_term_memory is None:
            self._conf.episodic_memory.long_term_memory = LongTermMemoryConfPartial()
            self._conf.episodic_memory.long_term_memory_enabled = False
        if self._conf.episodic_memory.short_term_memory is None:
            self._conf.episodic_memory.short_term_memory = ShortTermMemoryConfPartial()
            self._conf.episodic_memory.short_term_memory_enabled = False
        if self._conf.episodic_memory.long_term_memory.embedder is None:
            self._conf.episodic_memory.long_term_memory.embedder = (
                self._conf.default_long_term_memory_embedder
            )
        if self._conf.episodic_memory.long_term_memory.reranker is None:
            self._conf.episodic_memory.long_term_memory.reranker = (
                self._conf.default_long_term_memory_reranker
            )
        if self._conf.episodic_memory.short_term_memory.llm_model is None:
            self._conf.episodic_memory.short_term_memory.llm_model = "gpt-4.1"
        if self._conf.episodic_memory.short_term_memory.summary_prompt_system is None:
            self._conf.episodic_memory.short_term_memory.summary_prompt_system = (
                self._conf.prompt.episode_summary_system_prompt
            )
        if self._conf.episodic_memory.short_term_memory.summary_prompt_user is None:
            self._conf.episodic_memory.short_term_memory.summary_prompt_user = (
                self._conf.prompt.episode_summary_user_prompt
            )
        if self._conf.episodic_memory.long_term_memory.vector_graph_store is None:
            self._conf.episodic_memory.long_term_memory.vector_graph_store = (
                "default_store"
            )

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

        semantic_service = await self._resources.get_semantic_service()
        await semantic_service.start()

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False

        semantic_service = await self._resources.get_semantic_service()
        await semantic_service.stop()

        await self._resources.close()

    def _with_default_episodic_memory_conf(
        self,
        *,
        user_conf: EpisodicMemoryConfPartial | None = None,
        session_key: str,
    ) -> EpisodicMemoryConf:
        # Get default prompts from config, with fallbacks
        try:
            if user_conf is None:
                user_conf = EpisodicMemoryConfPartial()
            user_conf.session_key = session_key
            episodic_conf = user_conf.merge(self._conf.episodic_memory)
            if episodic_conf.long_term_memory is not None:
                if episodic_conf.long_term_memory.embedder is not None:
                    self._conf.check_embedder(episodic_conf.long_term_memory.embedder)
                if episodic_conf.long_term_memory.reranker is not None:
                    self._conf.check_reranker(episodic_conf.long_term_memory.reranker)
        except ValidationError as e:
            logger.exception(
                "Faield to merge configuration: %s, %s",
                str(user_conf),
                str(self._conf.episodic_memory),
            )
            raise ConfigurationError("Failed to merge configuration") from e
        return episodic_conf

    async def create_session(
        self,
        session_key: str,
        *,
        description: str = "",
        user_conf: EpisodicMemoryConfPartial | None = None,
    ) -> SessionDataManager.SessionInfo:
        """Create a new session."""
        episodic_memory_conf = self._with_default_episodic_memory_conf(
            user_conf=user_conf,
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
        ret = await self.get_session(session_key=session_key)
        if ret is None:
            raise RuntimeError(f"Failed to create session {session_key}")
        return ret

    async def get_session(
        self, session_key: str
    ) -> SessionDataManager.SessionInfo | None:
        session_data_manager = await self._resources.get_session_data_manager()
        return await session_data_manager.get_session_info(session_key)

    async def delete_session(self, session_data: SessionData) -> None:
        session = await self.get_session(session_data.session_key)
        if session is None:
            raise SessionNotFoundError(session_data.session_key)

        async def _delete_episode_store() -> None:
            episode_store = await self._resources.get_episode_storage()
            session_filter = FilterComparison(
                field="session_key",
                op="=",
                value=session_data.session_key,
            )
            await episode_store.delete_episode_messages(
                filter_expr=session_filter,
            )

        async def _delete_episodic_memory() -> None:
            episodic_memory_manager = (
                await self._resources.get_episodic_memory_manager()
            )

            await episodic_memory_manager.delete_episodic_session(
                session_key=session_data.session_key
            )

        async def _delete_semantic_memory() -> None:
            semantic_memory_manager = (
                await self._resources.get_semantic_session_manager()
            )
            await asyncio.gather(
                semantic_memory_manager.delete_feature_set(
                    session_data=session_data,
                    memory_type=[IsolationType.SESSION],
                ),
                semantic_memory_manager.delete_messages(session_data=session_data),
            )

        tasks = [
            _delete_episode_store(),
            _delete_episodic_memory(),
            _delete_semantic_memory(),
        ]

        await asyncio.gather(*tasks)

    async def search_sessions(
        self,
        search_filter: FilterExpr | None = None,
    ) -> list[str]:
        session_data_manager = await self._resources.get_session_data_manager()
        return await session_data_manager.get_sessions(
            filters=cast(dict[str, object] | None, to_property_filter(search_filter))
        )

    @staticmethod
    def _merge_filter_exprs(
        left: FilterExpr | None,
        right: FilterExpr | None,
    ) -> FilterExpr | None:
        if left is None:
            return right
        if right is None:
            return left
        return FilterAnd(left=left, right=right)

    async def add_episodes(
        self,
        session_data: InstanceOf[SessionData],
        episode_entries: list[EpisodeEntry],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
    ) -> list[EpisodeIdT]:
        episode_storage = await self._resources.get_episode_storage()
        episodes = await episode_storage.add_episodes(
            session_data.session_key,
            episode_entries,
        )
        episode_ids = [e.uid for e in episodes]

        tasks = []

        if MemoryType.Episodic in target_memories:
            episodic_memory_manager = (
                await self._resources.get_episodic_memory_manager()
            )
            async with episodic_memory_manager.open_or_create_episodic_memory(
                session_key=session_data.session_key,
                description="",
                episodic_memory_config=self._with_default_episodic_memory_conf(
                    session_key=session_data.session_key
                ),
                metadata={},
            ) as episodic_session:
                tasks.append(episodic_session.add_memory_episodes(episodes))

        if MemoryType.Semantic in target_memories:
            semantic_session_manager = (
                await self._resources.get_semantic_session_manager()
            )
            tasks.append(
                semantic_session_manager.add_message(
                    episode_ids=episode_ids,
                    session_data=session_data,
                )
            )

        await asyncio.gather(*tasks)
        return episode_ids

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
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        search_filter: FilterExpr | None = None,
    ) -> EpisodicMemory.QueryResponse | None:
        episodic_memory_manager = await self._resources.get_episodic_memory_manager()

        async with episodic_memory_manager.open_or_create_episodic_memory(
            session_key=session_data.session_key,
            description="",
            episodic_memory_config=self._with_default_episodic_memory_conf(
                session_key=session_data.session_key
            ),
            metadata={},
        ) as episodic_session:
            response = await episodic_session.query_memory(
                query=query,
                limit=limit,
                expand_context=expand_context,
                score_threshold=score_threshold,
                property_filter=search_filter,
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
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        search_filter: str | None = None,
    ) -> SearchResponse:
        episodic_task: Task | None = None
        semantic_task: Task | None = None

        property_filter = parse_filter(search_filter) if search_filter else None
        if MemoryType.Episodic in target_memories:
            episodic_task = asyncio.create_task(
                self._search_episodic_memory(
                    session_data=session_data,
                    query=query,
                    limit=limit,
                    expand_context=expand_context,
                    score_threshold=score_threshold,
                    search_filter=property_filter,
                )
            )

        if MemoryType.Semantic in target_memories:
            semantic_session = await self._resources.get_semantic_session_manager()
            semantic_task = asyncio.create_task(
                semantic_session.search(
                    memory_type=[IsolationType.SESSION],
                    message=query,
                    session_data=session_data,
                    limit=limit,
                    search_filter=property_filter,
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
        page_size: int | None = None,
        page_num: int | None = None,
    ) -> ListResults:
        search_filter_expr = parse_filter(search_filter) if search_filter else None

        episodic_task: Task | None = None
        semantic_task: Task | None = None

        if MemoryType.Episodic in target_memories:
            episode_storage = await self._resources.get_episode_storage()
            session_filter = FilterComparison(
                field="session_key",
                op="=",
                value=session_data.session_key,
            )
            combined_filter = self._merge_filter_exprs(
                session_filter,
                search_filter_expr,
            )
            episodic_task = asyncio.create_task(
                episode_storage.get_episode_messages(
                    page_size=page_size,
                    page_num=page_num,
                    filter_expr=combined_filter,
                )
            )

        if MemoryType.Semantic in target_memories:
            semantic_session = await self._resources.get_semantic_session_manager()
            semantic_task = asyncio.create_task(
                semantic_session.get_set_features(
                    session_data=session_data,
                    search_filter=search_filter_expr,
                    page_size=page_size,
                    page_num=page_num,
                )
            )

        episodic_result = await episodic_task if episodic_task else None
        semantic_result = await semantic_task if semantic_task else None

        return MemMachine.ListResults(
            episodic_memory=episodic_result,
            semantic_memory=semantic_result,
        )

    async def episodes_count(
        self,
        session_data: InstanceOf[SessionData],
        *,
        search_filter: str | None = None,
    ) -> int:
        """Count the number of episodes in the session that matches the search filter."""
        episode_storage = await self._resources.get_episode_storage()

        session_filter = FilterComparison(
            field="session_key",
            op="=",
            value=session_data.session_key,
        )

        search_filter_expr = parse_filter(search_filter) if search_filter else None
        combined_filter = self._merge_filter_exprs(session_filter, search_filter_expr)

        return await episode_storage.get_episode_messages_count(
            filter_expr=combined_filter,
        )

    async def delete_episodes(
        self,
        episode_ids: list[EpisodeIdT],
        session_data: InstanceOf[SessionData] | None = None,
    ) -> None:
        episode_storage = await self._resources.get_episode_storage()
        semantic_service = await self._resources.get_semantic_service()

        tasks: list[Coroutine[Any, Any, Any]] = []

        if session_data is not None:
            episodic_memory_manager = (
                await self._resources.get_episodic_memory_manager()
            )
            async with episodic_memory_manager.open_episodic_memory(
                session_data.session_key
            ) as episodic_session:
                t = episodic_session.delete_episodes(episode_ids)
                tasks.append(t)

        tasks.append(episode_storage.delete_episodes(episode_ids))
        tasks.append(semantic_service.delete_history(episode_ids))
        await asyncio.gather(*tasks)

    async def delete_features(
        self,
        feature_ids: list[FeatureIdT],
    ) -> None:
        semantic_session = await self._resources.get_semantic_session_manager()
        await semantic_session.delete_features(feature_ids)
