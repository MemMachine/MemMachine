"""Core MemMachine orchestration logic."""

import asyncio
import logging
from asyncio import Task
from collections.abc import Callable, Coroutine, Iterable, Mapping
from typing import Any, Final, Protocol, cast

from pydantic import BaseModel, InstanceOf, JsonValue, ValidationError

from memmachine.common.api import MemoryType
from memmachine.common.configuration import Configuration
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConf,
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.episode_store import Episode, EpisodeEntry, EpisodeIdT
from memmachine.common.errors import (
    ConfigurationError,
    ResourceNotReadyError,
    SessionNotFoundError,
)
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
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_model import (
    CategoryIdT,
    FeatureIdT,
    SemanticCategory,
    SemanticFeature,
    SetIdT,
    SetTypeEntry,
    TagIdT,
)
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager

logger = logging.getLogger(__name__)


ALL_MEMORY_TYPES: Final[list[MemoryType]] = list(MemoryType)


class MemMachine:
    """MemMachine class."""

    class SessionData(Protocol):
        """Protocol describing session-scoped context used by memories."""

        @property
        def org_id(self) -> str: ...

        @property
        def project_id(self) -> str: ...

        @property
        def session_key(self) -> str: ...

    def __init__(
        self, conf: Configuration, resources: ResourceManagerImpl | None = None
    ) -> None:
        """
        Create a MemMachine using the provided configuration.

        Args:
            conf: Application configuration.
            resources: Optional resource manager override.

        Returns:
            None.

        """
        self._conf = conf
        if resources is not None:
            self._resources = resources
        else:
            self._resources = ResourceManagerImpl(conf)
        self._initialize_default_episodic_configuration()
        self._started = False

    def _initialize_default_episodic_configuration(self) -> None:
        """
        Initialize missing episodic memory configuration defaults.

        Returns:
            None.

        """
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

        # Always resolve defaults so that merge() can produce valid full
        # configs.  _resolve_*_defaults() will gracefully disable the
        # subsystem if required resources are missing.
        self._resolve_ltm_defaults()
        self._resolve_stm_defaults()

    def _resolve_ltm_defaults(self) -> None:
        """Resolve long-term memory defaults, disabling if resources are unavailable."""
        ltm = self._conf.episodic_memory.long_term_memory
        assert ltm is not None

        ltm.embedder = self._resolve_ltm_resource_default(
            current_value=ltm.embedder,
            default_getter=lambda: self._conf.default_long_term_memory_embedder,
            missing_warning=(
                "No default embedder configured; disabling long-term episodic memory."
            ),
        )
        if ltm.embedder is None:
            return

        ltm.reranker = self._resolve_ltm_resource_default(
            current_value=ltm.reranker,
            default_getter=lambda: self._conf.default_long_term_memory_reranker,
            missing_warning=(
                "No default reranker configured; disabling long-term episodic memory."
            ),
        )
        if ltm.reranker is None:
            return

        if ltm.vector_graph_store is None:
            ltm.vector_graph_store = "default_store"

        if ltm.llm_model is None:
            ltm.llm_model = self._resolve_ltm_llm_model_fallback()
            if ltm.llm_model is None:
                self._disable_long_term_memory(
                    "No default language model configured; disabling long-term episodic memory."
                )
                return

    def _disable_long_term_memory(self, warning_message: str) -> None:
        """Disable long-term memory and emit a warning message."""
        logger.warning(warning_message)
        self._conf.episodic_memory.long_term_memory_enabled = False

    def _resolve_ltm_resource_default(
        self,
        *,
        current_value: str | None,
        default_getter: Callable[[], str],
        missing_warning: str,
    ) -> str | None:
        """Return configured value or fallback default; disable long-term memory on failure."""
        if current_value is not None:
            return current_value
        try:
            return default_getter()
        except (ConfigurationError, Exception):
            self._disable_long_term_memory(missing_warning)
            return None

    def _resolve_ltm_llm_model_fallback(self) -> str | None:
        """Resolve an LLM for long-term memory from available configured candidates."""
        candidates: list[str] = []
        stm = self._conf.episodic_memory.short_term_memory
        if stm is not None and stm.llm_model is not None:
            candidates.append(stm.llm_model)

        semantic_llm = self._conf.semantic_memory.llm_model
        if semantic_llm is not None:
            candidates.append(semantic_llm)

        for llm_name in candidates:
            if self._conf.resources.language_models.contains_language_model(llm_name):
                return llm_name

        return None

    def _resolve_stm_defaults(self) -> None:
        """Resolve short-term memory defaults."""
        stm = self._conf.episodic_memory.short_term_memory
        assert stm is not None

        if stm.llm_model is None:
            stm.llm_model = "gpt-4.1"
        if stm.summary_prompt_system is None:
            stm.summary_prompt_system = self._conf.prompt.episode_summary_system_prompt
        if stm.summary_prompt_user is None:
            stm.summary_prompt_user = self._conf.prompt.episode_summary_user_prompt

    async def start(self) -> None:
        """
        Start MemMachine background services.

        Returns:
            None.

        """
        if self._started:
            return
        self._started = True

        if self._conf.semantic_memory.enabled:
            semantic_service = await self._resources.get_semantic_service()
            await semantic_service.start()
        else:
            logger.info(
                "Semantic memory is disabled; skipping semantic service startup."
            )

    async def stop(self) -> None:
        """
        Stop MemMachine background services and release resources.

        Returns:
            None.

        """
        if not self._started:
            return
        self._started = False

        if self._conf.semantic_memory.enabled:
            semantic_service = await self._resources.get_semantic_service()
            await semantic_service.stop()

        await self._resources.close()

    @property
    def resource_manager(self) -> ResourceManagerImpl:
        """Return the resource manager."""
        return self._resources

    def _with_default_episodic_memory_conf(
        self,
        *,
        user_conf: EpisodicMemoryConfPartial | None = None,
        session_key: str,
    ) -> EpisodicMemoryConf:
        """
        Merge per-session episodic config with defaults.

        Args:
            user_conf: Optional episodic configuration overrides.
            session_key: Session key to associate with the config.

        Returns:
            The resolved episodic memory configuration.

        Raises:
            ConfigurationError: If the merged configuration is invalid.

        """
        # Get default prompts from config, with fallbacks
        try:
            if not self._conf.episodic_memory.enabled:
                raise ResourceNotReadyError("Episodic memory is disabled")
            if user_conf is None:
                user_conf = EpisodicMemoryConfPartial()
            user_conf.session_key = session_key
            episodic_conf = user_conf.merge(self._conf.episodic_memory)
            if episodic_conf.long_term_memory is not None:
                if episodic_conf.long_term_memory.embedder is not None:
                    self._conf.check_embedder(episodic_conf.long_term_memory.embedder)
                if episodic_conf.long_term_memory.reranker is not None:
                    self._conf.check_reranker(episodic_conf.long_term_memory.reranker)
                if episodic_conf.long_term_memory.llm_model is not None:
                    self._conf.check_llm_model(episodic_conf.long_term_memory.llm_model)
        except ValidationError as e:
            logger.exception(
                "Faield to merge configuration: %s, %s",
                str(user_conf),
                str(self._conf.episodic_memory),
            )
            raise ConfigurationError("Failed to merge configuration") from e
        return episodic_conf

    @staticmethod
    def _disabled_episodic_conf(session_key: str) -> EpisodicMemoryConf:
        """Return a minimal episodic config with all memory disabled."""
        return EpisodicMemoryConf(
            session_key=session_key,
            long_term_memory=None,
            short_term_memory=None,
            long_term_memory_enabled=False,
            short_term_memory_enabled=False,
            enabled=False,
        )

    async def create_session(
        self,
        session_key: str,
        *,
        description: str = "",
        user_conf: EpisodicMemoryConfPartial | None = None,
    ) -> SessionDataManager.SessionInfo:
        """
        Create and persist a new session.

        Args:
            session_key: Unique identifier for the session.
            description: Optional human-readable session description.
            user_conf: Optional episodic-memory configuration overrides.

        Returns:
            The created session info.

        """
        if not self._conf.episodic_memory.enabled:
            episodic_memory_conf = self._disabled_episodic_conf(session_key)
        else:
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
        """
        Fetch stored session info.

        Args:
            session_key: Unique identifier for the session.

        Returns:
            The session info, or `None` if it does not exist.

        """
        session_data_manager = await self._resources.get_session_data_manager()
        return await session_data_manager.get_session_info(session_key)

    async def delete_session(self, session_data: SessionData) -> None:
        """
        Delete all data associated with a session.

        Args:
            session_data: Session context providing the session key.

        Returns:
            None.

        Raises:
            SessionNotFoundError: If the session does not exist.

        """
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
                ),
                semantic_memory_manager.delete_all_project_messages(
                    session_data=session_data
                ),
            )

        tasks = [_delete_episode_store()]
        if self._conf.episodic_memory.enabled:
            tasks.append(_delete_episodic_memory())
        if self._conf.semantic_memory.enabled:
            tasks.append(_delete_semantic_memory())

        await asyncio.gather(*tasks)

    async def search_sessions(
        self,
        search_filter: FilterExpr | None = None,
    ) -> list[str]:
        """
        List session keys matching a filter.

        Args:
            search_filter: Optional property filter expression.

        Returns:
            Session keys matching the filter.

        """
        session_data_manager = await self._resources.get_session_data_manager()
        return await session_data_manager.get_sessions(
            filters=cast(dict[str, object] | None, to_property_filter(search_filter))
        )

    async def update_session_episodic_config(
        self,
        session_key: str,
        enabled: bool | None = None,
        long_term_memory_enabled: bool | None = None,
        short_term_memory_enabled: bool | None = None,
    ) -> None:
        """
        Update episodic memory configuration flags for a session.

        This allows enabling/disabling episodic memory, long-term memory,
        and short-term memory independently for a specific session.

        Args:
            session_key: The session key to update.
            enabled: Whether episodic memory is enabled overall.
            long_term_memory_enabled: Whether long-term memory is enabled.
            short_term_memory_enabled: Whether short-term memory is enabled.

        Raises:
            SessionNotFoundError: If the session does not exist.

        """
        session_data_manager = await self._resources.get_session_data_manager()
        await session_data_manager.update_session_episodic_config(
            session_key=session_key,
            enabled=enabled,
            long_term_memory_enabled=long_term_memory_enabled,
            short_term_memory_enabled=short_term_memory_enabled,
        )

    @staticmethod
    def _merge_filter_exprs(
        left: FilterExpr | None,
        right: FilterExpr | None,
    ) -> FilterExpr | None:
        """
        Combine two filter expressions with logical AND.

        Args:
            left: Left-hand filter expression.
            right: Right-hand filter expression.

        Returns:
            The combined filter, or the non-`None` operand.

        """
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
        """
        Append episodes to storage and selected memory backends.

        Args:
            session_data: Session context used to route writes.
            episode_entries: Episode messages/entries to add.
            target_memories: Memory types to update (episodic, semantic).

        Returns:
            IDs of the created episodes.

        """
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
                    episodes=episodes,
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
        agent_mode: bool = False,
    ) -> EpisodicMemory.QueryResponse | None:
        """
        Query episodic memory for relevant episodes.

        Args:
            session_data: Session context used to select the memory.
            query: Query string.
            limit: Optional maximum number of results.
            expand_context: Number of surrounding episodes to return with each match.
            search_filter: Optional property filter for narrowing results.
            score_threshold: Optional minimum score threshold for results.
            agent_mode: Whether to use retrieval-agent mode for episodic search.

        Returns:
            Episodic memory query response, if episodic memory is enabled.

        """
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
                agent_mode=agent_mode,
            )

        return response

    async def query_search(
        self,
        session_data: InstanceOf[SessionData],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
        set_metadata: Mapping[str, JsonValue] | None = None,
        query: str,
        limit: int
        | None = None,  # TODO: Define if limit is per memory or is global limit
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        search_filter: str | None = None,
        agent_mode: bool = False,
    ) -> SearchResponse:
        """
        Search across enabled memory types using a query string.

        Args:
            session_data: Session context used to route the search.
            target_memories: Which memory types to query.
            set_metadata: Optional metadata tags used to select semantic sets.
            query: Query string.
            limit: Optional maximum number of results per memory.
            expand_context: Number of surrounding episodes to return with each match.
            search_filter: Optional filter string applied to each memory query.
            score_threshold: Optional minimum score threshold for results.
            agent_mode: Whether to use retrieval-agent mode for episodic search.

        Returns:
            Aggregated search results across memory types.

        """
        episodic_task: Task | None = None
        semantic_task: Task | None = None

        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "MemMachine querying search: query=%s agent_mode=%s",
            query,
            agent_mode,
        )
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
                    agent_mode=agent_mode,
                )
            )

        if MemoryType.Semantic in target_memories:
            semantic_session = await self._resources.get_semantic_session_manager()
            semantic_task = asyncio.create_task(
                semantic_session.search(
                    message=query,
                    session_data=session_data,
                    set_metadata=set_metadata,
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
        set_metadata: Mapping[str, JsonValue] | None = None,
        search_filter: str | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
    ) -> ListResults:
        """
        List episodes/features matching a filter with pagination.

        Args:
            session_data: Session context used to route the query.
            target_memories: Which memory types to query.
            set_metadata: Optional metadata tags used to select semantic sets.
            search_filter: Optional filter string applied to the query.
            page_size: Optional page size.
            page_num: Optional 1-based page number.

        Returns:
            Aggregated list results across memory types.

        """
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
                    set_metadata=set_metadata,
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
        """
        Count episodes in a session matching a filter.

        Args:
            session_data: Session context providing the session key.
            search_filter: Optional filter string narrowing which episodes count.

        Returns:
            Number of episodes matching the filter.

        """
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
        """
        Delete episodes from storage and memory backends.

        Args:
            episode_ids: IDs of episodes to delete.
            session_data: Optional session context for episodic memory deletion.

        Returns:
            None.

        """
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
        """
        Delete semantic features by ID.

        Args:
            feature_ids: Feature identifiers to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()
        await semantic_session.delete_features(feature_ids)

    async def add_feature(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        tag: str,
        feature: str,
        value: str,
        feature_metadata: dict[str, JsonValue] | None = None,
        citations: list[EpisodeIdT] | None = None,
    ) -> FeatureIdT:
        """
        Add a semantic feature to the current semantic set.

        Args:
            set_id: Set ID to add the feature to.
            feature_metadata: Optional metadata to store alongside the feature.
            category_name: Category name to attach the feature to.
            feature: Feature name/key.
            value: Feature value.
            tag: Tag name to associate with the feature.
            citations: Optional episode IDs supporting this feature.

        Returns:
            The created feature ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_feature(
            set_id=set_id,
            feature_metadata=feature_metadata,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            citations=citations,
        )

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        """
        Fetch a semantic feature by ID.

        Args:
            feature_id: Feature identifier.
            load_citations: Whether to load referenced episode IDs.

        Returns:
            The feature, or `None` if it does not exist.

        """
        semantic_session = await self._resources.get_semantic_session_manager()
        return await semantic_session.get_feature(
            feature_id=feature_id,
            load_citations=load_citations,
        )

    async def update_feature(
        self,
        *,
        feature_id: FeatureIdT,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """
        Update an existing semantic feature.

        Only fields that are not `None` are updated.

        Args:
            feature_id: Feature identifier.
            category_name: New category name.
            feature: New feature name/key.
            value: New feature value.
            tag: New tag name.
            metadata: Replacement metadata payload.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.update_feature(
            feature_id=feature_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
        )

    async def create_semantic_set_type(
        self,
        *,
        session_data: SessionData,
        is_org_level: bool,
        metadata_tags: list[str],
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Create a new semantic set type.

        Args:
            session_data: Context used to locate the project/org scope.
            set_metadata: Optional metadata tags used to select semantic sets.
            is_org_level: Whether the set type is org-scoped.
            metadata_tags: Ordered list of metadata tag keys defining the set.
            name: Optional name for the set type.
            description: Optional description for the set type.

        Returns:
            The created set type ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.create_set_type(
            session_data=session_data,
            is_org_level=is_org_level,
            metadata_tags=metadata_tags,
            name=name,
            description=description,
        )

    async def semantic_list_set_ids(
        self,
        *,
        session_data: SessionData,
        set_metadata: Mapping[str, JsonValue] | None = None,
    ) -> Iterable[SemanticSessionManager.Set]:
        """
        List set IDs for the given session data.

        Args:
            session_data: Context used to locate the project/org scope.
            set_metadata: Metadata tags used to select semantic sets.

        Returns:
            List of set IDs matching the session data criteria.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.list_set_ids(
            session_data=session_data,
            set_metadata=set_metadata,
        )

    async def semantic_get_set_id(
        self,
        *,
        session_data: SessionData,
        is_org_level: bool,
        metadata_tags: list[str],
        set_metadata: Mapping[str, JsonValue] | None = None,
    ) -> SetIdT:
        """
        Retrieve the set ID for a given set type and metadata tags.

        Args:
            session_data: Context used to locate the project/org scope.
            is_org_level: Whether the set is org-scoped (vs project-scoped).
            metadata_tags: Ordered list of metadata tag keys defining the set.
            set_metadata: Optional metadata tags used to select semantic sets.

        Returns:
            The set ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.get_set_id(
            session_data=session_data,
            is_org_level=is_org_level,
            set_metadata_keys=metadata_tags,
            set_metadata=set_metadata,
        )

    async def delete_semantic_set_type(self, set_type_id: str) -> None:
        """
        Delete a semantic set type by ID.

        Args:
            set_type_id: Set type identifier to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.delete_set_type(
            set_type_id=set_type_id,
        )

    async def list_semantic_set_type(
        self,
        *,
        session_data: SessionData,
    ) -> Iterable[SetTypeEntry]:
        """
        List semantic set types available for the context.

        Args:
            session_data: Context used to locate the project/org scope.

        Returns:
            Available semantic set types for the context.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.list_set_types(
            session_data=session_data,
        )

    async def configure_semantic_set(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        """
        Configure the semantic set used for feature extraction/storage.

        Args:
            set_id: The set ID to configure.
            embedder_name: Optional embedder override for this set.
            llm_name: Optional LLM override for this set.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.configure_set(
            set_id=set_id,
            embedder_name=embedder_name,
            llm_name=llm_name,
        )

    async def semantic_get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> SemanticConfigStorage.Category | None:
        """
        Get a semantic category by its ID.

        Args:
            category_id: The ID of the category to retrieve.

        Returns:
            The category if found, otherwise None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.get_category(category_id=category_id)

    async def semantic_add_category(
        self,
        *,
        set_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        """
        Create a new semantic category within the configured set.

        Args:
            set_id: Set ID to create the category in.
            category_name: Name of the category.
            prompt: Prompt to go with the category.
            description: Human-readable category description.

        Returns:
            The created category ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_new_category(
            set_id=set_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

    async def semantic_add_category_template(
        self,
        *,
        set_type_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        """Create a new semantic category on a set type.

        Categories created here are inherited by set_ids mapped to the set type.

        Args:
            set_type_id: The set type identifier.
            category_name: Name of the category.
            prompt: Prompt to go with the category.
            description: Human-readable category description.

        Returns:
            The created category ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_category_template(
            set_type_id=set_type_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

    async def semantic_list_category_templates(
        self,
        *,
        set_type_id: str,
    ) -> Iterable[SemanticCategory]:
        """List semantic categories defined on a set type."""
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.list_category_templates(set_type_id=set_type_id)

    async def semantic_disable_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        """
        Disable a default semantic category for the configured set.

        Args:
            set_id: Set ID to disable the category in.
            category_name: Name of the default category to disable.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.disable_category(
            set_id=set_id,
            category_name=category_name,
        )

    async def semantic_get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> Iterable[SetIdT]:
        """
        Get the set_ids associated with a semantic category.

        Args:
            category_id: Category identifier to query.

        Returns:
            List of set_ids associated with the category.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.get_category_set_ids(
            category_id=category_id,
        )

    async def semantic_delete_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> None:
        """
        Delete a semantic category and all its tags.

        Args:
            category_id: Category identifier to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.delete_category(
            category_id=category_id,
        )

    async def semantic_add_tag_to_category(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        tag_description: str,
    ) -> TagIdT:
        """
        Add a tag to an existing semantic category.

        Args:
            category_id: Category identifier.
            tag_name: Name of the new tag.
            tag_description: Human-readable description of the tag.

        Returns:
            The created tag ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_tag(
            category_id=category_id,
            tag_name=tag_name,
            tag_description=tag_description,
        )

    async def semantic_delete_tag(self, *, tag_id: TagIdT) -> None:
        """
        Delete a semantic tag by ID.

        Args:
            tag_id: Tag identifier to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.delete_tag(tag_id=tag_id)

    async def delete_all(self) -> None:
        """
        Delete all MemMachine data from backing stores.

        This is a destructive operation intended for testing/administration.

        Returns:
            None.

        """
        logger.info("Deleting all data from MemMachine")

        # TODO: Add episodic memory deletion

        semantic_resource_manager = await self._resources.get_semantic_manager()
        semantic_storage = await semantic_resource_manager.get_semantic_storage()
        semantic_config_storage = (
            await semantic_resource_manager.get_semantic_config_storage()
        )

        episodic_store = await self._resources.get_episode_storage()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(semantic_storage.delete_all())
            tg.create_task(semantic_config_storage.delete_all())
            tg.create_task(episodic_store.delete_all())
