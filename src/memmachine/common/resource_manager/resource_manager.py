"""Resource manager wiring together storage, embedders, and models."""

import asyncio

from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.configuration import Configuration
from memmachine.common.configuration.metrics_conf import WithMetricsFactoryId
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import CountCachingEpisodeStorage, EpisodeStorage
from memmachine.common.episode_store.episode_sqlalchemy_store import (
    SqlAlchemyEpisodeStore,
)
from memmachine.common.language_model import LanguageModel
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.reranker import Reranker
from memmachine.common.resource_manager.database_manager import DatabaseManager
from memmachine.common.resource_manager.embedder_manager import EmbedderManager
from memmachine.common.resource_manager.language_model_manager import (
    LanguageModelManager,
)
from memmachine.common.resource_manager.reranker_manager import RerankerManager
from memmachine.common.resource_manager.semantic_manager import SemanticResourceManager
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.common.session_manager.session_data_manager_sql_impl import (
    SessionDataManagerSQL,
)
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
    EpisodicMemoryManagerParams,
)
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager


class ResourceManagerImpl:
    """Concrete resource manager for MemMachine services."""

    def __init__(self, conf: Configuration) -> None:
        """Initialize managers from configuration."""
        self._conf = conf
        self._conf.logging.apply()
        self._database_manager: DatabaseManager = DatabaseManager(
            self._conf.resources.databases
        )
        self._embedder_manager: EmbedderManager = EmbedderManager(
            self._conf.resources.embedders
        )
        self._model_manager: LanguageModelManager = LanguageModelManager(
            self._conf.resources.language_models,
        )
        self._reranker_manager: RerankerManager = RerankerManager(
            self._conf.resources.rerankers,
            embedder_factory=self._embedder_manager,
        )

        self._session_data_manager: SessionDataManager | None = None
        self._episodic_memory_manager: EpisodicMemoryManager | None = None

        self._episode_storage: EpisodeStorage | None = None
        self._semantic_manager: SemanticResourceManager | None = None

    async def build(self) -> None:
        """Build all configured resources in parallel."""
        tasks = [
            self._database_manager.build_all(validate=True),
            self._embedder_manager.build_all(),
            self._model_manager.build_all(),
            self._reranker_manager.build_all(),
        ]

        await asyncio.gather(*tasks)

        # TODO: Build semantic storage, episodic storage, and session data storage lazily when actually used

    async def close(self) -> None:
        """Close resources and clean up state."""
        tasks = []
        if self._semantic_manager is not None:
            tasks.append(self._semantic_manager.close())

        tasks.append(self._database_manager.close())

        await asyncio.gather(*tasks)

    async def get_sql_engine(self, name: str) -> AsyncEngine:
        """Return a SQL engine by name."""
        return await self._database_manager.async_get_sql_engine(name)

    async def get_neo4j_driver(self, name: str) -> AsyncDriver:
        """Return a Neo4j driver by name."""
        return await self._database_manager.async_get_neo4j_driver(name)

    async def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        """Return a vector graph store by name."""
        return await self._database_manager.async_get_vector_graph_store(name)

    async def get_embedder(self, name: str) -> Embedder:
        """Return an embedder by name."""
        return await self._embedder_manager.get_embedder(name)

    async def get_language_model(self, name: str) -> LanguageModel:
        """Return a language model by name."""
        return await self._model_manager.get_language_model(name)

    async def get_reranker(self, name: str) -> Reranker:
        """Return a reranker by name."""
        return await self._reranker_manager.get_reranker(name)

    @property
    def config(self) -> Configuration:
        """Return the configuration instance."""
        return self._conf

    async def get_session_data_manager(self) -> SessionDataManager:
        """Lazy-load the session data manager."""
        if self._session_data_manager is not None:
            return self._session_data_manager
        database = self._conf.session_manager.database
        engine = await self._database_manager.async_get_sql_engine(database)

        self._session_data_manager = SessionDataManagerSQL(engine)
        await self._session_data_manager.create_tables()

        return self._session_data_manager

    async def get_episodic_memory_manager(self) -> EpisodicMemoryManager:
        """Lazy-load the episodic memory manager."""
        if self._episodic_memory_manager is not None:
            return self._episodic_memory_manager
        session_data_manager = await self.get_session_data_manager()
        params = EpisodicMemoryManagerParams(
            resource_manager=self,
            session_data_manager=session_data_manager,
        )
        self._episodic_memory_manager = EpisodicMemoryManager(params)
        return self._episodic_memory_manager

    async def get_episode_storage(self) -> EpisodeStorage:
        """Return the episode storage instance."""
        if self._episode_storage is not None:
            return self._episode_storage

        episode_storage_conf = getattr(self._conf, "episode_storage", None)
        if episode_storage_conf is None:
            episode_storage_conf = self._conf.episode_store

        database = episode_storage_conf.database
        engine = await self.get_sql_engine(database)

        episode_storage = SqlAlchemyEpisodeStore(engine)
        await episode_storage.startup()

        if episode_storage_conf.with_count_cache:
            episode_storage = CountCachingEpisodeStorage(episode_storage)

        self._episode_storage = episode_storage
        return self._episode_storage

    async def get_semantic_service(self) -> SemanticService:
        """Return the semantic service manager."""
        semantic_manager = await self.get_semantic_manager()
        return await semantic_manager.get_semantic_service()

    async def get_semantic_manager(self) -> SemanticResourceManager:
        """Return the semantic resource manager, constructing if needed."""
        if self._semantic_manager is not None:
            return self._semantic_manager

        episode_storage = await self.get_episode_storage()
        self._semantic_manager = SemanticResourceManager(
            semantic_conf=self._conf.semantic_memory,
            prompt_conf=self._conf.prompt,
            resource_manager=self,  # type: ignore[arg-type]
            episode_storage=episode_storage,
        )
        return self._semantic_manager

    async def get_semantic_session_manager(self) -> SemanticSessionManager:
        """Return the semantic session manager."""
        semantic_manager = await self.get_semantic_manager()
        return await semantic_manager.get_semantic_session_manager()

    @staticmethod
    async def get_metrics_factory(name: str) -> MetricsFactory:
        """Return the metrics factory by name."""
        factory_cache = WithMetricsFactoryId(metrics_factory_id=name)
        ret = factory_cache.get_metrics_factory()
        if ret is None:
            raise ValueError(f"MetricsFactory '{name}' could not be created.")
        return ret
