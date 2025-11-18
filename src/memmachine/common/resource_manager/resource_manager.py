"""Resource manager wiring together storage, embedders, and models."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.configuration import Configuration
from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.reranker import Reranker
from memmachine.common.resource_manager.embedder_manager import EmbedderManager
from memmachine.common.resource_manager.language_model_manager import (
    LanguageModelManager,
)
from memmachine.common.resource_manager.reranker_manager import RerankerManager
from memmachine.common.resource_manager.semantic_manager import SemanticResourceManager
from memmachine.common.resource_manager.storage_manager import StorageManager
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.common.session_manager.session_data_manager_sql_impl import (
    SessionDataManagerSQL,
)
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.episode_store.episode_sqlalchemy_store import SqlAlchemyEpisodeStore
from memmachine.episode_store.episode_storage import EpisodeStorage
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
    EpisodicMemoryManagerParams,
)
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager


class ResourceManagerImpl:
    """Concrete resource manager for MemMachine services."""

    def __init__(self, conf: Configuration) -> None:
        """Initialize managers from configuration."""
        self._conf = conf
        self._conf.logging.apply()
        self._storage_manager: StorageManager = StorageManager(self._conf.resources.storages)
        self._embedder_manager: EmbedderManager = EmbedderManager(self._conf.resources.embedders)
        self._model_manager: LanguageModelManager = LanguageModelManager(
            self._conf.model,
        )
        self._reranker_manager: RerankerManager = RerankerManager(
            self._conf.reranker,
            embedder_factory=self._embedder_manager,
        )

        self._session_data_manager: SessionDataManager | None = None
        self._episodic_memory_manager: EpisodicMemoryManager | None = None

        self._history_storage: EpisodeStorage | None = None
        self._semantic_manager: SemanticResourceManager | None = None

    async def build(self) -> None:
        """Build all configured resources in parallel."""
        tasks = [
            self._storage_manager.build_all(validate=True),
            self._embedder_manager.build_all(),
            self._model_manager.build_all(),
            self._reranker_manager.build_all(),
        ]

        await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close resources and clean up state."""
        tasks = []
        if self._semantic_manager is not None:
            tasks.append(self._semantic_manager.close())

        tasks.append(self._storage_manager.close())

        await asyncio.gather(*tasks)

    async def get_sql_engine(self, name: str) -> AsyncEngine:
        """Return a SQL engine by name."""
        return self._storage_manager.get_sql_engine(name)

    async def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        """Return a vector graph store by name."""
        return self._storage_manager.get_vector_graph_store(name)

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
    def session_data_manager(self) -> SessionDataManager:
        """Lazy-load the session data manager."""
        if self._session_data_manager is not None:
            return self._session_data_manager
        engine = self._storage_manager.get_sql_engine(self._conf.session_db.storage_id)
        self._session_data_manager = SessionDataManagerSQL(engine)
        return self._session_data_manager

    @property
    def episodic_memory_manager(self) -> EpisodicMemoryManager:
        """Lazy-load the episodic memory manager."""
        if self._episodic_memory_manager is not None:
            return self._episodic_memory_manager
        params = EpisodicMemoryManagerParams(
            resource_manager=self,
            session_data_manager=self.session_data_manager,
        )
        self._episodic_memory_manager = EpisodicMemoryManager(params)
        return self._episodic_memory_manager

    @property
    def history_storage(self) -> EpisodeStorage:
        """Lazy-load the episode history storage."""
        if self._history_storage is not None:
            return self._history_storage

        conf = self._conf.history_storage
        engine = self._storage_manager.get_sql_engine(conf.database)

        self._history_storage = SqlAlchemyEpisodeStore(engine)

        return self._history_storage

    async def get_semantic_manager(self) -> SemanticResourceManager:
        """Return the semantic resource manager, constructing if needed."""
        if self._semantic_manager is not None:
            return self._semantic_manager

        self._semantic_manager = SemanticResourceManager(
            semantic_conf=self._conf.semantic_memory,
            prompt_conf=self._conf.prompt,
            resource_manager=self,
            history_storage=self.history_storage,
        )
        return self._semantic_manager

    async def get_semantic_session_manager(self) -> SemanticSessionManager:
        """Return the semantic session manager."""
        semantic_manager = await self.get_semantic_manager()
        return await semantic_manager.get_semantic_session_manager()
