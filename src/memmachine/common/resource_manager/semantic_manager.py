"""Manager for semantic memory resources and services."""

import asyncio

from pydantic import InstanceOf

from memmachine.common.configuration import PromptConf, SemanticMemoryConf
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import EpisodeStorage
from memmachine.common.language_model import LanguageModel
from memmachine.common.resource_manager import CommonResourceManager
from memmachine.semantic_memory.config_store.caching_semantic_config_storage import (
    CachingSemanticConfigStorage,
)
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.config_store.config_store_sqlalchemy import (
    SemanticConfigStorageSqlAlchemy,
)
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    SemanticCategory,
    SetIdT,
)
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager
from memmachine.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
)
from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage


class SemanticResourceManager:
    """Build and cache components used by semantic memory."""

    def __init__(
        self,
        *,
        semantic_conf: SemanticMemoryConf,
        prompt_conf: PromptConf,
        resource_manager: InstanceOf[CommonResourceManager],
        episode_storage: EpisodeStorage,
    ) -> None:
        """Store configuration and supporting managers."""
        self._resource_manager = resource_manager
        self._conf = semantic_conf
        self._prompt_conf = prompt_conf
        self._episode_storage = episode_storage

        self._semantic_service: SemanticService | None = None
        self._semantic_session_manager: SemanticSessionManager | None = None

    async def close(self) -> None:
        """Stop semantic services if they were started."""
        tasks = []

        if self._semantic_service is not None:
            tasks.append(self._semantic_service.stop())

        await asyncio.gather(*tasks)

    async def get_semantic_storage(self) -> SemanticStorage:
        database = self._conf.database

        # TODO: validate/choose based on database provider
        storage: SemanticStorage
        try:
            sql_engine = await self._resource_manager.get_sql_engine(
                database, validate=True
            )
            storage = SqlAlchemyPgVectorSemanticStorage(sql_engine)
        except ValueError:
            # try graph store
            neo4j_engine = await self._resource_manager.get_neo4j_driver(
                database, validate=True
            )
            storage = Neo4jSemanticStorage(neo4j_engine)

        await storage.startup()
        return storage

    async def get_semantic_config_storage(self) -> SemanticConfigStorage:
        database = self._conf.config_database

        sql_engine = await self._resource_manager.get_sql_engine(database)
        storage = SemanticConfigStorageSqlAlchemy(sql_engine)

        if self._conf.with_config_cache:
            storage = CachingSemanticConfigStorage(
                wrapped=storage,
            )

        await storage.startup()

        return storage

    async def _get_default_embedder(self) -> Embedder:
        embedder = self._conf.embedding_model
        return await self._resource_manager.get_embedder(embedder, validate=True)

    async def _get_default_language_model(self) -> LanguageModel:
        language_model = self._conf.llm_model
        return await self._resource_manager.get_language_model(
            language_model, validate=True
        )

    async def get_semantic_service(self) -> SemanticService:
        """Return the semantic service, constructing it if needed."""
        if self._semantic_service is not None:
            return self._semantic_service

        semantic_storage = await self.get_semantic_storage()
        episode_store = self._episode_storage

        semantic_categories_by_isolation = self._prompt_conf.default_semantic_categories

        def get_default_categories(set_id: SetIdT) -> list[SemanticCategory]:
            def_type = SemanticSessionManager.get_default_set_id_type(set_id)
            return semantic_categories_by_isolation[def_type]

        embedder = await self._get_default_embedder()
        llm_model = await self._get_default_language_model()

        config_store = await self.get_semantic_config_storage()

        self._semantic_service = SemanticService(
            SemanticService.Params(
                semantic_storage=semantic_storage,
                episode_storage=episode_store,
                resource_manager=self._resource_manager,
                default_embedder=embedder,
                default_embedder_name=self._conf.embedding_model,
                default_language_model=llm_model,
                default_category_retriever=get_default_categories,
                semantic_config_storage=config_store,
                uningested_time_limit=self._conf.ingestion_trigger_age,
                uningested_message_limit=self._conf.ingestion_trigger_messages,
            ),
        )
        return self._semantic_service

    async def get_semantic_session_manager(self) -> SemanticSessionManager:
        """Return the semantic session manager, constructing if needed."""
        if self._semantic_session_manager is not None:
            return self._semantic_session_manager

        self._semantic_session_manager = SemanticSessionManager(
            semantic_service=await self.get_semantic_service(),
            semantic_config_storage=await self.get_semantic_config_storage(),
        )
        return self._semantic_session_manager
