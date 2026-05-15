"""Manager for semantic memory resources and services."""

import asyncio
from typing import cast

from pydantic import InstanceOf

from memmachine_server.common.configuration import (
    PromptConf,
    SemanticMemoryConf,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeStorage
from memmachine_server.common.errors import ResourceNotReadyError
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import VectorStoreCollectionConfig
from memmachine_server.semantic_memory.config_store.caching_semantic_config_storage import (
    CachingSemanticConfigStorage,
)
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.config_store.config_store_sqlalchemy import (
    SemanticConfigStorageSqlAlchemy,
)
from memmachine_server.semantic_memory.semantic_memory import (
    ResourceManager,
    SemanticService,
)
from memmachine_server.semantic_memory.semantic_model import (
    SemanticCategory,
    SetIdT,
)
from memmachine_server.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)
from memmachine_server.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
)
from memmachine_server.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorStoreSemanticStorage,
)

_VECTOR_STORE_NAMESPACE = "semantic_memory"
_VECTOR_STORE_COLLECTION_NAME = "semantic_memory"


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
        if self._conf.storage_backend == SemanticMemoryStorageBackend.VECTOR_STORE:
            return await self._get_vector_store_semantic_storage()

        database = self._conf.database

        if database is None:
            raise ResourceNotReadyError(
                "No database configured for semantic storage.", "semantic_memory"
            )

        if self._conf.storage_backend == SemanticMemoryStorageBackend.PGVECTOR:
            sql_engine = await self._resource_manager.get_sql_engine(
                database, validate=True
            )
            storage = SqlAlchemyPgVectorSemanticStorage(sql_engine)
            await storage.startup()
            return storage

        if self._conf.storage_backend == SemanticMemoryStorageBackend.NEO4J:
            neo4j_engine = await self._resource_manager.get_neo4j_driver(
                database, validate=True
            )
            storage = Neo4jSemanticStorage(neo4j_engine)
            await storage.startup()
            return storage

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

    async def _get_vector_store_semantic_storage(self) -> SemanticStorage:
        feature_store_name = self._conf.feature_store
        if not feature_store_name:
            raise ResourceNotReadyError(
                "No feature_store configured for vector-backed semantic storage.",
                "semantic_memory",
            )

        vector_store_name = self._conf.vector_collection
        if not vector_store_name:
            raise ResourceNotReadyError(
                "No vector_collection configured for vector-backed semantic storage.",
                "semantic_memory",
            )

        sql_engine = await self._resource_manager.get_sql_engine(
            feature_store_name,
            validate=True,
        )
        vector_store = await self._resource_manager.get_vector_store(vector_store_name)
        vector_dimensions = self._conf.vector_dimensions
        if vector_dimensions is None:
            vector_dimensions = (await self._get_default_embedder()).dimensions

        collection = await vector_store.open_or_create_collection(
            namespace=_VECTOR_STORE_NAMESPACE,
            name=_VECTOR_STORE_COLLECTION_NAME,
            config=VectorStoreCollectionConfig(
                vector_dimensions=vector_dimensions,
                similarity_metric=self._conf.vector_similarity_metric,
                indexed_properties_schema={
                    "feature_id": str,
                    "set_id": str,
                    "set": str,
                    "semantic_category_id": str,
                    "category_name": str,
                    "category": str,
                    "tag_id": str,
                    "tag": str,
                    "feature": str,
                    "feature_name": str,
                    "value": str,
                },
            ),
        )
        storage = VectorStoreSemanticStorage(sql_engine, collection)
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

    def _get_default_embedder_name(self) -> str:
        embedder = self._conf.embedding_model
        if not embedder:
            raise ResourceNotReadyError(
                "No embedding model configured for semantic memory.",
                "semantic_memory",
            )
        return embedder

    def _get_default_language_model_name(self) -> str:
        language_model = self._conf.llm_model
        if not language_model:
            raise ResourceNotReadyError(
                "No language model configured for semantic memory.",
                "semantic_memory",
            )
        return language_model

    async def _get_default_embedder(self) -> Embedder:
        embedder_name = self._get_default_embedder_name()
        return await self._resource_manager.get_embedder(embedder_name, validate=True)

    async def _get_default_language_model(self) -> LanguageModel:
        language_model_name = self._get_default_language_model_name()
        return await self._resource_manager.get_language_model(
            language_model_name, validate=True
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

        embedder_name = self._get_default_embedder_name()
        embedder = await self._get_default_embedder()
        llm_model = await self._get_default_language_model()

        config_store = await self.get_semantic_config_storage()

        self._semantic_service = SemanticService(
            SemanticService.Params(
                semantic_storage=semantic_storage,
                episode_storage=episode_store,
                resource_manager=cast(ResourceManager, self._resource_manager),
                default_embedder=embedder,
                default_embedder_name=embedder_name,
                default_language_model=llm_model,
                default_category_retriever=get_default_categories,
                semantic_config_storage=config_store,
                uningested_time_limit=self._conf.ingestion_trigger_age,
                uningested_message_limit=self._conf.ingestion_trigger_messages,
                max_features_per_update=self._conf.max_features_per_update,
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
