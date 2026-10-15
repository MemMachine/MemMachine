"""Manager for semantic memory resources and services."""

import asyncio
from typing import cast

from neo4j import AsyncDriver
from pydantic import InstanceOf
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine

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
    Feature,
    SqlAlchemyPgVectorSemanticStorage,
)
from memmachine_server.semantic_memory.storage.storage_base import (
    SemanticDeletionStorage,
    SemanticStorage,
)
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorSemanticFeature,
    VectorStoreSemanticStorage,
)

_VECTOR_STORE_NAMESPACE = "semantic_memory"
_VECTOR_STORE_COLLECTION_NAME = "semantic_memory"


def _vector_store_resource_names(conf: SemanticMemoryConf) -> tuple[str, str]:
    if not conf.feature_store:
        raise ResourceNotReadyError(
            "No feature_store configured for vector-backed semantic storage.",
            "semantic_memory",
        )
    if not conf.vector_collection:
        raise ResourceNotReadyError(
            "No vector_collection configured for vector-backed semantic storage.",
            "semantic_memory",
        )
    return conf.feature_store, conf.vector_collection


async def _database_backend(
    conf: SemanticMemoryConf,
    resource_manager: CommonResourceManager,
) -> AsyncEngine | AsyncDriver:
    database = conf.database
    if not database:
        raise ResourceNotReadyError(
            "No database configured for semantic storage.", "semantic_memory"
        )

    if conf.storage_backend == SemanticMemoryStorageBackend.PGVECTOR:
        return await resource_manager.get_sql_engine(database, validate=True)
    if conf.storage_backend == SemanticMemoryStorageBackend.NEO4J:
        return await resource_manager.get_neo4j_driver(database, validate=True)
    try:
        return await resource_manager.get_sql_engine(database, validate=True)
    except ValueError:
        return await resource_manager.get_neo4j_driver(database, validate=True)


def _database_storage(backend: AsyncEngine | AsyncDriver) -> SemanticStorage:
    if isinstance(backend, AsyncEngine):
        return SqlAlchemyPgVectorSemanticStorage(backend)
    return Neo4jSemanticStorage(backend)


async def _has_table(engine: AsyncEngine, name: str) -> bool:
    async with engine.connect() as conn:
        return await conn.run_sync(lambda sync_conn: inspect(sync_conn).has_table(name))


async def build_semantic_deletion_storage(
    conf: SemanticMemoryConf,
    resource_manager: CommonResourceManager,
) -> SemanticDeletionStorage | None:
    """Build semantic storage for deletes only, or None if never initialized.

    Runs no startup, creates no collection and resolves no model, so it works
    while semantic memory is disabled.
    """
    if conf.storage_backend == SemanticMemoryStorageBackend.VECTOR_STORE:
        feature_store_name, vector_store_name = _vector_store_resource_names(conf)
        sql_engine = await resource_manager.get_sql_engine(
            feature_store_name, validate=True
        )
        if not await _has_table(sql_engine, VectorSemanticFeature.__tablename__):
            return None
        vector_store = await resource_manager.get_vector_store(vector_store_name)
        collection = await vector_store.open_collection(
            namespace=_VECTOR_STORE_NAMESPACE,
            name=_VECTOR_STORE_COLLECTION_NAME,
        )
        if collection is None:
            raise ResourceNotReadyError(
                "Semantic feature table exists but its vector collection is "
                f"missing from '{vector_store_name}'.",
                "semantic_memory",
            )
        return VectorStoreSemanticStorage(sql_engine, collection)

    backend = await _database_backend(conf, resource_manager)
    if isinstance(backend, AsyncEngine) and not await _has_table(
        backend, Feature.__tablename__
    ):
        return None
    return _database_storage(backend)


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

        storage = _database_storage(
            await _database_backend(self._conf, self._resource_manager)
        )
        await storage.startup()
        return storage

    async def _get_vector_store_semantic_storage(self) -> SemanticStorage:
        feature_store_name, vector_store_name = _vector_store_resource_names(self._conf)

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
            ),
        )
        storage = VectorStoreSemanticStorage(sql_engine, collection)
        await storage.startup()
        return storage

    async def get_semantic_config_storage(self) -> SemanticConfigStorage:
        database = self._conf.config_database

        if not database:
            raise ResourceNotReadyError(
                "No config database configured for semantic memory.",
                "semantic_memory",
            )

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
                missing_episode_grace_period_sec=self._conf.missing_episode_grace_period_sec,
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
