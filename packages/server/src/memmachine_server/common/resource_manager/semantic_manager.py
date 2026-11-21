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
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionDeletedError,
    PartitionManager,
    PartitionPendingError,
)
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import (
    VectorStore,
    VectorStorePartition,
    VectorStoreSchema,
)
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

_VECTOR_STORE_NAME = "semantic_memory"
"""The name of semantic memory's one vector store, whatever its embedder.

Fixed, because the name locates the store's data.
"""
_PARTITION_NAMESPACE = "semantic_memory"
"""The partition namespace semantic memory owns, the same in every deployment."""
_PARTITION_NAME = "semantic_memory"
"""Semantic memory's one partition, which it never deletes."""
_TOMBSTONE_RETENTION_SECONDS = 86400
"""The retention of semantic memory's namespace, whose tombstones are those of failed or canceled creations."""

# The attempts `_get_or_create_partition` makes,
# _GET_OR_CREATE_RETRY_DELAY_SECONDS apart, before giving up on a partition
# that stays pending or a name other processes keep taking.
_MAX_GET_OR_CREATE_ATTEMPTS = 10
_GET_OR_CREATE_RETRY_DELAY_SECONDS = 1


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

        self._semantic_storage: SemanticStorage | None = None
        self._semantic_storage_lock = asyncio.Lock()
        self._semantic_service: SemanticService | None = None
        self._semantic_session_manager: SemanticSessionManager | None = None

    async def stop(self) -> None:
        """Stop the semantic service and the semantic storage it built and started."""
        if self._semantic_service is not None:
            await self._semantic_service.stop()
        if self._semantic_storage is not None:
            await self._semantic_storage.stop()

    async def prepare(self) -> None:
        """
        Prepare semantic memory's stores.

        Its storage and its config store are each prepared when configured,
        whether or not semantic memory is enabled. Run it with no server
        using them, before semantic memory starts.
        """
        if self._conf.storage_configured():
            if self._conf.storage_backend == SemanticMemoryStorageBackend.VECTOR_STORE:
                await self._prepare_vector_store_semantic_storage()
            else:
                await (await self._construct_semantic_storage()).prepare()
        if self._conf.config_database:
            await (await self._build_semantic_config_storage()).prepare()

    async def get_semantic_storage(self) -> SemanticStorage:
        """Return the semantic storage, building and starting it on first use.

        Resolves no language model, so deletion can use it while semantic
        memory is disabled. The vector-store backend resolves the embedder
        only when `vector_dimensions` is unset.
        """
        async with self._semantic_storage_lock:
            if self._semantic_storage is None:
                self._semantic_storage = await self._build_semantic_storage()
            return self._semantic_storage

    async def _build_semantic_storage(self) -> SemanticStorage:
        if self._conf.storage_backend == SemanticMemoryStorageBackend.VECTOR_STORE:
            return await self._get_vector_store_semantic_storage()
        storage = await self._construct_semantic_storage()
        await storage.start()
        return storage

    async def _construct_semantic_storage(self) -> SemanticStorage:
        database = self._conf.database

        if not database:
            raise ResourceNotReadyError(
                "No database configured for semantic storage.", "semantic_memory"
            )

        if self._conf.storage_backend == SemanticMemoryStorageBackend.PGVECTOR:
            sql_engine = await self._resource_manager.get_sql_engine(
                database, validate=True
            )
            return SqlAlchemyPgVectorSemanticStorage(sql_engine)

        if self._conf.storage_backend == SemanticMemoryStorageBackend.NEO4J:
            neo4j_engine = await self._resource_manager.get_neo4j_driver(
                database, validate=True
            )
            return Neo4jSemanticStorage(neo4j_engine)

        try:
            sql_engine = await self._resource_manager.get_sql_engine(
                database, validate=True
            )
            return SqlAlchemyPgVectorSemanticStorage(sql_engine)
        except ValueError:
            # try graph store
            neo4j_engine = await self._resource_manager.get_neo4j_driver(
                database, validate=True
            )
            return Neo4jSemanticStorage(neo4j_engine)

    async def _prepare_vector_store_semantic_storage(self) -> None:
        (
            feature_store_name,
            vector_store_id,
            store_schema,
        ) = await self._vector_store_semantic_resources()
        await self._resource_manager.prepare_vector_store(
            vector_store_id,
            vector_store_name=_VECTOR_STORE_NAME,
            store_schema=store_schema,
        )
        partition_registry = await self._resource_manager.build_partition_registry(
            feature_store_name,
            partition_namespace=_PARTITION_NAMESPACE,
            tombstone_retention_seconds=_TOMBSTONE_RETENTION_SECONDS,
        )
        await partition_registry.prepare()
        vector_store = await self._resource_manager.get_vector_store(
            vector_store_id,
            vector_store_name=_VECTOR_STORE_NAME,
            store_schema=store_schema,
        )
        # The storage's partition is created here, so starting the storage
        # finds it; a registry built for preparing runs no purge rounds.
        partition = await SemanticResourceManager._get_or_create_partition(
            PartitionManager(
                partition_registry,
                [vector_store],
                tracker=OperationTracker(
                    await self._resource_manager.get_metrics_factory("prometheus"),
                    prefix=_PARTITION_NAMESPACE,
                ),
            ),
            vector_store,
        )
        sql_engine = await self._resource_manager.get_sql_engine(
            feature_store_name,
            validate=True,
        )
        await VectorStoreSemanticStorage(sql_engine, partition).prepare()

    async def _get_vector_store_semantic_storage(self) -> SemanticStorage:
        (
            feature_store_name,
            vector_store_id,
            store_schema,
        ) = await self._vector_store_semantic_resources()
        sql_engine = await self._resource_manager.get_sql_engine(
            feature_store_name,
            validate=True,
        )
        vector_store = await self._resource_manager.get_vector_store(
            vector_store_id,
            vector_store_name=_VECTOR_STORE_NAME,
            store_schema=store_schema,
        )

        # Semantic memory owns this partition and creates it when the storage
        # is built, through a registry in the database the storage uses.
        partition_registry = await self._resource_manager.get_partition_registry(
            feature_store_name,
            partition_namespace=_PARTITION_NAMESPACE,
            tombstone_retention_seconds=_TOMBSTONE_RETENTION_SECONDS,
            partitioned_stores=[vector_store],
        )
        partition = await SemanticResourceManager._get_or_create_partition(
            PartitionManager(
                partition_registry,
                [vector_store],
                tracker=OperationTracker(
                    await self._resource_manager.get_metrics_factory("prometheus"),
                    prefix=_PARTITION_NAMESPACE,
                ),
            ),
            vector_store,
        )
        storage = VectorStoreSemanticStorage(sql_engine, partition)
        await storage.start()
        return storage

    async def _vector_store_semantic_resources(
        self,
    ) -> tuple[str, str, VectorStoreSchema]:
        """The feature store, the vector store's id, and the schema of the store semantic memory keeps on it."""
        feature_store_name = self._conf.feature_store
        if not feature_store_name:
            raise ResourceNotReadyError(
                "No feature_store configured for vector-backed semantic storage.",
                "semantic_memory",
            )

        vector_store_id = self._conf.vector_collection
        if not vector_store_id:
            raise ResourceNotReadyError(
                "No vector_collection configured for vector-backed semantic storage.",
                "semantic_memory",
            )

        vector_dimensions = self._conf.vector_dimensions
        if vector_dimensions is None:
            vector_dimensions = (await self._get_default_embedder()).dimensions
        store_schema = VectorStoreSchema(
            vector_dimensions=vector_dimensions,
            similarity_metric=self._conf.vector_similarity_metric,
            indexed_properties={},
        )
        return feature_store_name, vector_store_id, store_schema

    @staticmethod
    async def _get_or_create_partition(
        partition_manager: PartitionManager,
        vector_store: VectorStore,
    ) -> VectorStorePartition:
        """
        Return the vector store's handle on semantic memory's partition, creating the partition if no partition holds its name.

        Read-then-create, retried a bounded number of times: a live partition
        is returned; a pending one is another process's creation, which the
        call waits for; no partition means create, then read again. The call
        retries when the creation finds the name taken or the partition
        deleted while the store created it, and when the partition is deleted
        before the read after the creation or before the store returns its
        handle.

        Raises:
            PartitionPendingError:
                If the partition was another process's creation and still
                pending at the last attempt.
            PartitionCreateAttemptsExhaustedError:
                If the attempts ran out with the last one losing the name or
                the partition to another process, or if the registry gave up
                after repeated attempts to reserve the free name failed.
        """
        pending_error: PartitionPendingError | None = None
        lost_race: PartitionAlreadyExistsError | PartitionDeletedError | None = None
        for attempt in range(_MAX_GET_OR_CREATE_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_GET_OR_CREATE_RETRY_DELAY_SECONDS)
            try:
                registration = await partition_manager.resolve_partition(
                    _PARTITION_NAME
                )
                if registration is None:
                    await partition_manager.create_partition(_PARTITION_NAME)
                    registration = await partition_manager.resolve_partition(
                        _PARTITION_NAME
                    )
            except PartitionPendingError as err:
                pending_error = err
                continue
            except (PartitionAlreadyExistsError, PartitionDeletedError) as err:
                pending_error = None
                lost_race = err
                continue
            pending_error = None
            if registration is None:
                continue
            partition = await vector_store.get_partition(registration.partition_uuid)
            if partition is not None:
                return partition
        # The last lookup found the partition pending.
        if pending_error is not None:
            raise pending_error
        raise PartitionCreateAttemptsExhaustedError(
            f"Getting or creating semantic memory's partition made no progress "
            f"after {_MAX_GET_OR_CREATE_ATTEMPTS} attempts"
        ) from lost_race

    async def get_semantic_config_storage(self) -> SemanticConfigStorage:
        storage = await self._build_semantic_config_storage()
        await storage.start()
        return storage

    async def _build_semantic_config_storage(self) -> SemanticConfigStorage:
        database = self._conf.config_database

        if not database:
            raise ResourceNotReadyError(
                "No config database configured for semantic memory.",
                "semantic_memory",
            )

        sql_engine = await self._resource_manager.get_sql_engine(database)
        storage: SemanticConfigStorage = SemanticConfigStorageSqlAlchemy(sql_engine)

        if self._conf.with_config_cache:
            storage = CachingSemanticConfigStorage(
                wrapped=storage,
            )
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
