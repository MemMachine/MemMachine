"""Manager for semantic memory resources and services."""

import asyncio
import contextlib
import logging
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
from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionIncarnationEndedError,
    PartitionPendingError,
    PartitionRegistry,
    Registration,
    Reservation,
)
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.common.vector_store import VectorStore, VectorStoreSchema
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
"""The retention of semantic memory's namespace, whose tombstones are only those of creations that failed."""

logger = logging.getLogger(__name__)

# Attempts opening or creating the partition makes,
# _OPEN_OR_CREATE_RETRY_DELAY_SECONDS apart, before it gives up on a
# partition that stays pending or a name it keeps losing.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10
_OPEN_OR_CREATE_RETRY_DELAY_SECONDS = 1

# Reservations cancelled after a failed creation, held until done so the
# garbage collector cannot drop one whose creation was cancelled.
_reservation_cancellations: set[asyncio.Task[None]] = set()


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

    async def close(self) -> None:
        """Stop semantic services if they were started."""
        tasks = []

        if self._semantic_service is not None:
            tasks.append(self._semantic_service.stop())

        await asyncio.gather(*tasks)

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

        database = self._conf.database

        if not database:
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

        vector_store_id = self._conf.vector_collection
        if not vector_store_id:
            raise ResourceNotReadyError(
                "No vector_collection configured for vector-backed semantic storage.",
                "semantic_memory",
            )

        sql_engine = await self._resource_manager.get_sql_engine(
            feature_store_name,
            validate=True,
        )
        vector_dimensions = self._conf.vector_dimensions
        if vector_dimensions is None:
            vector_dimensions = (await self._get_default_embedder()).dimensions
        vector_store = await self._resource_manager.get_vector_store(
            vector_store_id,
            vector_store_name=_VECTOR_STORE_NAME,
            store_schema=VectorStoreSchema(
                vector_dimensions=vector_dimensions,
                similarity_metric=self._conf.vector_similarity_metric,
                indexed_properties={},
            ),
        )

        # The manager owns this partition and provisions it when the storage is
        # built, through a registry in the database the storage already uses.
        partition_registry = await self._resource_manager.get_partition_registry(
            feature_store_name,
            partition_namespace=_PARTITION_NAMESPACE,
            tombstone_retention_seconds=_TOMBSTONE_RETENTION_SECONDS,
            purge_round=vector_store.purge_partition_incarnation,
        )
        registration = await SemanticResourceManager._open_or_create_partition(
            partition_registry, vector_store
        )
        vector_partition = await vector_store.open_partition_incarnation(
            registration.incarnation
        )
        storage = VectorStoreSemanticStorage(sql_engine, vector_partition)
        await storage.startup()
        return storage

    @staticmethod
    async def _open_or_create_partition(
        partition_registry: PartitionRegistry, vector_store: VectorStore
    ) -> Registration:
        """
        Answer the registration of semantic memory's partition, creating it if no partition holds its name.

        Read-then-create, retried a bounded number of times: a live partition
        is answered; a pending one is another process's creation, so the call
        waits for it; no partition means create; losing the reservation means
        another process took the name meanwhile; and losing the confirmation
        means the partition was deleted while its storage was prepared, so the
        call creates again.
        """
        pending_error: PartitionPendingError | None = None
        lost_race: (
            PartitionAlreadyExistsError | PartitionIncarnationEndedError | None
        ) = None
        for attempt in range(_MAX_OPEN_OR_CREATE_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_OPEN_OR_CREATE_RETRY_DELAY_SECONDS)
            try:
                registration = await partition_registry.resolve(_PARTITION_NAME)
            except PartitionPendingError as err:
                pending_error = err
                continue
            pending_error = None
            if registration is not None:
                return registration
            try:
                reservation = await partition_registry.reserve(_PARTITION_NAME)
            except PartitionAlreadyExistsError as err:
                lost_race = err
                continue
            try:
                return await SemanticResourceManager._prepare_and_confirm_or_cancel(
                    reservation, vector_store
                )
            except PartitionIncarnationEndedError as err:
                lost_race = err
                continue
        # The last lookup found the partition pending.
        if pending_error is not None:
            raise pending_error
        raise PartitionCreateAttemptsExhaustedError(
            f"Opening or creating semantic memory's partition made no progress "
            f"after {_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
        ) from lost_race

    @staticmethod
    async def _prepare_and_confirm_or_cancel(
        reservation: Reservation, vector_store: VectorStore
    ) -> Registration:
        """Prepare the reserved partition's storage and confirm the reservation, cancelling it if either raises or the creation is cancelled.

        The cancel acts only on a pending partition, so a confirmation that
        committed before its failure or cancellation was observed stands. A
        partition whose reservation the registry cannot cancel stays pending
        until it is deleted.
        """
        try:
            await vector_store.prepare_partition_incarnation(reservation.incarnation)
            return await reservation.confirm()
        except BaseException:
            # Shielded, so a cancelled creation still frees the name. The task
            # reports its own failure, since a creation cancelled again stops
            # awaiting it before it ends.
            cancellation = asyncio.create_task(reservation.cancel())
            _reservation_cancellations.add(cancellation)

            def finish(task: asyncio.Task[None]) -> None:
                _reservation_cancellations.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    logger.exception(
                        "Could not cancel the reservation of semantic memory's "
                        "partition after its creation failed or was cancelled; "
                        "it stays pending until deleted",
                        exc_info=task.exception(),
                    )

            cancellation.add_done_callback(finish)
            with contextlib.suppress(Exception):
                await asyncio.shield(cancellation)
            raise

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
