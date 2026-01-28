"""Manager for semantic memory resources and services."""

import asyncio
from datetime import UTC, datetime

from pydantic import InstanceOf

from memmachine.common.configuration import PromptConf, SemanticMemoryConf
from memmachine.common.episode_store import EpisodeStorage
from memmachine.common.resource_manager import CommonResourceManager
from memmachine.common.errors import ModelUnavailableError
from memmachine.common.embedder import Embedder
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    ResourceRetriever,
    Resources,
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

        self._semantic_session_resource_manager: (
            InstanceOf[ResourceRetriever] | None
        ) = None
        self._semantic_service: SemanticService | None = None
        self._semantic_session_manager: SemanticSessionManager | None = None
        self._reembed_task: asyncio.Task | None = None
        self._reembed_status: dict[str, object] | None = None
        self._reembed_lock = asyncio.Lock()

    async def close(self) -> None:
        """Stop semantic services if they were started."""
        tasks = []

        if self._semantic_service is not None:
            tasks.append(self._semantic_service.stop())

        await asyncio.gather(*tasks)

    async def get_semantic_session_resource_manager(
        self,
    ) -> InstanceOf[ResourceRetriever]:
        """Return a resource retriever for semantic sessions."""
        if self._semantic_session_resource_manager is not None:
            return self._semantic_session_resource_manager

        if not self._conf.enabled:
            raise ModelUnavailableError("Semantic memory is disabled.")
        if not self._conf.embedding_model or not self._conf.llm_model:
            raise ModelUnavailableError("Chat model or embedding model is not configured.")

        semantic_categories_by_isolation = self._prompt_conf.default_semantic_categories

        default_embedder = await self._resource_manager.get_embedder(
            self._conf.embedding_model,
            validate=True,
        )
        default_model = await self._resource_manager.get_language_model(
            self._conf.llm_model,
            validate=True,
        )

        class SemanticResourceRetriever:
            def get_resources(self, set_id: SetIdT) -> Resources:
                isolation_type = SemanticSessionManager.set_id_isolation_type(set_id)

                return Resources(
                    language_model=default_model,
                    embedder=default_embedder,
                    semantic_categories=semantic_categories_by_isolation[
                        isolation_type
                    ],
                )

        self._semantic_session_resource_manager = SemanticResourceRetriever()
        return self._semantic_session_resource_manager

    def reset_resource_retriever(self) -> None:
        """Reset cached semantic resource retriever to pick up new defaults."""
        self._semantic_session_resource_manager = None

    def get_reindex_status(self) -> dict[str, object] | None:
        """Return current reindex status snapshot."""
        return self._reembed_status

    async def _reembed_all_features(self, embedder: Embedder) -> None:
        semantic_service = await self.get_semantic_service()
        processed = await semantic_service.reembed_all_features(embedder=embedder)

        async with self._reembed_lock:
            if self._reembed_status is not None:
                self._reembed_status["status"] = "completed"
                self._reembed_status["completed_at"] = datetime.now(tz=UTC).isoformat()
                self._reembed_status["processed"] = processed

    async def schedule_reembedding(self) -> None:
        if not self._conf.embedding_model or not self._conf.enabled:
            return

        embedder = await self._resource_manager.get_embedder(
            self._conf.embedding_model,
            validate=True,
        )

        async with self._reembed_lock:
            if self._reembed_task is not None and not self._reembed_task.done():
                self._reembed_task.cancel()
            self._reembed_status = {
                "status": "running",
                "model_id": self._conf.embedding_model,
                "started_at": datetime.now(tz=UTC).isoformat(),
                "processed": 0,
            }
            self._reembed_task = asyncio.create_task(
                self._reembed_all_features(embedder)
            )

    async def _get_semantic_storage(self) -> SemanticStorage:
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

    async def get_semantic_service(self) -> SemanticService:
        """Return the semantic service, constructing it if needed."""
        if self._semantic_service is not None:
            return self._semantic_service

        semantic_storage = await self._get_semantic_storage()
        episode_store = self._episode_storage
        resource_retriever = await self.get_semantic_session_resource_manager()

        self._semantic_service = SemanticService(
            SemanticService.Params(
                semantic_storage=semantic_storage,
                episode_storage=episode_store,
                resource_retriever=resource_retriever,
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
            await self.get_semantic_service(),
        )
        return self._semantic_session_manager
