"""Protocols for accessing shared MemMachine resources."""

from typing import Protocol, runtime_checkable

from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.reranker import Reranker
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.common.vector_graph_store import VectorGraphStore


@runtime_checkable
class CommonResourceManager(Protocol):
    """Protocol for constructing and retrieving shared resources."""

    async def build(self) -> None:
        """Construct underlying resource instances."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release resources and close connections."""
        raise NotImplementedError

    async def get_sql_engine(self, name: str) -> AsyncEngine:
        """Return the SQL engine by name."""
        raise NotImplementedError

    async def get_neo4j_driver(self, name: str) -> AsyncDriver:
        """Return the Neo4j driver by name."""
        raise NotImplementedError

    async def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        """Return the vector graph store by name."""
        raise NotImplementedError

    async def get_embedder(self, name: str) -> Embedder:
        """Return the embedder by name."""
        raise NotImplementedError

    async def get_language_model(self, name: str) -> LanguageModel:
        """Return the language model by name."""
        raise NotImplementedError

    async def get_reranker(self, name: str) -> Reranker:
        """Return the reranker by name."""
        raise NotImplementedError

    async def get_metrics_factory(self, name: str) -> MetricsFactory:
        """Return the metrics factory by name."""
        raise NotImplementedError

    async def get_session_data_manager(self) -> SessionDataManager:
        """Return the session data manager."""
        raise NotImplementedError
