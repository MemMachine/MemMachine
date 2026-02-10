"""Configuration management interface for MemMachine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import requests

from memmachine.common.api.config_spec import (
    AddEmbedderSpec,
    AddLanguageModelSpec,
    DeleteResourceResponse,
    EpisodicMemoryConfigResponse,
    GetConfigResponse,
    LongTermMemoryConfigResponse,
    ResourcesStatus,
    SemanticMemoryConfigResponse,
    ShortTermMemoryConfigResponse,
    UpdateEpisodicMemorySpec,
    UpdateLongTermMemorySpec,
    UpdateMemoryConfigResponse,
    UpdateMemoryConfigSpec,
    UpdateResourceResponse,
    UpdateSemanticMemorySpec,
    UpdateShortTermMemorySpec,
)

if TYPE_CHECKING:
    from .client import MemMachineClient

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration interface for managing MemMachine server settings.

    Provides methods for reading and updating server configuration including
    resources (embedders, language models, rerankers) and memory settings.

    Example:
        ```python
        from memmachine import MemMachineClient

        client = MemMachineClient(base_url="http://localhost:8080")
        config = client.config()

        # Get full configuration
        cfg = config.get_config()

        # Add an embedder
        config.add_embedder(
            name="my-embedder",
            provider="openai",
            config={"api_key": "sk-...", "model": "text-embedding-3-small"},
        )
        ```

    """

    def __init__(self, client: MemMachineClient) -> None:
        """
        Initialize Config instance.

        Args:
            client: MemMachineClient instance

        """
        self.client = client

    def _check_closed(self) -> None:
        if self.client.closed:
            raise RuntimeError("Cannot use config: client has been closed")

    def get_config(self, timeout: int | None = None) -> GetConfigResponse:
        """
        Get the full server configuration.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            GetConfigResponse containing resources status

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "GET",
                f"{self.client.base_url}/api/v2/config",
                timeout=timeout,
            )
            response.raise_for_status()
            return GetConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to get config")
            raise

    def get_resources(self, timeout: int | None = None) -> ResourcesStatus:
        """
        Get the status of all configured resources.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            ResourcesStatus containing status of all resources

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "GET",
                f"{self.client.base_url}/api/v2/config/resources",
                timeout=timeout,
            )
            response.raise_for_status()
            return ResourcesStatus(**response.json())
        except requests.RequestException:
            logger.exception("Failed to get resources")
            raise

    def update_memory_config(
        self,
        episodic_memory: UpdateEpisodicMemorySpec | None = None,
        semantic_memory: UpdateSemanticMemorySpec | None = None,
        timeout: int | None = None,
    ) -> UpdateMemoryConfigResponse:
        """
        Update memory configuration settings.

        Args:
            episodic_memory: Episodic memory configuration updates
            semantic_memory: Semantic memory configuration updates
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateMemoryConfigResponse indicating success

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        spec = UpdateMemoryConfigSpec(
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
        )
        try:
            response = self.client.request(
                "PUT",
                f"{self.client.base_url}/api/v2/config/memory",
                json=spec.model_dump(exclude_none=True),
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to update memory config")
            raise

    def get_episodic_memory_config(
        self, timeout: int | None = None
    ) -> EpisodicMemoryConfigResponse:
        """
        Get the current episodic memory configuration.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            EpisodicMemoryConfigResponse containing current episodic memory settings
            including both long-term and short-term memory configuration

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "GET",
                f"{self.client.base_url}/api/v2/config/memory/episodic",
                timeout=timeout,
            )
            response.raise_for_status()
            return EpisodicMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to get episodic memory config")
            raise

    def update_episodic_memory_config(
        self,
        long_term_memory: UpdateLongTermMemorySpec | None = None,
        short_term_memory: UpdateShortTermMemorySpec | None = None,
        long_term_memory_enabled: bool | None = None,
        short_term_memory_enabled: bool | None = None,
        enabled: bool | None = None,
        timeout: int | None = None,
    ) -> UpdateMemoryConfigResponse:
        """
        Update episodic memory configuration.

        Args:
            long_term_memory: Long-term memory configuration updates
            short_term_memory: Short-term memory configuration updates
            long_term_memory_enabled: Whether long-term memory is enabled
            short_term_memory_enabled: Whether short-term memory is enabled
            enabled: Whether episodic memory is enabled overall
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateMemoryConfigResponse indicating success

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        spec = UpdateEpisodicMemorySpec(
            long_term_memory=long_term_memory,
            short_term_memory=short_term_memory,
            long_term_memory_enabled=long_term_memory_enabled,
            short_term_memory_enabled=short_term_memory_enabled,
            enabled=enabled,
        )
        payload = spec.model_dump(exclude_none=True)
        try:
            response = self.client.request(
                "PUT",
                f"{self.client.base_url}/api/v2/config/memory/episodic",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to update episodic memory config")
            raise

    def get_long_term_memory_config(
        self, timeout: int | None = None
    ) -> LongTermMemoryConfigResponse:
        """
        Get the current long-term memory configuration.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            LongTermMemoryConfigResponse containing current LTM settings

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "GET",
                f"{self.client.base_url}/api/v2/config/memory/episodic/long_term",
                timeout=timeout,
            )
            response.raise_for_status()
            return LongTermMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to get long-term memory config")
            raise

    def get_short_term_memory_config(
        self, timeout: int | None = None
    ) -> ShortTermMemoryConfigResponse:
        """
        Get the current short-term memory configuration.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            ShortTermMemoryConfigResponse containing current STM settings

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "GET",
                f"{self.client.base_url}/api/v2/config/memory/episodic/short_term",
                timeout=timeout,
            )
            response.raise_for_status()
            return ShortTermMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to get short-term memory config")
            raise

    def update_long_term_memory_config(
        self,
        embedder: str | None = None,
        reranker: str | None = None,
        vector_graph_store: str | None = None,
        enabled: bool | None = None,
        timeout: int | None = None,
    ) -> UpdateMemoryConfigResponse:
        """
        Update long-term memory configuration.

        Args:
            embedder: Name of the embedder to use for long-term memory
            reranker: Name of the reranker to use for long-term memory
            vector_graph_store: Name of the vector graph store to use
            enabled: Whether long-term memory is enabled
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateMemoryConfigResponse indicating success

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        spec = UpdateLongTermMemorySpec(
            embedder=embedder,
            reranker=reranker,
            vector_graph_store=vector_graph_store,
        )
        payload = spec.model_dump(exclude_none=True)
        if enabled is not None:
            payload["enabled"] = enabled
        try:
            response = self.client.request(
                "PUT",
                f"{self.client.base_url}/api/v2/config/memory/episodic/long_term",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to update long-term memory config")
            raise

    def update_short_term_memory_config(
        self,
        llm_model: str | None = None,
        message_capacity: int | None = None,
        enabled: bool | None = None,
        timeout: int | None = None,
    ) -> UpdateMemoryConfigResponse:
        """
        Update short-term memory configuration.

        Args:
            llm_model: Name of the language model to use for short-term memory
            message_capacity: Maximum number of messages to keep in short-term memory
            enabled: Whether short-term memory is enabled
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateMemoryConfigResponse indicating success

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        spec = UpdateShortTermMemorySpec(
            llm_model=llm_model,
            message_capacity=message_capacity,
        )
        payload = spec.model_dump(exclude_none=True)
        if enabled is not None:
            payload["enabled"] = enabled
        try:
            response = self.client.request(
                "PUT",
                f"{self.client.base_url}/api/v2/config/memory/episodic/short_term",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to update short-term memory config")
            raise

    def get_semantic_memory_config(
        self, timeout: int | None = None
    ) -> SemanticMemoryConfigResponse:
        """
        Get the current semantic memory configuration.

        Args:
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            SemanticMemoryConfigResponse containing current semantic memory settings

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "GET",
                f"{self.client.base_url}/api/v2/config/memory/semantic",
                timeout=timeout,
            )
            response.raise_for_status()
            return SemanticMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to get semantic memory config")
            raise

    def update_semantic_memory_config(
        self,
        enabled: bool | None = None,
        database: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        ingestion_trigger_messages: int | None = None,
        ingestion_trigger_age_seconds: int | None = None,
        cluster_similarity_threshold: float | None = None,
        cluster_max_time_gap_seconds: int | None = None,
        timeout: int | None = None,
    ) -> UpdateMemoryConfigResponse:
        """
        Update semantic memory configuration.

        Args:
            enabled: Whether semantic memory is enabled
            database: Name of the database to use for semantic memory
            llm_model: Name of the language model to use for feature extraction
            embedding_model: Name of the embedder to use for semantic similarity
            ingestion_trigger_messages: Number of messages before triggering ingestion
            ingestion_trigger_age_seconds: Age threshold in seconds for triggering ingestion
            cluster_similarity_threshold: Cosine similarity threshold for clustering
            cluster_max_time_gap_seconds: Maximum time gap in seconds for clustering
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateMemoryConfigResponse indicating success

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        spec = UpdateSemanticMemorySpec(
            enabled=enabled,
            database=database,
            llm_model=llm_model,
            embedding_model=embedding_model,
            ingestion_trigger_messages=ingestion_trigger_messages,
            ingestion_trigger_age_seconds=ingestion_trigger_age_seconds,
            cluster_similarity_threshold=cluster_similarity_threshold,
            cluster_max_time_gap_seconds=cluster_max_time_gap_seconds,
        )
        payload = spec.model_dump(exclude_none=True)
        try:
            response = self.client.request(
                "PUT",
                f"{self.client.base_url}/api/v2/config/memory/semantic",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateMemoryConfigResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to update semantic memory config")
            raise

    def add_embedder(
        self,
        name: str,
        provider: Literal["openai", "amazon-bedrock", "sentence-transformer"],
        config: dict[str, Any],
        timeout: int | None = None,
    ) -> UpdateResourceResponse:
        """
        Add a new embedder resource.

        Args:
            name: Name for the embedder
            provider: Provider type
            config: Provider-specific configuration
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateResourceResponse with status of the added resource

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        spec = AddEmbedderSpec(name=name, provider=provider, config=config)
        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/config/resources/embedders",
                json=spec.model_dump(),
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateResourceResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to add embedder '%s'", name)
            raise

    def add_language_model(
        self,
        name: str,
        provider: Literal[
            "openai-responses", "openai-chat-completions", "amazon-bedrock"
        ],
        config: dict[str, Any],
        timeout: int | None = None,
    ) -> UpdateResourceResponse:
        """
        Add a new language model resource.

        Args:
            name: Name for the language model
            provider: Provider type
            config: Provider-specific configuration
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateResourceResponse with status of the added resource

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        spec = AddLanguageModelSpec(name=name, provider=provider, config=config)
        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/config/resources/language_models",
                json=spec.model_dump(),
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateResourceResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to add language model '%s'", name)
            raise

    def delete_embedder(
        self,
        name: str,
        timeout: int | None = None,
    ) -> DeleteResourceResponse:
        """
        Delete an embedder resource.

        Args:
            name: Name of the embedder to delete
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            DeleteResourceResponse indicating success

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "DELETE",
                f"{self.client.base_url}/api/v2/config/resources/embedders/{name}",
                timeout=timeout,
            )
            response.raise_for_status()
            return DeleteResourceResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to delete embedder '%s'", name)
            raise

    def delete_language_model(
        self,
        name: str,
        timeout: int | None = None,
    ) -> DeleteResourceResponse:
        """
        Delete a language model resource.

        Args:
            name: Name of the language model to delete
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            DeleteResourceResponse indicating success

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "DELETE",
                f"{self.client.base_url}/api/v2/config/resources/language_models/{name}",
                timeout=timeout,
            )
            response.raise_for_status()
            return DeleteResourceResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to delete language model '%s'", name)
            raise

    def retry_embedder(
        self,
        name: str,
        timeout: int | None = None,
    ) -> UpdateResourceResponse:
        """
        Retry initialization of a failed embedder resource.

        Args:
            name: Name of the embedder to retry
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateResourceResponse with updated status

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/config/resources/embedders/{name}/retry",
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateResourceResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to retry embedder '%s'", name)
            raise

    def retry_language_model(
        self,
        name: str,
        timeout: int | None = None,
    ) -> UpdateResourceResponse:
        """
        Retry initialization of a failed language model resource.

        Args:
            name: Name of the language model to retry
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateResourceResponse with updated status

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/config/resources/language_models/{name}/retry",
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateResourceResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to retry language model '%s'", name)
            raise

    def retry_reranker(
        self,
        name: str,
        timeout: int | None = None,
    ) -> UpdateResourceResponse:
        """
        Retry initialization of a failed reranker resource.

        Args:
            name: Name of the reranker to retry
            timeout: Request timeout in seconds (uses client default if not provided)

        Returns:
            UpdateResourceResponse with updated status

        Raises:
            requests.RequestException: If the request fails
            RuntimeError: If the client has been closed

        """
        self._check_closed()
        try:
            response = self.client.request(
                "POST",
                f"{self.client.base_url}/api/v2/config/resources/rerankers/{name}/retry",
                timeout=timeout,
            )
            response.raise_for_status()
            return UpdateResourceResponse(**response.json())
        except requests.RequestException:
            logger.exception("Failed to retry reranker '%s'", name)
            raise

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"Config(client={self.client!r})"
