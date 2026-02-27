"""Configuration service for managing runtime resources."""

import logging
from datetime import timedelta
from typing import Any

from memmachine_common.api.config_spec import (
    EpisodicMemoryConfigResponse,
    LongTermMemoryConfigResponse,
    ResourceInfo,
    ResourcesStatus,
    ResourceStatus,
    SemanticMemoryConfigResponse,
    ShortTermMemoryConfigResponse,
    UpdateEpisodicMemorySpec,
    UpdateLongTermMemorySpec,
    UpdateSemanticMemorySpec,
    UpdateShortTermMemorySpec,
)
from pydantic import SecretStr

from memmachine_server.common.configuration import SemanticMemoryConf
from memmachine_server.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConf,
    EmbeddersConf,
    OpenAIEmbedderConf,
    SentenceTransformerEmbedderConf,
)
from memmachine_server.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
)
from memmachine_server.common.configuration.language_model_conf import (
    AmazonBedrockLanguageModelConf,
    LanguageModelsConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)
from memmachine_server.common.configuration.reranker_conf import RerankersConf
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)

logger = logging.getLogger(__name__)


def _get_embedder_provider(manager_conf: EmbeddersConf, name: str) -> str:
    """Determine the provider type for an embedder."""
    if name in manager_conf.openai:
        return "openai"
    if name in manager_conf.amazon_bedrock:
        return "amazon-bedrock"
    if name in manager_conf.sentence_transformer:
        return "sentence-transformer"
    return "unknown"


def _get_language_model_provider(manager_conf: LanguageModelsConf, name: str) -> str:
    """Determine the provider type for a language model."""
    if name in manager_conf.openai_responses_language_model_confs:
        return "openai-responses"
    if name in manager_conf.openai_chat_completions_language_model_confs:
        return "openai-chat-completions"
    if name in manager_conf.amazon_bedrock_language_model_confs:
        return "amazon-bedrock"
    return "unknown"


def _get_reranker_provider(manager_conf: RerankersConf, name: str) -> str:
    """Determine the provider type for a reranker."""
    if name in manager_conf.bm25:
        return "bm25"
    if name in manager_conf.cohere:
        return "cohere"
    if name in manager_conf.cross_encoder:
        return "cross-encoder"
    if name in manager_conf.amazon_bedrock:
        return "amazon-bedrock"
    if name in manager_conf.embedder:
        return "embedder"
    if name in manager_conf.identity:
        return "identity"
    if name in manager_conf.rrf_hybrid:
        return "rrf-hybrid"
    return "unknown"


def _create_embedder_config(
    provider: str, config: dict[str, Any]
) -> AmazonBedrockEmbedderConf | OpenAIEmbedderConf | SentenceTransformerEmbedderConf:
    """Create an embedder configuration object from provider and config dict."""
    if provider == "openai":
        # Convert api_key to SecretStr
        if "api_key" in config:
            config["api_key"] = SecretStr(config["api_key"])
        return OpenAIEmbedderConf(**config)
    if provider == "amazon-bedrock":
        # Convert AWS secrets to SecretStr
        for key in ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]:
            if key in config and config[key] is not None:
                config[key] = SecretStr(config[key])
        return AmazonBedrockEmbedderConf(**config)
    if provider == "sentence-transformer":
        return SentenceTransformerEmbedderConf(**config)
    raise ValueError(f"Unknown embedder provider: {provider}")


def _create_language_model_config(
    provider: str, config: dict[str, Any]
) -> (
    OpenAIResponsesLanguageModelConf
    | OpenAIChatCompletionsLanguageModelConf
    | AmazonBedrockLanguageModelConf
):
    """Create a language model configuration object from provider and config dict."""
    if provider == "openai-responses":
        # Convert api_key to SecretStr
        if "api_key" in config:
            config["api_key"] = SecretStr(config["api_key"])
        return OpenAIResponsesLanguageModelConf(**config)
    if provider == "openai-chat-completions":
        # Convert api_key to SecretStr
        if "api_key" in config:
            config["api_key"] = SecretStr(config["api_key"])
        return OpenAIChatCompletionsLanguageModelConf(**config)
    if provider == "amazon-bedrock":
        # Convert AWS secrets to SecretStr
        for key in ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]:
            if key in config and config[key] is not None:
                config[key] = SecretStr(config[key])
        return AmazonBedrockLanguageModelConf(**config)
    raise ValueError(f"Unknown language model provider: {provider}")


def _apply_ltm_updates(
    em: EpisodicMemoryConfPartial,
    spec_ltm: UpdateLongTermMemorySpec,
) -> list[str]:
    """Apply long-term memory updates and return change descriptions."""
    from memmachine_server.common.configuration.episodic_config import (
        LongTermMemoryConfPartial,
    )

    ltm = em.long_term_memory
    if ltm is None:
        ltm = LongTermMemoryConfPartial()
        em.long_term_memory = ltm

    changes: list[str] = []
    if spec_ltm.embedder is not None:
        ltm.embedder = spec_ltm.embedder
        changes.append(f"episodic_memory.long_term_memory.embedder={spec_ltm.embedder}")
    if spec_ltm.reranker is not None:
        ltm.reranker = spec_ltm.reranker
        changes.append(f"episodic_memory.long_term_memory.reranker={spec_ltm.reranker}")
    if spec_ltm.vector_graph_store is not None:
        ltm.vector_graph_store = spec_ltm.vector_graph_store
        changes.append(
            f"episodic_memory.long_term_memory.vector_graph_store={spec_ltm.vector_graph_store}"
        )
    return changes


def _apply_stm_updates(
    em: EpisodicMemoryConfPartial,
    spec_stm: UpdateShortTermMemorySpec,
) -> list[str]:
    """Apply short-term memory updates and return change descriptions."""
    from memmachine_server.common.configuration.episodic_config import (
        ShortTermMemoryConfPartial,
    )

    stm = em.short_term_memory
    if stm is None:
        stm = ShortTermMemoryConfPartial()
        em.short_term_memory = stm

    changes: list[str] = []
    if spec_stm.llm_model is not None:
        stm.llm_model = spec_stm.llm_model
        changes.append(
            f"episodic_memory.short_term_memory.llm_model={spec_stm.llm_model}"
        )
    if spec_stm.message_capacity is not None:
        stm.message_capacity = spec_stm.message_capacity
        changes.append(
            f"episodic_memory.short_term_memory.message_capacity={spec_stm.message_capacity}"
        )
    return changes


def _apply_episodic_memory_updates(
    em: EpisodicMemoryConfPartial,
    spec: UpdateEpisodicMemorySpec,
) -> list[str]:
    """Apply episodic memory updates and return list of change descriptions."""
    changes: list[str] = []

    if spec.long_term_memory is not None:
        changes.extend(_apply_ltm_updates(em, spec.long_term_memory))
    if spec.short_term_memory is not None:
        changes.extend(_apply_stm_updates(em, spec.short_term_memory))

    if spec.long_term_memory_enabled is not None:
        em.long_term_memory_enabled = spec.long_term_memory_enabled
        changes.append(
            f"episodic_memory.long_term_memory_enabled={spec.long_term_memory_enabled}"
        )
    if spec.short_term_memory_enabled is not None:
        em.short_term_memory_enabled = spec.short_term_memory_enabled
        changes.append(
            f"episodic_memory.short_term_memory_enabled={spec.short_term_memory_enabled}"
        )
    if spec.enabled is not None:
        em.enabled = spec.enabled
        changes.append(f"episodic_memory.enabled={spec.enabled}")

    return changes


def _apply_semantic_memory_updates(
    sm: SemanticMemoryConf,
    spec: UpdateSemanticMemorySpec,
) -> list[str]:
    """Apply semantic memory updates and return list of change descriptions."""
    changes: list[str] = []

    if spec.enabled is not None:
        sm.enabled = spec.enabled
        changes.append(f"semantic_memory.enabled={spec.enabled}")
    if spec.database is not None:
        sm.database = spec.database
        changes.append(f"semantic_memory.database={spec.database}")
    if spec.llm_model is not None:
        sm.llm_model = spec.llm_model
        changes.append(f"semantic_memory.llm_model={spec.llm_model}")
    if spec.embedding_model is not None:
        sm.embedding_model = spec.embedding_model
        changes.append(f"semantic_memory.embedding_model={spec.embedding_model}")
    if spec.ingestion_trigger_messages is not None:
        sm.ingestion_trigger_messages = spec.ingestion_trigger_messages
        changes.append(
            f"semantic_memory.ingestion_trigger_messages={spec.ingestion_trigger_messages}"
        )
    if spec.ingestion_trigger_age_seconds is not None:
        sm.ingestion_trigger_age = timedelta(seconds=spec.ingestion_trigger_age_seconds)
        changes.append(
            f"semantic_memory.ingestion_trigger_age={spec.ingestion_trigger_age_seconds}s"
        )

    return changes


class ConfigService:
    """Service for managing runtime configuration resources."""

    def __init__(self, resource_manager: ResourceManagerImpl) -> None:
        """Initialize the config service with a resource manager."""
        self._resource_manager = resource_manager

    def _persist_config(self) -> None:
        """Persist configuration changes to file if a config file path is set."""
        try:
            if self._resource_manager.config.config_file_path:
                self._resource_manager.save_config()
        except Exception as e:
            logger.warning("Failed to persist configuration to file: %s", e)

    def get_resources_status(self) -> ResourcesStatus:
        """Get the status of all configured resources."""
        resource_manager = self._resource_manager
        embedders: list[ResourceInfo] = []
        language_models: list[ResourceInfo] = []
        rerankers: list[ResourceInfo] = []
        databases: list[ResourceInfo] = []

        # Get embedder status
        embedder_manager = resource_manager.embedder_manager
        for name in embedder_manager.get_all_names():
            status = embedder_manager.get_resource_status(name)
            error = embedder_manager.get_resource_error(name)
            provider = _get_embedder_provider(embedder_manager.conf, name)
            embedders.append(
                ResourceInfo(
                    name=name,
                    provider=provider,
                    status=ResourceStatus(status.value),
                    error=str(error) if error else None,
                )
            )

        # Get language model status
        lm_manager = resource_manager.language_model_manager
        for name in lm_manager.get_all_names():
            status = lm_manager.get_resource_status(name)
            error = lm_manager.get_resource_error(name)
            provider = _get_language_model_provider(lm_manager.conf, name)
            language_models.append(
                ResourceInfo(
                    name=name,
                    provider=provider,
                    status=ResourceStatus(status.value),
                    error=str(error) if error else None,
                )
            )

        # Get reranker status
        reranker_manager = resource_manager.reranker_manager
        for name in reranker_manager.get_all_names():
            status = reranker_manager.get_resource_status(name)
            error = reranker_manager.get_resource_error(name)
            provider = _get_reranker_provider(reranker_manager.conf, name)
            rerankers.append(
                ResourceInfo(
                    name=name,
                    provider=provider,
                    status=ResourceStatus(status.value),
                    error=str(error) if error else None,
                )
            )

        # Get database status - databases are always ready if they exist
        # (they fail hard at startup if unavailable)
        db_manager = resource_manager.database_manager
        databases.extend(
            ResourceInfo(
                name=name,
                provider="neo4j",
                status=ResourceStatus.READY,
                error=None,
            )
            for name in db_manager.conf.neo4j_confs
        )
        databases.extend(
            ResourceInfo(
                name=name,
                provider="sqlite" if conf.dialect == "sqlite" else "postgres",
                status=ResourceStatus.READY,
                error=None,
            )
            for name, conf in db_manager.conf.relational_db_confs.items()
        )

        return ResourcesStatus(
            embedders=embedders,
            language_models=language_models,
            rerankers=rerankers,
            databases=databases,
        )

    async def add_embedder(
        self,
        name: str,
        provider: str,
        config: dict[str, Any],
    ) -> ResourceStatus:
        """Add a new embedder configuration and build it."""
        embedder_config = _create_embedder_config(provider, config)
        self._resource_manager.embedder_manager.add_embedder_config(
            name, provider, embedder_config
        )

        # Persist configuration to file
        self._persist_config()

        # Attempt to build it immediately
        try:
            await self._resource_manager.embedder_manager.get_embedder(name)
        except Exception as e:
            logger.warning("Failed to build new embedder '%s': %s", name, e)
            return ResourceStatus.FAILED
        else:
            return ResourceStatus.READY

    async def add_language_model(
        self,
        name: str,
        provider: str,
        config: dict[str, Any],
    ) -> ResourceStatus:
        """Add a new language model configuration and build it."""
        lm_config = _create_language_model_config(provider, config)
        self._resource_manager.language_model_manager.add_language_model_config(
            name, provider, lm_config
        )

        # Persist configuration to file
        self._persist_config()

        # Attempt to build it immediately
        try:
            await self._resource_manager.language_model_manager.get_language_model(name)
        except Exception as e:
            logger.warning("Failed to build new language model '%s': %s", name, e)
            return ResourceStatus.FAILED
        else:
            return ResourceStatus.READY

    def remove_embedder(self, name: str) -> bool:
        """Remove an embedder from the manager."""
        removed = self._resource_manager.embedder_manager.remove_embedder(name)
        if removed:
            self._persist_config()
        return removed

    def remove_language_model(self, name: str) -> bool:
        """Remove a language model from the manager."""
        removed = self._resource_manager.language_model_manager.remove_language_model(
            name
        )
        if removed:
            self._persist_config()
        return removed

    async def retry_embedder(self, name: str) -> ResourceStatus:
        """
        Retry building a failed embedder.

        Raises:
            InvalidEmbedderError: If the embedder is not configured.

        """
        from memmachine_server.common.errors import InvalidEmbedderError

        try:
            await self._resource_manager.embedder_manager.retry_build(name)
        except InvalidEmbedderError:
            raise
        except Exception as e:
            logger.warning("Failed to retry building embedder '%s': %s", name, e)
            return ResourceStatus.FAILED
        else:
            return ResourceStatus.READY

    async def retry_language_model(self, name: str) -> ResourceStatus:
        """
        Retry building a failed language model.

        Raises:
            InvalidLanguageModelError: If the language model is not configured.

        """
        from memmachine_server.common.errors import InvalidLanguageModelError

        try:
            await self._resource_manager.language_model_manager.retry_build(name)
        except InvalidLanguageModelError:
            raise
        except Exception as e:
            logger.warning("Failed to retry building language model '%s': %s", name, e)
            return ResourceStatus.FAILED
        else:
            return ResourceStatus.READY

    async def retry_reranker(self, name: str) -> ResourceStatus:
        """
        Retry building a failed reranker.

        Raises:
            InvalidRerankerError: If the reranker is not configured.

        """
        from memmachine_server.common.errors import InvalidRerankerError

        try:
            await self._resource_manager.reranker_manager.retry_build(name)
        except InvalidRerankerError:
            raise
        except Exception as e:
            logger.warning("Failed to retry building reranker '%s': %s", name, e)
            return ResourceStatus.FAILED
        else:
            return ResourceStatus.READY

    def update_memory_config(
        self,
        episodic_memory: UpdateEpisodicMemorySpec | None,
        semantic_memory: UpdateSemanticMemorySpec | None,
    ) -> str:
        """
        Update episodic and/or semantic memory configuration in-place.

        Only supplied (non-None) fields are updated; the rest stay unchanged.
        Returns a human-readable summary of what was changed.
        """
        config = self._resource_manager.config
        changes: list[str] = []

        if episodic_memory is not None:
            changes.extend(
                _apply_episodic_memory_updates(config.episodic_memory, episodic_memory)
            )

        if semantic_memory is not None:
            changes.extend(
                _apply_semantic_memory_updates(config.semantic_memory, semantic_memory)
            )

        if changes:
            self._persist_config()
            return "Updated: " + ", ".join(changes)
        return "No changes applied."

    def get_embedder_error(self, name: str) -> str | None:
        """Get the error message for a failed embedder."""
        error = self._resource_manager.embedder_manager.get_resource_error(name)
        return str(error) if error else None

    def get_language_model_error(self, name: str) -> str | None:
        """Get the error message for a failed language model."""
        error = self._resource_manager.language_model_manager.get_resource_error(name)
        return str(error) if error else None

    def get_reranker_error(self, name: str) -> str | None:
        """Get the error message for a failed reranker."""
        error = self._resource_manager.reranker_manager.get_resource_error(name)
        return str(error) if error else None

    def get_episodic_memory_config(self) -> EpisodicMemoryConfigResponse:
        """Get the current episodic memory configuration."""
        config = self._resource_manager.config
        em = config.episodic_memory

        ltm_config = LongTermMemoryConfigResponse(
            embedder=em.long_term_memory.embedder if em.long_term_memory else None,
            reranker=em.long_term_memory.reranker if em.long_term_memory else None,
            vector_graph_store=em.long_term_memory.vector_graph_store
            if em.long_term_memory
            else None,
            enabled=em.long_term_memory_enabled
            if em.long_term_memory_enabled is not None
            else True,
        )

        stm_config = ShortTermMemoryConfigResponse(
            llm_model=em.short_term_memory.llm_model if em.short_term_memory else None,
            message_capacity=em.short_term_memory.message_capacity
            if em.short_term_memory
            else None,
            enabled=em.short_term_memory_enabled
            if em.short_term_memory_enabled is not None
            else True,
        )

        return EpisodicMemoryConfigResponse(
            long_term_memory=ltm_config,
            short_term_memory=stm_config,
            enabled=em.enabled if em.enabled is not None else True,
        )

    def get_semantic_memory_config(self) -> SemanticMemoryConfigResponse:
        """Get the current semantic memory configuration."""
        config = self._resource_manager.config
        sm = config.semantic_memory

        return SemanticMemoryConfigResponse(
            enabled=sm.enabled if sm.enabled is not None else False,
            database=sm.database,
            llm_model=sm.llm_model,
            embedding_model=sm.embedding_model,
        )

    def get_long_term_memory_config(self) -> LongTermMemoryConfigResponse:
        """Get the current long-term memory configuration."""
        config = self._resource_manager.config
        em = config.episodic_memory

        return LongTermMemoryConfigResponse(
            embedder=em.long_term_memory.embedder if em.long_term_memory else None,
            reranker=em.long_term_memory.reranker if em.long_term_memory else None,
            vector_graph_store=em.long_term_memory.vector_graph_store
            if em.long_term_memory
            else None,
            enabled=em.long_term_memory_enabled
            if em.long_term_memory_enabled is not None
            else True,
        )

    def get_short_term_memory_config(self) -> ShortTermMemoryConfigResponse:
        """Get the current short-term memory configuration."""
        config = self._resource_manager.config
        em = config.episodic_memory

        return ShortTermMemoryConfigResponse(
            llm_model=em.short_term_memory.llm_model if em.short_term_memory else None,
            message_capacity=em.short_term_memory.message_capacity
            if em.short_term_memory
            else None,
            enabled=em.short_term_memory_enabled
            if em.short_term_memory_enabled is not None
            else True,
        )

    def update_long_term_memory_config(
        self,
        spec: UpdateLongTermMemorySpec,
        enabled: bool | None = None,
    ) -> str:
        """
        Update long-term memory configuration.

        Only supplied (non-None) fields are updated; the rest stay unchanged.
        Returns a human-readable summary of what was changed.
        """
        config = self._resource_manager.config
        em = config.episodic_memory
        changes: list[str] = []

        changes.extend(_apply_ltm_updates(em, spec))

        if enabled is not None:
            em.long_term_memory_enabled = enabled
            changes.append(f"long_term_memory_enabled={enabled}")

        if changes:
            self._persist_config()
            return "Updated: " + ", ".join(changes)
        return "No changes applied."

    def update_short_term_memory_config(
        self,
        spec: UpdateShortTermMemorySpec,
        enabled: bool | None = None,
    ) -> str:
        """
        Update short-term memory configuration.

        Only supplied (non-None) fields are updated; the rest stay unchanged.
        Returns a human-readable summary of what was changed.
        """
        config = self._resource_manager.config
        em = config.episodic_memory
        changes: list[str] = []

        changes.extend(_apply_stm_updates(em, spec))

        if enabled is not None:
            em.short_term_memory_enabled = enabled
            changes.append(f"short_term_memory_enabled={enabled}")

        if changes:
            self._persist_config()
            return "Updated: " + ", ".join(changes)
        return "No changes applied."

    def update_semantic_memory_config(
        self,
        spec: UpdateSemanticMemorySpec,
    ) -> str:
        """
        Update semantic memory configuration.

        Only supplied (non-None) fields are updated; the rest stay unchanged.
        Returns a human-readable summary of what was changed.
        """
        config = self._resource_manager.config
        sm = config.semantic_memory
        changes: list[str] = []

        changes.extend(_apply_semantic_memory_updates(sm, spec))

        if changes:
            self._persist_config()
            return "Updated: " + ", ".join(changes)
        return "No changes applied."

    def update_episodic_memory_config(
        self,
        spec: UpdateEpisodicMemorySpec,
    ) -> str:
        """
        Update episodic memory configuration.

        Only supplied (non-None) fields are updated; the rest stay unchanged.
        Returns a human-readable summary of what was changed.
        """
        config = self._resource_manager.config
        em = config.episodic_memory
        changes: list[str] = []

        changes.extend(_apply_episodic_memory_updates(em, spec))

        if changes:
            self._persist_config()
            return "Updated: " + ", ".join(changes)
        return "No changes applied."
