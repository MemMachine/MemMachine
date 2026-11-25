"""Core MemMachine orchestration logic."""

import asyncio
import contextlib
import logging
from asyncio import Task
from collections.abc import (
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    Sequence,
)
from datetime import UTC, datetime
from typing import Any, Final, Protocol
from uuid import UUID, uuid4

from memmachine_common.api import MemoryType
from pydantic import BaseModel, InstanceOf, JsonValue, ValidationError

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.configuration.episodic_config import (
    EpisodicMemoryConf,
    EpisodicMemoryConfPartial,
    EventLongTermMemoryConf,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine_server.common.configuration.retrieval_config import RetrievalAgentConf
from memmachine_server.common.episode_store import (
    Episode,
    EpisodeEntry,
    EpisodeResponse,
)
from memmachine_server.common.errors import (
    ConfigurationError,
    InvalidArgumentError,
    ResourceNotFoundError,
    ResourceNotReadyError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from memmachine_server.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    parse_filter,
)
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.partition_registry import (
    ListedPartition,
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionDeletedError,
    PartitionManager,
    PartitionPendingError,
    Registration,
)
from memmachine_server.common.property_keys import validate_user_property_key
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.common.session_manager.session_data_manager import (
    SessionDataManager,
)
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    deployment_event_memory_ids,
)
from memmachine_server.retrieval_agent import create_retrieval_agent
from memmachine_server.retrieval_agent.common.agent_api import (
    AgentToolBase,
    QueryParam,
    QueryPolicy,
)
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.semantic_memory import with_has_set_ids
from memmachine_server.semantic_memory.semantic_model import (
    CategoryIdT,
    FeatureIdT,
    SemanticCategory,
    SemanticFeature,
    SetIdT,
    SetTypeEntry,
    TagIdT,
)
from memmachine_server.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage

from .partitioned_stores import (
    EpisodePartitionedStore,
    LongTermMemoryPartitionedStore,
    SessionDataPartitionedStore,
)

logger = logging.getLogger(__name__)


ALL_MEMORY_TYPES: Final[list[MemoryType]] = list(MemoryType)

_PARTITION_NAMESPACE = "memmachine"
"""The partition namespace MemMachine owns, the same in every deployment.

Fixed, because the namespace locates the partitions' registrations.
"""

# The attempts `_get_or_create_partition` makes,
# _GET_OR_CREATE_RETRY_DELAY_SECONDS apart, before giving up on a partition
# that stays pending or a name other creators and deleters keep taking.
_MAX_GET_OR_CREATE_ATTEMPTS = 10
_GET_OR_CREATE_RETRY_DELAY_SECONDS = 1


def _raise_add_episode_errors(
    task_names: Sequence[str], results: Sequence[object], episode_ids: list[UUID]
) -> None:
    """Log every failed write and propagate the first failure."""
    first_error: BaseException | None = None
    for task_name, result in zip(task_names, results, strict=True):
        if isinstance(result, BaseException):
            logger.error(
                "Failed to add episodes %s to %s",
                episode_ids,
                task_name,
                exc_info=(type(result), result, result.__traceback__),
            )
            if first_error is None:
                first_error = result
    if first_error is not None:
        raise first_error


class MemMachine:
    """MemMachine class."""

    class SessionData(Protocol):
        """Protocol describing session-scoped context used by memories."""

        @property
        def org_id(self) -> str: ...

        @property
        def project_id(self) -> str: ...

        @property
        def session_key(self) -> str: ...

    def __init__(
        self, conf: Configuration, resources: ResourceManagerImpl | None = None
    ) -> None:
        """
        Create a MemMachine using the provided configuration.

        Args:
            conf: Application configuration.
            resources: Optional resource manager override.

        Returns:
            None.

        """
        self._conf = conf
        if resources is not None:
            self._resources = resources
        else:
            self._resources = ResourceManagerImpl(conf)
        self._initialize_default_episodic_configuration()
        self._initialize_default_retrieval_agent_configuration()
        self._retrieval_agent: AgentToolBase | None = None
        self._retrieval_agent_lock = asyncio.Lock()
        self._started = False

    def _initialize_default_episodic_configuration(self) -> None:
        """
        Initialize missing episodic memory configuration defaults.

        Returns:
            None.

        """
        # initialize the default value for episodic memory configuration
        # Can not put the logic into the data type

        if self._conf.episodic_memory is None:
            self._conf.episodic_memory = EpisodicMemoryConfPartial()
            self._conf.episodic_memory.enabled = False
        if self._conf.episodic_memory.long_term_memory is None:
            self._conf.episodic_memory.long_term_memory = LongTermMemoryConfPartial()
            self._conf.episodic_memory.long_term_memory_enabled = False
        if self._conf.episodic_memory.short_term_memory is None:
            self._conf.episodic_memory.short_term_memory = ShortTermMemoryConfPartial()
            self._conf.episodic_memory.short_term_memory_enabled = False

        # Always resolve defaults so that merge() can produce valid full
        # configs.  _resolve_*_defaults() will gracefully disable the
        # subsystem if required resources are missing.
        self._resolve_ltm_defaults()
        self._resolve_stm_defaults()

    def _resolve_ltm_defaults(self) -> None:
        """Resolve long-term memory defaults, disabling if resources are unavailable."""
        ltm = self._conf.episodic_memory.long_term_memory
        assert ltm is not None

        ltm.embedder = self._resolve_ltm_resource_default(
            current_value=ltm.embedder,
            default_getter=lambda: self._conf.default_long_term_memory_embedder,
            missing_warning=(
                "No default embedder configured; disabling long-term episodic memory."
            ),
        )
        if ltm.embedder is None:
            return

        # Reranker is required for declarative; optional for event.
        # `backend is None` means the writer predates the discriminator, so
        # treat it as declarative for backwards compatibility (matches the
        # parse-time default).
        backend = ltm.backend if ltm.backend is not None else "declarative"
        if backend == "declarative":
            ltm.reranker = self._resolve_ltm_resource_default(
                current_value=ltm.reranker,
                default_getter=lambda: self._conf.default_long_term_memory_reranker,
                missing_warning=(
                    "No default reranker configured; disabling long-term episodic memory."
                ),
            )
            if ltm.reranker is None:
                return

            if ltm.vector_graph_store is None:
                ltm.vector_graph_store = "default_store"
        else:
            # Event backend: reranker is optional; only resolve if explicit or
            # available as a default.
            if ltm.reranker is None:
                try:
                    ltm.reranker = self._conf.default_long_term_memory_reranker
                except Exception:
                    ltm.reranker = None

            # vector_store, segment_store, and partition_registry are required
            # to materialize an event-backed long-term memory. If any is
            # missing, disable.
            missing = [
                name
                for name, value in (
                    ("vector_store", ltm.vector_store),
                    ("segment_store", ltm.segment_store),
                    ("partition_registry", ltm.partition_registry),
                )
                if value is None
            ]
            if missing:
                self._disable_long_term_memory(
                    "Event-backed long-term memory requires vector_store, "
                    f"segment_store, and partition_registry; missing: "
                    f"{', '.join(missing)}. Disabling long-term episodic memory."
                )

    def _disable_long_term_memory(self, warning_message: str) -> None:
        """Disable long-term memory and emit a warning message."""
        logger.warning(warning_message)
        self._conf.episodic_memory.long_term_memory_enabled = False

    def _resolve_ltm_resource_default(
        self,
        *,
        current_value: str | None,
        default_getter: Callable[[], str],
        missing_warning: str,
    ) -> str | None:
        """Return configured value or fallback default; disable long-term memory on failure."""
        if current_value is not None:
            return current_value
        try:
            return default_getter()
        except Exception:
            self._disable_long_term_memory(missing_warning)
            return None

    def _resolve_stm_defaults(self) -> None:
        """Resolve short-term memory defaults."""
        stm = self._conf.episodic_memory.short_term_memory
        assert stm is not None

        if stm.llm_model is None:
            stm.llm_model = "gpt-4.1"
        if stm.summary_prompt_system is None:
            stm.summary_prompt_system = self._conf.prompt.episode_summary_system_prompt
        if stm.summary_prompt_user is None:
            stm.summary_prompt_user = self._conf.prompt.episode_summary_user_prompt

    def _initialize_default_retrieval_agent_configuration(self) -> None:
        """Initialize retrieval-agent defaults independent of episodic config."""
        retrieval_conf = getattr(self._conf, "retrieval_agent", None)
        if retrieval_conf is None:
            self._conf.retrieval_agent = RetrievalAgentConf()
            retrieval_conf = self._conf.retrieval_agent

        if retrieval_conf.llm_model is None:
            retrieval_conf.llm_model = self._resolve_retrieval_llm_model_fallback()
        if retrieval_conf.reranker is None:
            retrieval_conf.reranker = self._resolve_retrieval_reranker_fallback()

        if (
            retrieval_conf.llm_model is not None
            and not self._conf.resources.language_models.contains_language_model(
                retrieval_conf.llm_model
            )
        ):
            logger.warning(
                "Retrieval agent disabled: language model '%s' is not configured.",
                retrieval_conf.llm_model,
            )
            retrieval_conf.llm_model = None

        if (
            retrieval_conf.reranker is not None
            and not self._conf.resources.rerankers.contains_reranker(
                retrieval_conf.reranker
            )
        ):
            logger.warning(
                "Retrieval agent disabled: reranker '%s' is not configured.",
                retrieval_conf.reranker,
            )
            retrieval_conf.reranker = None

    def _resolve_retrieval_llm_model_fallback(self) -> str | None:
        """Resolve retrieval-agent LLM from existing memory configuration."""
        candidates: list[str] = []
        stm = self._conf.episodic_memory.short_term_memory
        if stm is not None and stm.llm_model is not None:
            candidates.append(stm.llm_model)

        semantic_llm = self._conf.semantic_memory.llm_model
        if semantic_llm is not None:
            candidates.append(semantic_llm)

        for llm_name in candidates:
            if self._conf.resources.language_models.contains_language_model(llm_name):
                return llm_name
        return None

    def _resolve_retrieval_reranker_fallback(self) -> str | None:
        """Resolve retrieval-agent reranker from long-term memory defaults."""
        ltm = self._conf.episodic_memory.long_term_memory
        if (
            ltm is not None
            and ltm.reranker is not None
            and self._conf.resources.rerankers.contains_reranker(ltm.reranker)
        ):
            return ltm.reranker
        return None

    async def _get_retrieval_agent(self) -> AgentToolBase | None:
        """Lazily initialize the retrieval-agent instance from top-level config."""
        conf = self._conf.retrieval_agent
        if conf.llm_model is None or conf.reranker is None:
            return None
        if self._retrieval_agent is not None:
            return self._retrieval_agent

        async with self._retrieval_agent_lock:
            if self._retrieval_agent is not None:
                return self._retrieval_agent
            try:
                llm_model = await self._resources.get_language_model(conf.llm_model)
                reranker = await self._resources.get_reranker(conf.reranker)
            except Exception:
                logger.exception("Failed to initialize retrieval agent resources.")
                return None

            try:
                self._retrieval_agent = create_retrieval_agent(
                    model=llm_model,
                    reranker=reranker,
                    use_optimized_coq=conf.use_optimized_coq,
                    optimized_coq=conf.optimized_coq,
                )
            except Exception:
                logger.exception("Failed to initialize retrieval agent.")
                return None

        return self._retrieval_agent

    async def _partition_manager(
        self,
        create_session_record: Callable[[str], Awaitable[None]] | None = None,
    ) -> PartitionManager:
        """
        Build the partition manager of MemMachine's partition namespace.

        A session is a partition of the namespace, named by its session key.
        Every component holds a session's data under the string of the
        partition's UUID, so a session created again under a key shares
        nothing with the session deleted before it.

        Every call gives stores over the same components, so the managers of
        every process, and the purge rounds the resource manager runs, purge
        and drop the same data. A manager that creates a partition is built
        with the function that writes the partition's record.
        """
        episode_store = EpisodePartitionedStore(self._resources)
        long_term_memory_store = LongTermMemoryPartitionedStore(self._resources)
        partition_registry = await self._resources.get_partition_registry(
            self._conf.session_manager.database,
            partition_namespace=_PARTITION_NAMESPACE,
            tombstone_retention_seconds=(
                self._conf.session_manager.tombstone_retention_seconds
            ),
            # Purge rounds create no partition, so their store takes no record.
            partitioned_stores=[
                SessionDataPartitionedStore(self._resources),
                episode_store,
                long_term_memory_store,
            ],
        )
        return PartitionManager(
            partition_registry,
            [
                SessionDataPartitionedStore(self._resources, create_session_record),
                episode_store,
                long_term_memory_store,
            ],
            tracker=OperationTracker(
                await self._resources.get_metrics_factory("prometheus"),
                prefix=_PARTITION_NAMESPACE,
            ),
        )

    async def _live_partition(self, session_key: str) -> Registration | None:
        """
        Resolve a session key to its live partition, or None when none holds the key.

        A pending partition is not created yet, and a key the registry refuses
        as a partition name names no partition.
        """
        partition_manager = await self._partition_manager()
        try:
            return await partition_manager.resolve_partition(session_key)
        except (PartitionPendingError, ValueError):
            return None

    async def _get_or_create_partition(self, session_key: str) -> Registration:
        """
        Resolve a session key to its live partition, creating the partition when none holds the key.

        Read-then-create, retried a bounded number of times: a live partition
        is returned; a pending one is another creator's, which the call waits
        for; no partition means create, with the deployment's default
        settings, then read again. The call retries when the creation finds
        the key taken, or the partition is deleted while it is created or
        before the read after it.

        Raises:
            ValueError:
                If the registry refuses the session key as a partition name.
            PartitionPendingError:
                If the partition was another creator's and still pending at
                the last attempt.
            PartitionCreateAttemptsExhaustedError:
                If the attempts ran out with the last one losing the key to
                another creator or deleter.

        """
        partition_manager = await self._partition_manager()
        pending_error: PartitionPendingError | None = None
        for attempt in range(_MAX_GET_OR_CREATE_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_GET_OR_CREATE_RETRY_DELAY_SECONDS)
            try:
                registration = await partition_manager.resolve_partition(session_key)
                if registration is None:
                    await self._create_partition(session_key)
                    registration = await partition_manager.resolve_partition(
                        session_key
                    )
            except PartitionPendingError as err:
                pending_error = err
                continue
            except (PartitionAlreadyExistsError, PartitionDeletedError):
                pending_error = None
                continue
            pending_error = None
            if registration is not None:
                return registration
        if pending_error is not None:
            raise pending_error
        raise PartitionCreateAttemptsExhaustedError(
            f"Getting or creating the partition of session {session_key!r} made "
            f"no progress after {_MAX_GET_OR_CREATE_ATTEMPTS} attempts"
        )

    @staticmethod
    async def _require_live_partition(
        registration: Registration, session_key: str
    ) -> None:
        """
        Raise unless the session's partition is still the one resolved.

        Raises:
            SessionNotFoundError: If the partition was deleted since.

        """
        try:
            await registration.require_current()
        except PartitionDeletedError as err:
            raise SessionNotFoundError(session_key) from err

    async def _delete_session_semantic_memory(
        self,
        session: SessionData,
        semantic_storage: SemanticStorage,
    ) -> None:
        set_ids = SemanticSessionManager.project_deletion_set_ids(
            org_id=session.org_id,
            project_id=session.project_id,
        )
        if not set_ids:
            # An empty filter would delete every feature in the store.
            raise RuntimeError(
                f"Session {session.session_key} has no semantic set ids to delete."
            )
        await semantic_storage.delete_feature_set(
            filter_expr=with_has_set_ids(set_ids, None)
        )

    async def _semantic_deletion_storage(self) -> SemanticStorage | None:
        if not self._conf.semantic_memory.storage_configured():
            logger.debug("No semantic storage configured; skipping semantic cleanup.")
            return None
        return await self._resources.get_semantic_storage()

    async def start(self) -> None:
        """
        Start MemMachine background services.

        Returns:
            None.

        """
        if self._started:
            return
        # Starts the registry of MemMachine's partition namespace and its
        # purge rounds, so this process finishes deletions from its start.
        await self._partition_manager()
        self._started = True

        if self._conf.semantic_memory.enabled:
            semantic_service = await self._resources.get_semantic_service()
            await semantic_service.start()
        else:
            logger.info(
                "Semantic memory is disabled; skipping semantic service startup."
            )

    async def stop(self) -> None:
        """
        Stop MemMachine background services and release resources.

        Returns:
            None.

        """
        if not self._started:
            return
        self._started = False
        if self._conf.semantic_memory.enabled:
            semantic_service = await self._resources.get_semantic_service()
            await semantic_service.stop()

        await self._resources.close()

    @property
    def resource_manager(self) -> ResourceManagerImpl:
        """Return the resource manager."""
        return self._resources

    def _with_default_episodic_memory_conf(
        self,
        *,
        user_conf: EpisodicMemoryConfPartial | None = None,
        session_key: str,
    ) -> EpisodicMemoryConf:
        """
        Merge per-session episodic config with defaults.

        Args:
            user_conf: Optional episodic configuration overrides.
            session_key: Session key to associate with the config.

        Returns:
            The resolved episodic memory configuration.

        Raises:
            ConfigurationError: If the merged configuration is invalid, or
                has event memory on a deployment that configures none.

        """
        # Get default prompts from config, with fallbacks
        try:
            if not self._conf.episodic_memory.enabled:
                raise ResourceNotReadyError("Episodic memory is disabled")
            if user_conf is None:
                user_conf = EpisodicMemoryConfPartial()
            user_conf.session_key = session_key
            episodic_conf = user_conf.merge(self._conf.episodic_memory)
            if episodic_conf.long_term_memory is not None:
                if isinstance(episodic_conf.long_term_memory, EventLongTermMemoryConf):
                    deployment_event_memory_ids(self._conf)
                elif episodic_conf.long_term_memory.embedder is not None:
                    self._conf.check_embedder(episodic_conf.long_term_memory.embedder)
                if episodic_conf.long_term_memory.reranker is not None:
                    self._conf.check_reranker(episodic_conf.long_term_memory.reranker)
        except ValidationError as e:
            logger.exception(
                "Failed to merge configuration: %s, %s",
                str(user_conf),
                str(self._conf.episodic_memory),
            )
            raise ConfigurationError("Failed to merge configuration") from e
        return episodic_conf

    @staticmethod
    def _disabled_episodic_conf(session_key: str) -> EpisodicMemoryConf:
        """Return a minimal episodic config with all memory disabled."""
        return EpisodicMemoryConf(
            session_key=session_key,
            long_term_memory=None,
            short_term_memory=None,
            long_term_memory_enabled=False,
            short_term_memory_enabled=False,
            enabled=False,
        )

    async def create_session(
        self,
        session_key: str,
        *,
        description: str = "",
        user_conf: EpisodicMemoryConfPartial | None = None,
    ) -> SessionDataManager.SessionInfo:
        """
        Create and persist a new session.

        The session's partition is created with its record, written under
        the partition's key. A session whose partition exists is answered
        when the stored record's data matches.

        Args:
            session_key: Unique identifier for the session.
            description: Optional human-readable session description.
            user_conf: Optional episodic-memory configuration overrides.

        Returns:
            The created session info.

        Raises:
            InvalidArgumentError:
                If the registry refuses the session key as a partition name.
            PartitionPendingError:
                If another creation holds the session key and has not
                completed. A caller that takes it for abandoned deletes the
                session and creates it again.
            SessionAlreadyExistsError:
                If the stored session data conflicts with the request.
            SessionNotFoundError:
                If the session was deleted while it was created.

        """
        # Resolved before the partition is created, so a configuration error
        # leaves no partition behind.
        self._episodic_memory_conf(user_conf=user_conf, session_key=session_key)

        partition_manager = await self._partition_manager()
        try:
            # A key that is taken is live or pending; resolving it tells which.
            with contextlib.suppress(PartitionAlreadyExistsError):
                await self._create_partition(
                    session_key, description=description, user_conf=user_conf
                )
            registration = await partition_manager.resolve_partition(session_key)
        except ValueError as err:
            raise InvalidArgumentError(str(err)) from err
        if registration is None:
            raise SessionNotFoundError(session_key)
        partition_key = str(registration.partition_uuid)

        try:
            # A partition found live is checked against the request.
            await self._create_or_validate_session_record(
                partition_key, description=description, user_conf=user_conf
            )
        except SessionAlreadyExistsError as err:
            raise SessionAlreadyExistsError(session_key) from err
        await self._require_live_partition(registration, session_key)
        session_data_manager = await self._resources.get_session_data_manager()
        ret = await session_data_manager.get_session_info(
            partition_key, status=SessionDataManager.SessionStatus.Active
        )
        if ret is None:
            raise RuntimeError(f"Failed to create session {session_key}")
        return ret

    async def _create_partition(
        self,
        session_key: str,
        *,
        description: str = "",
        user_conf: EpisodicMemoryConfPartial | None = None,
    ) -> None:
        """
        Create a session's partition, with the session's record.

        The record is written while the partition is pending, so every live
        partition has its record.

        Raises:
            ValueError:
                If the registry refuses the session key as a partition name.
            PartitionAlreadyExistsError:
                If the session key is taken, by a live or a pending
                partition.

        """

        async def create_session_record(partition_key: str) -> None:
            await self._create_or_validate_session_record(
                partition_key, description=description, user_conf=user_conf
            )

        partition_manager = await self._partition_manager(create_session_record)
        await partition_manager.create_partition(session_key)

    async def _create_or_validate_session_record(
        self,
        partition_key: str,
        *,
        description: str,
        user_conf: EpisodicMemoryConfPartial | None,
    ) -> None:
        """
        Write a session's record under its partition's key, or accept the stored record when its data matches.

        Raises:
            SessionAlreadyExistsError:
                If the stored record's data differs.

        """
        session_data_manager = await self._resources.get_session_data_manager()
        await session_data_manager.create_or_validate_session(
            session_key=partition_key,
            configuration={},
            param=self._episodic_memory_conf(
                user_conf=user_conf, session_key=partition_key
            ),
            description=description,
            metadata={},
        )

    def _episodic_memory_conf(
        self,
        *,
        user_conf: EpisodicMemoryConfPartial | None,
        session_key: str,
    ) -> EpisodicMemoryConf:
        """Resolve a session's episodic memory configuration, with all memory disabled on a deployment that disables episodic memory."""
        if not self._conf.episodic_memory.enabled:
            return self._disabled_episodic_conf(session_key)
        return self._with_default_episodic_memory_conf(
            user_conf=user_conf,
            session_key=session_key,
        )

    async def get_session(
        self, session_key: str
    ) -> SessionDataManager.SessionInfo | None:
        """
        Fetch stored session info.

        Args:
            session_key: Unique identifier for the session.

        Returns:
            The session info, or `None` if it does not exist.

        """
        registration = await self._live_partition(session_key)
        if registration is None:
            return None
        session_data_manager = await self._resources.get_session_data_manager()
        return await session_data_manager.get_session_info(
            str(registration.partition_uuid),
            status=SessionDataManager.SessionStatus.Active,
        )

    async def delete_session(self, session_data: SessionData) -> None:
        """
        Delete all data associated with a session.

        The session's partition is deleted in one transaction: no request
        resolves the session once this returns, its key is free, and the
        partition's purge rounds delete its data once the retention has
        passed. Semantic sets are named by organization and project, so
        their features are deleted first, while the key still names this
        session.

        Args:
            session_data: Session context providing the session key.

        Returns:
            None.

        Raises:
            SessionNotFoundError: If the session does not exist.

        """
        partition_manager = await self._partition_manager()
        try:
            registration = await partition_manager.resolve_partition(
                session_data.session_key
            )
        except ValueError as err:
            raise SessionNotFoundError(session_data.session_key) from err
        except PartitionPendingError:
            # Deleted like a live partition. No request was served under it,
            # so semantic memory holds nothing of it.
            pass
        else:
            if registration is None:
                raise SessionNotFoundError(session_data.session_key)
            semantic_storage = await self._semantic_deletion_storage()
            if semantic_storage is not None:
                await self._delete_session_semantic_memory(
                    session_data, semantic_storage
                )
        await partition_manager.delete_partition(session_data.session_key)

    async def search_sessions(self, prefix: str = "") -> list[ListedPartition]:
        """
        List the sessions, pending and live, whose keys start with a prefix.

        A session is listed from the reservation of its key to its deletion,
        with whether it is live. The registry's listing is read a page at a
        time, so a session created or deleted while the pages are read may or
        may not be listed.

        Args:
            prefix: What a listed session's key starts with.

        Returns:
            The sessions' registrations as the listing found them, in the
            byte order of their keys.

        Raises:
            ValueError: If the registry refuses the prefix.

        """
        partition_manager = await self._partition_manager()
        limit = partition_manager.max_list_limit
        sessions: list[ListedPartition] = []
        after: str | None = None
        while True:
            page = await partition_manager.list_partitions(
                prefix=prefix, after=after, limit=limit
            )
            sessions.extend(page)
            if len(page) < limit:
                return sessions
            after = page[-1].partition_name

    async def update_session_episodic_config(
        self,
        session_key: str,
        enabled: bool | None = None,
        long_term_memory_enabled: bool | None = None,
        short_term_memory_enabled: bool | None = None,
    ) -> None:
        """
        Update episodic memory configuration flags for a session.

        This allows enabling/disabling episodic memory, long-term memory,
        and short-term memory independently for a specific session.

        Args:
            session_key: The session key to update.
            enabled: Whether episodic memory is enabled overall.
            long_term_memory_enabled: Whether long-term memory is enabled.
            short_term_memory_enabled: Whether short-term memory is enabled.

        Raises:
            SessionNotFoundError: If the session does not exist.

        """
        registration = await self._live_partition(session_key)
        if registration is None:
            raise SessionNotFoundError(session_key)
        session_data_manager = await self._resources.get_session_data_manager()
        try:
            await session_data_manager.update_session_episodic_config(
                session_key=str(registration.partition_uuid),
                enabled=enabled,
                long_term_memory_enabled=long_term_memory_enabled,
                short_term_memory_enabled=short_term_memory_enabled,
            )
        except SessionNotFoundError as err:
            raise SessionNotFoundError(session_key) from err
        await self._require_live_partition(registration, session_key)

    @staticmethod
    def _merge_filter_exprs(
        left: FilterExpr | None,
        right: FilterExpr | None,
    ) -> FilterExpr | None:
        """
        Combine two filter expressions with logical AND.

        Args:
            left: Left-hand filter expression.
            right: Right-hand filter expression.

        Returns:
            The combined filter, or the non-`None` operand.

        """
        if left is None:
            return right
        if right is None:
            return left
        return FilterAnd(left=left, right=right)

    def _should_dispatch_to_semantic_memory(
        self, target_memories: list[MemoryType]
    ) -> bool:
        """
        Report whether a request should reach semantic memory.

        Requests default `target_memories` to every memory type, so the type
        list alone says nothing about whether semantic memory is configured.
        Dispatching on it while semantic memory is disabled reaches a backend
        whose required fields are legally unset and raises
        `ResourceNotReadyError`; skip it instead, as the startup and shutdown
        paths already do.

        Args:
            target_memories: Memory types this request asked for.

        Returns:
            `True` when semantic memory is both requested and enabled.

        """
        return (
            MemoryType.Semantic in target_memories
            and self._conf.semantic_memory.enabled
        )

    async def add_episodes(
        self,
        session_data: InstanceOf[SessionData],
        episode_entries: list[EpisodeEntry],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
    ) -> list[UUID]:
        """
        Append episodes to storage and selected memory backends.

        Args:
            session_data: Session context used to route writes.
            episode_entries: Episode messages/entries to add.
            target_memories: Memory types to update (episodic, semantic).

        Returns:
            IDs of the created episodes.

        Raises:
            ValueError:
                If an entry's metadata holds a key that is not a legal user
                property key (`validate_user_property_key`), before any store
                is written.

        """
        episode_storage = await self._resources.get_episode_storage()
        if not episode_entries:
            return []
        for entry in episode_entries:
            for key in entry.metadata or {}:
                validate_user_property_key(key)
        registration = await self._get_or_create_partition(session_data.session_key)
        partition_key = str(registration.partition_uuid)
        created_at = datetime.now(UTC)
        # UUID order preserves input order within a batch when timestamps tie.
        batch_ids = sorted(uuid4() for _ in episode_entries)
        episode_entries = [
            entry.model_copy(
                update={
                    "created_at": entry.created_at or created_at,
                    "uid": batch_ids[index],
                }
            )
            for index, entry in enumerate(episode_entries)
        ]
        episodes = [
            Episode(
                session_key=partition_key,
                metadata=entry.metadata or None,
                **entry.model_dump(exclude_none=True, exclude={"metadata"}),
            )
            for entry in episode_entries
        ]
        episode_ids = [e.uid for e in episodes]

        episodic_memory_manager = None
        semantic_session_manager = None

        if MemoryType.Episodic in target_memories:
            episodic_memory_manager = (
                await self._resources.get_episodic_memory_manager()
            )

            async def add_to_episodic_memory() -> None:
                async with episodic_memory_manager.open_or_create_episodic_memory(
                    session_key=partition_key,
                    description="",
                    episodic_memory_config=self._with_default_episodic_memory_conf(
                        session_key=partition_key
                    ),
                    metadata={},
                ) as episodic_session:
                    await episodic_session.add_memory_episodes(episodes)

        if self._should_dispatch_to_semantic_memory(target_memories):
            semantic_session_manager = (
                await self._resources.get_semantic_session_manager()
            )

        tasks: list[Coroutine[Any, Any, object]] = [
            episode_storage.add_episodes(
                partition_key,
                episode_entries,
            )
        ]
        task_names = ["episode storage"]

        if episodic_memory_manager is not None:
            tasks.append(add_to_episodic_memory())
            task_names.append("episodic memory")

        if semantic_session_manager is not None:
            tasks.append(
                semantic_session_manager.add_message(
                    episodes=episodes,
                    session_data=session_data,
                )
            )
            task_names.append("semantic memory")

        # TODO(#1738): Track write intent so partial failures across these
        # parallel backends can be detected and repaired.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        _raise_add_episode_errors(task_names, results, episode_ids)
        # A session deleted while the episodes were written holds none of
        # them: they are under the deleted partition's key until its purge.
        await self._require_live_partition(registration, session_data.session_key)

        return episode_ids

    class SearchResponse(BaseModel):
        """Aggregated search results across memory types."""

        episodic_memory: EpisodicMemory.QueryResponse | None = None
        semantic_memory: list[SemanticFeature] | None = None

    async def _search_episodic_memory(
        self,
        *,
        session_data: InstanceOf[SessionData],
        query: str,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        search_filter: FilterExpr | None = None,
        retrieval_agent: AgentToolBase | None = None,
    ) -> EpisodicMemory.QueryResponse | None:
        """
        Query episodic memory for relevant episodes.

        Args:
            session_data: Session context used to select the memory.
            query: Query string.
            limit: Optional maximum number of results.
            expand_context: Number of surrounding episodes to return with each match.
            search_filter: Optional property filter for narrowing results.
            score_threshold: Optional minimum score threshold for results.
            retrieval_agent: Optional top-level retrieval agent for long-term search.

        Returns:
            Episodic memory query response, if episodic memory is enabled.

        """
        episodic_memory_manager = await self._resources.get_episodic_memory_manager()
        registration = await self._get_or_create_partition(session_data.session_key)
        partition_key = str(registration.partition_uuid)

        async with episodic_memory_manager.open_or_create_episodic_memory(
            session_key=partition_key,
            description="",
            episodic_memory_config=self._with_default_episodic_memory_conf(
                session_key=partition_key
            ),
            metadata={},
        ) as episodic_session:
            if retrieval_agent is None or episodic_session.long_term_memory is None:
                response = await episodic_session.query_memory(
                    query=query,
                    limit=limit,
                    expand_context=expand_context,
                    score_threshold=score_threshold,
                    property_filter=search_filter,
                )
            else:
                response = await self._query_episodic_with_retrieval_agent(
                    episodic_session=episodic_session,
                    retrieval_agent=retrieval_agent,
                    query=query,
                    limit=limit,
                    expand_context=expand_context,
                    score_threshold=score_threshold,
                    search_filter=search_filter,
                )

        return response

    async def _query_episodic_with_retrieval_agent(
        self,
        *,
        episodic_session: EpisodicMemory,
        retrieval_agent: AgentToolBase,
        query: str,
        limit: int | None,
        expand_context: int,
        score_threshold: float,
        search_filter: FilterExpr | None,
    ) -> EpisodicMemory.QueryResponse | None:
        """Build episodic query response using retrieval-agent long-term search."""
        if episodic_session.long_term_memory is None:
            return await episodic_session.query_memory(
                query=query,
                limit=limit,
                expand_context=expand_context,
                score_threshold=score_threshold,
                property_filter=search_filter,
                mode=EpisodicMemory.QueryMode.SHORT_TERM_ONLY,
            )

        search_limit = limit if limit is not None else 20
        normalized_long_episodes = await self._run_retrieval_agent_long_term_search(
            episodic_session=episodic_session,
            retrieval_agent=retrieval_agent,
            query=query,
            limit=search_limit,
            expand_context=expand_context,
            score_threshold=score_threshold,
            search_filter=search_filter,
        )

        short_response = await self._query_short_term_response_for_agent(
            episodic_session=episodic_session,
            query=query,
            limit=search_limit,
            expand_context=expand_context,
            score_threshold=score_threshold,
            search_filter=search_filter,
        )

        unique_scored_long_episodes = (
            MemMachine._dedupe_and_score_agent_long_term_episodes(
                normalized_long_episodes=normalized_long_episodes,
                short_response=short_response,
                score_threshold=score_threshold,
            )
        )

        return EpisodicMemory.QueryResponse(
            short_term_memory=short_response,
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[
                    EpisodeResponse(score=score, **episode.model_dump())
                    for score, episode in unique_scored_long_episodes
                ],
            ),
        )

    async def _run_retrieval_agent_long_term_search(
        self,
        *,
        episodic_session: EpisodicMemory,
        retrieval_agent: AgentToolBase,
        query: str,
        limit: int,
        expand_context: int,
        score_threshold: float,
        search_filter: FilterExpr | None,
    ) -> list[Episode]:
        if episodic_session.long_term_memory is None:
            return []

        long_episodes, _ = await retrieval_agent.do_query(
            QueryPolicy(
                token_cost=10,
                time_cost=10,
                accuracy_score=10,
                confidence_score=10,
                max_attempts=3,
                max_return_len=10000,
            ),
            QueryParam(
                query=query,
                limit=limit,
                expand_context=expand_context,
                score_threshold=score_threshold,
                property_filter=search_filter,
                memory=episodic_session,
            ),
        )

        return long_episodes

    async def _query_short_term_response_for_agent(
        self,
        *,
        episodic_session: EpisodicMemory,
        query: str,
        limit: int,
        expand_context: int,
        score_threshold: float,
        search_filter: FilterExpr | None,
    ) -> EpisodicMemory.QueryResponse.ShortTermMemoryResponse:
        if episodic_session.short_term_memory is None:
            return EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[],
                episode_summary=[""],
            )

        short_term_result = await episodic_session.query_memory(
            query=query,
            limit=limit,
            expand_context=expand_context,
            score_threshold=score_threshold,
            property_filter=search_filter,
            mode=EpisodicMemory.QueryMode.SHORT_TERM_ONLY,
        )
        if short_term_result is None:
            return EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[],
                episode_summary=[""],
            )
        return short_term_result.short_term_memory

    @staticmethod
    def _dedupe_and_score_agent_long_term_episodes(
        *,
        normalized_long_episodes: list[Episode],
        short_response: EpisodicMemory.QueryResponse.ShortTermMemoryResponse,
        score_threshold: float,
    ) -> list[tuple[float, Episode]]:
        scored_long_episodes = [
            (1.0, episode)
            for episode in normalized_long_episodes
            if score_threshold <= 1.0
        ]
        episode_uid_set = {episode.uid for episode in short_response.episodes}
        unique_scored_long_episodes: list[tuple[float, Episode]] = []
        for score, episode in scored_long_episodes:
            if episode.uid in episode_uid_set:
                continue
            episode_uid_set.add(episode.uid)
            unique_scored_long_episodes.append((score, episode))
        return unique_scored_long_episodes

    async def query_search(
        self,
        session_data: InstanceOf[SessionData],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
        set_metadata: Mapping[str, JsonValue] | None = None,
        query: str,
        limit: int
        | None = None,  # TODO: Define if limit is per memory or is global limit
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        search_filter: str | None = None,
        agent_mode: bool = False,
    ) -> SearchResponse:
        """
        Search across enabled memory types using a query string.

        Args:
            session_data: Session context used to route the search.
            target_memories: Which memory types to query.
            set_metadata: Optional metadata tags used to select semantic sets.
            query: Query string.
            limit: Optional maximum number of results per memory.
            expand_context: Number of surrounding episodes to return with each match.
            search_filter: Optional filter string applied to each memory query.
            score_threshold: Optional minimum score threshold for results.
            agent_mode: Whether to enable top-level retrieval-agent orchestration.

        Returns:
            Aggregated search results across memory types.

        """
        episodic_task: Task | None = None
        semantic_task: Task | None = None

        property_filter = parse_filter(search_filter) if search_filter else None
        if MemoryType.Episodic in target_memories:
            retrieval_agent = await self._get_retrieval_agent() if agent_mode else None
            episodic_task = asyncio.create_task(
                self._search_episodic_memory(
                    session_data=session_data,
                    query=query,
                    limit=limit,
                    expand_context=expand_context,
                    score_threshold=score_threshold,
                    search_filter=property_filter,
                    retrieval_agent=retrieval_agent,
                )
            )

        if self._should_dispatch_to_semantic_memory(target_memories):
            semantic_session = await self._resources.get_semantic_session_manager()

            async def _collect_semantic_results() -> list[SemanticFeature]:
                return [
                    feature
                    async for feature in semantic_session.search(
                        message=query,
                        session_data=session_data,
                        set_metadata=set_metadata,
                        limit=limit,
                        search_filter=property_filter,
                    )
                ]

            semantic_task = asyncio.create_task(_collect_semantic_results())

        return MemMachine.SearchResponse(
            episodic_memory=await episodic_task if episodic_task else None,
            semantic_memory=await semantic_task if semantic_task else None,
        )

    class ListResults(BaseModel):
        """Result payload for list-style memory queries."""

        episodic_memory: list[Episode] | None = None
        semantic_memory: list[SemanticFeature] | None = None

    async def list_search(
        self,
        session_data: InstanceOf[SessionData],
        *,
        target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
        set_metadata: Mapping[str, JsonValue] | None = None,
        search_filter: str | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
    ) -> ListResults:
        """
        List episodes/features matching a filter with pagination.

        Args:
            session_data: Session context used to route the query.
            target_memories: Which memory types to query.
            set_metadata: Optional metadata tags used to select semantic sets.
            search_filter: Optional filter string applied to the query.
            page_size: Optional page size.
            page_num: Optional 1-based page number.

        Returns:
            Aggregated list results across memory types.

        """
        search_filter_expr = parse_filter(search_filter) if search_filter else None

        episodic_task: Task | None = None
        semantic_task: Task | None = None
        episodic_result: list[Episode] | None = None

        if MemoryType.Episodic in target_memories:
            episode_storage = await self._resources.get_episode_storage()
            registration = await self._live_partition(session_data.session_key)
            if registration is None:
                # No session holds the key, so it has no episodes.
                episodic_result = []
            else:
                session_filter = FilterComparison(
                    field="session_key",
                    op="=",
                    value=str(registration.partition_uuid),
                )
                combined_filter = self._merge_filter_exprs(
                    session_filter,
                    search_filter_expr,
                )
                episodic_task = asyncio.create_task(
                    episode_storage.get_episode_messages(
                        page_size=page_size,
                        page_num=page_num,
                        filter_expr=combined_filter,
                    )
                )

        if self._should_dispatch_to_semantic_memory(target_memories):
            semantic_session = await self._resources.get_semantic_session_manager()

            async def _collect_semantic_results() -> list[SemanticFeature]:
                return [
                    feature
                    async for feature in semantic_session.get_set_features(
                        session_data=session_data,
                        set_metadata=set_metadata,
                        search_filter=search_filter_expr,
                        page_size=page_size,
                        page_num=page_num,
                    )
                ]

            semantic_task = asyncio.create_task(_collect_semantic_results())

        if episodic_task is not None:
            episodic_result = await episodic_task
        semantic_result = await semantic_task if semantic_task else None

        return MemMachine.ListResults(
            episodic_memory=episodic_result,
            semantic_memory=semantic_result,
        )

    async def episodes_count(
        self,
        session_data: InstanceOf[SessionData],
        *,
        search_filter: str | None = None,
    ) -> int:
        """
        Count episodes in a session matching a filter.

        Args:
            session_data: Session context providing the session key.
            search_filter: Optional filter string narrowing which episodes count.

        Returns:
            Number of episodes matching the filter.

        """
        episode_storage = await self._resources.get_episode_storage()
        registration = await self._live_partition(session_data.session_key)
        if registration is None:
            return 0

        session_filter = FilterComparison(
            field="session_key",
            op="=",
            value=str(registration.partition_uuid),
        )

        search_filter_expr = parse_filter(search_filter) if search_filter else None
        combined_filter = self._merge_filter_exprs(session_filter, search_filter_expr)

        return await episode_storage.get_episode_messages_count(
            filter_expr=combined_filter,
        )

    async def delete_episodes(
        self,
        episode_ids: list[UUID],
        session_data: InstanceOf[SessionData],
    ) -> None:
        """
        Delete episodes from storage and memory backends.

        Args:
            episode_ids: IDs of episodes to delete.
            session_data: Session context used to verify episode ownership and
                delete from episodic memory.

        Returns:
            None.

        """
        if session_data is None:
            raise ValueError("session_data is required")

        episode_storage = await self._resources.get_episode_storage()
        tasks: list[Coroutine[Any, Any, Any]] = []

        registration = await self._live_partition(session_data.session_key)
        partition_key = (
            str(registration.partition_uuid) if registration is not None else None
        )
        episodes = await episode_storage.get_episodes(episode_ids)
        for episode in episodes:
            if episode.session_key != partition_key:
                raise ResourceNotFoundError(
                    f"Episode '{episode.uid}' was not found in this session."
                )
        if partition_key is None:
            return

        semantic_storage = await self._semantic_deletion_storage()
        if semantic_storage is not None:
            # First, so a failure leaves the episodes in place for a retry.
            await semantic_storage.delete_history(episode_ids)

        episodic_memory_manager = await self._resources.get_episodic_memory_manager()

        async def delete_from_episodic_memory() -> None:
            try:
                async with episodic_memory_manager.open_episodic_memory(
                    partition_key
                ) as episodic_session:
                    await episodic_session.delete_episodes(episode_ids)
            except SessionNotFoundError:
                logger.debug(
                    "No episodic session for %s during idempotent delete",
                    session_data.session_key,
                )

        tasks.append(delete_from_episodic_memory())

        tasks.append(episode_storage.delete_episodes(episode_ids))
        await asyncio.gather(*tasks)

    async def delete_features(
        self,
        feature_ids: list[FeatureIdT],
    ) -> None:
        """
        Delete semantic features by ID.

        Args:
            feature_ids: Feature identifiers to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()
        await semantic_session.delete_features(feature_ids)

    async def add_feature(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        tag: str,
        feature: str,
        value: str,
        feature_metadata: dict[str, JsonValue] | None = None,
        citations: list[UUID] | None = None,
    ) -> FeatureIdT:
        """
        Add a semantic feature to the current semantic set.

        Args:
            set_id: Set ID to add the feature to.
            feature_metadata: Optional metadata to store alongside the feature.
            category_name: Category name to attach the feature to.
            feature: Feature name/key.
            value: Feature value.
            tag: Tag name to associate with the feature.
            citations: Optional episode IDs supporting this feature.

        Returns:
            The created feature ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_feature(
            set_id=set_id,
            feature_metadata=feature_metadata,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            citations=citations,
        )

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        """
        Fetch a semantic feature by ID.

        Args:
            feature_id: Feature identifier.
            load_citations: Whether to load referenced episode IDs.

        Returns:
            The feature, or `None` if it does not exist.

        """
        semantic_session = await self._resources.get_semantic_session_manager()
        return await semantic_session.get_feature(
            feature_id=feature_id,
            load_citations=load_citations,
        )

    async def update_feature(
        self,
        *,
        feature_id: FeatureIdT,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """
        Update an existing semantic feature.

        Only fields that are not `None` are updated.

        Args:
            feature_id: Feature identifier.
            category_name: New category name.
            feature: New feature name/key.
            value: New feature value.
            tag: New tag name.
            metadata: Replacement metadata payload.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.update_feature(
            feature_id=feature_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
        )

    async def create_semantic_set_type(
        self,
        *,
        session_data: SessionData,
        is_org_level: bool,
        metadata_tags: list[str],
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Create a new semantic set type.

        Args:
            session_data: Context used to locate the project/org scope.
            set_metadata: Optional metadata tags used to select semantic sets.
            is_org_level: Whether the set type is org-scoped.
            metadata_tags: Ordered list of metadata tag keys defining the set.
            name: Optional name for the set type.
            description: Optional description for the set type.

        Returns:
            The created set type ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.create_set_type(
            session_data=session_data,
            is_org_level=is_org_level,
            metadata_tags=metadata_tags,
            name=name,
            description=description,
        )

    async def semantic_list_set_ids(
        self,
        *,
        session_data: SessionData,
        set_metadata: Mapping[str, JsonValue] | None = None,
    ) -> Iterable[SemanticSessionManager.Set]:
        """
        List set IDs for the given session data.

        Args:
            session_data: Context used to locate the project/org scope.
            set_metadata: Metadata tags used to select semantic sets.

        Returns:
            List of set IDs matching the session data criteria.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.list_set_ids(
            session_data=session_data,
            set_metadata=set_metadata,
        )

    async def semantic_get_set_id(
        self,
        *,
        session_data: SessionData,
        is_org_level: bool,
        metadata_tags: list[str],
        set_metadata: Mapping[str, JsonValue] | None = None,
    ) -> SetIdT:
        """
        Retrieve the set ID for a given set type and metadata tags.

        Args:
            session_data: Context used to locate the project/org scope.
            is_org_level: Whether the set is org-scoped (vs project-scoped).
            metadata_tags: Ordered list of metadata tag keys defining the set.
            set_metadata: Optional metadata tags used to select semantic sets.

        Returns:
            The set ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.get_set_id(
            session_data=session_data,
            is_org_level=is_org_level,
            set_metadata_keys=metadata_tags,
            set_metadata=set_metadata,
        )

    async def delete_semantic_set_type(self, set_type_id: str) -> None:
        """
        Delete a semantic set type by ID.

        Args:
            set_type_id: Set type identifier to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.delete_set_type(
            set_type_id=set_type_id,
        )

    async def list_semantic_set_type(
        self,
        *,
        session_data: SessionData,
    ) -> Iterable[SetTypeEntry]:
        """
        List semantic set types available for the context.

        Args:
            session_data: Context used to locate the project/org scope.

        Returns:
            Available semantic set types for the context.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.list_set_types(
            session_data=session_data,
        )

    async def configure_semantic_set(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        """
        Configure the semantic set used for feature extraction/storage.

        Args:
            set_id: The set ID to configure.
            embedder_name: Optional embedder override for this set.
            llm_name: Optional LLM override for this set.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.configure_set(
            set_id=set_id,
            embedder_name=embedder_name,
            llm_name=llm_name,
        )

    async def semantic_get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> SemanticConfigStorage.Category | None:
        """
        Get a semantic category by its ID.

        Args:
            category_id: The ID of the category to retrieve.

        Returns:
            The category if found, otherwise None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.get_category(category_id=category_id)

    async def semantic_add_category(
        self,
        *,
        set_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        """
        Create a new semantic category within the configured set.

        Args:
            set_id: Set ID to create the category in.
            category_name: Name of the category.
            prompt: Prompt to go with the category.
            description: Human-readable category description.

        Returns:
            The created category ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_new_category(
            set_id=set_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

    async def semantic_add_category_template(
        self,
        *,
        set_type_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        """Create a new semantic category on a set type.

        Categories created here are inherited by set_ids mapped to the set type.

        Args:
            set_type_id: The set type identifier.
            category_name: Name of the category.
            prompt: Prompt to go with the category.
            description: Human-readable category description.

        Returns:
            The created category ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_category_template(
            set_type_id=set_type_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

    async def semantic_list_category_templates(
        self,
        *,
        set_type_id: str,
    ) -> Iterable[SemanticCategory]:
        """List semantic categories defined on a set type."""
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.list_category_templates(set_type_id=set_type_id)

    async def semantic_disable_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        """
        Disable a default semantic category for the configured set.

        Args:
            set_id: Set ID to disable the category in.
            category_name: Name of the default category to disable.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.disable_category(
            set_id=set_id,
            category_name=category_name,
        )

    async def semantic_get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> Iterable[SetIdT]:
        """
        Get the set_ids associated with a semantic category.

        Args:
            category_id: Category identifier to query.

        Returns:
            List of set_ids associated with the category.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.get_category_set_ids(
            category_id=category_id,
        )

    async def semantic_delete_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> None:
        """
        Delete a semantic category and all its tags.

        Args:
            category_id: Category identifier to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.delete_category(
            category_id=category_id,
        )

    async def semantic_add_tag_to_category(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        tag_description: str,
    ) -> TagIdT:
        """
        Add a tag to an existing semantic category.

        Args:
            category_id: Category identifier.
            tag_name: Name of the new tag.
            tag_description: Human-readable description of the tag.

        Returns:
            The created tag ID.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.add_tag(
            category_id=category_id,
            tag_name=tag_name,
            tag_description=tag_description,
        )

    async def semantic_delete_tag(self, *, tag_id: TagIdT) -> None:
        """
        Delete a semantic tag by ID.

        Args:
            tag_id: Tag identifier to delete.

        Returns:
            None.

        """
        semantic_session = await self._resources.get_semantic_session_manager()

        return await semantic_session.delete_tag(tag_id=tag_id)

    async def delete_all(self) -> None:
        """
        Delete all MemMachine data from backing stores.

        This is a destructive operation intended for testing/administration.

        Returns:
            None.

        """
        logger.info("Deleting all data from MemMachine")

        # TODO: Add episodic memory deletion

        semantic_resource_manager = await self._resources.get_semantic_manager()
        semantic_storage = await semantic_resource_manager.get_semantic_storage()
        semantic_config_storage = (
            await semantic_resource_manager.get_semantic_config_storage()
        )

        episodic_store = await self._resources.get_episode_storage()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(semantic_storage.delete_all())
            tg.create_task(semantic_config_storage.delete_all())
            tg.create_task(episodic_store.delete_all())


async def prepare_memmachine_partition_registry(
    resource_manager: ResourceManagerImpl,
) -> None:
    """
    Prepare the registry of MemMachine's partition namespace.

    Run it with no server using the session manager's database, before one
    starts.
    """
    session_manager = resource_manager.config.session_manager
    partition_registry = await resource_manager.build_partition_registry(
        session_manager.database,
        partition_namespace=_PARTITION_NAMESPACE,
        tombstone_retention_seconds=session_manager.tombstone_retention_seconds,
    )
    await partition_registry.prepare()
