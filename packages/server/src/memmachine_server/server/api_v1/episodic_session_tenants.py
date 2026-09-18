"""The default resolution of a v1 tenant name: an episodic-memory session.

A tenant's data lives in the event memory of one episodic-memory session
whose key is the tenant's name under a prefix of v1's own, so v1 tenants
and the v2 API's projects address disjoint sets of sessions.
"""

from typing import Final, override

from memmachine_server import MemMachine
from memmachine_server.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
)
from memmachine_server.common.errors import (
    ConfigurationError,
    ResourceNotReadyError,
    SessionAlreadyExistsError,
    SessionDeletedError,
    SessionNotFoundError,
)

from .tenant_event_memories import (
    ComponentNotEnabledError,
    EpisodicMemoryDefaults,
    TenantEventMemories,
    TenantEventMemory,
    TenantNotFoundError,
)

TENANT_SESSION_KEY_PREFIX: Final[str] = "v1/tenants/"
"""What the session key of every v1 tenant begins with.

A v2 session key is an organization id and a project id joined by a
slash, and neither id may contain one, so a v2 key holds exactly one
slash and a key under this prefix holds at least two. No tenant name,
whatever it contains, can therefore be read as a v2 project.
"""

TENANT_DESCRIPTION: Final[str] = "A MemMachine v1 tenant"
"""The description the session carries, naming what created it."""


def session_key_for_tenant(tenant: str) -> str:
    """The episodic-memory session key holding a v1 tenant's event memory."""
    return f"{TENANT_SESSION_KEY_PREFIX}{tenant}"


class EpisodicSessionTenantEventMemories(TenantEventMemories):
    """v1 tenants as episodic-memory sessions under v1's own key prefix.

    Each tenant is one session, configured with the event-memory backend
    and without short-term memory: v1 reads and writes events, and
    translates no episodes. The session's own partition and vector store
    collection are what a tenant owns, and deleting the tenant drops them.
    """

    def __init__(self, memmachine: MemMachine) -> None:
        """Resolve tenants against the sessions of this MemMachine."""
        self._memmachine = memmachine
        self._defaults = EpisodicMemoryDefaults()

    @override
    async def create(self, tenant: str) -> bool:
        created = True
        try:
            await self._memmachine.create_session(
                session_key_for_tenant(tenant),
                description=TENANT_DESCRIPTION,
                user_conf=_event_backend_configuration(),
            )
        except SessionAlreadyExistsError:
            created = False
        except (ConfigurationError, ResourceNotReadyError) as error:
            raise ComponentNotEnabledError(
                f"No event-backed episodic memory for tenant {tenant!r}: {error}"
            ) from error
        # Resolving materializes the segment store partition and the vector
        # store collection, so a deployment that cannot serve the tenant says
        # so here rather than at the tenant's first query.
        await self.resolve(tenant)
        return created

    @override
    async def exists(self, tenant: str) -> bool:
        session = await self._memmachine.get_session(session_key_for_tenant(tenant))
        return session is not None

    @override
    async def delete(self, tenant: str) -> None:
        session_key = session_key_for_tenant(tenant)
        resources = self._memmachine.resource_manager
        episodic_memory_manager = await resources.get_episodic_memory_manager()
        try:
            await episodic_memory_manager.delete_episodic_session(session_key)
        except SessionNotFoundError:
            return
        session_data_manager = await resources.get_session_data_manager()
        await session_data_manager.delete_session(session_key)

    @override
    async def resolve(self, tenant: str) -> TenantEventMemory:
        session_key = session_key_for_tenant(tenant)
        resources = self._memmachine.resource_manager
        episodic_memory_manager = await resources.get_episodic_memory_manager()
        try:
            # The manager counts references to the instance it caches; the
            # memory this yields outlives the block, and the stores it holds
            # belong to the resource manager, not to the instance.
            async with episodic_memory_manager.open_episodic_memory(
                session_key
            ) as episodic_memory:
                long_term_memory = episodic_memory.long_term_memory
        except (SessionNotFoundError, SessionDeletedError) as error:
            raise TenantNotFoundError(tenant) from error
        except (ConfigurationError, ResourceNotReadyError, ValueError) as error:
            # A session whose configuration names no memory this deployment
            # can build: the component is not enabled for this tenant.
            raise ComponentNotEnabledError(
                f"No episodic memory for tenant {tenant!r}: {error}"
            ) from error
        if long_term_memory is None:
            raise ComponentNotEnabledError(
                f"Long-term episodic memory is not enabled for tenant {tenant!r}"
            )
        event_memory = long_term_memory.event_memory
        if event_memory is None:
            raise ComponentNotEnabledError(
                f"Tenant {tenant!r} is not backed by the event memory"
            )
        return TenantEventMemory(
            event_memory=event_memory,
            reranker=long_term_memory.reranker,
            defaults=self._defaults,
        )


def _event_backend_configuration() -> EpisodicMemoryConfPartial:
    """What a v1 tenant's session overrides in the deployment's configuration.

    The backend is the event memory, whatever the deployment defaults to,
    and short-term memory is off: nothing in v1 summarizes or reads it.
    Every resource the backend binds -- the embedder, the vector store, the
    segment store, the reranker -- stays the deployment's.
    """
    return EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(backend="event"),
        long_term_memory_enabled=True,
        short_term_memory_enabled=False,
    )
