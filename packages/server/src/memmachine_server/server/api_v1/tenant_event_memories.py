"""The seam between the v1 routes and whatever holds a tenant's event memory.

The routes address a tenant by name and never see a lifecycle: they ask
`TenantEventMemories` for the tenant's `EventMemory`, and the
implementation decides what a name resolves to. Replacing the
implementation moves every v1 tenant to another substrate without
touching a route.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, Field

from memmachine_server.common.reranker import Reranker
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory


class TenantNotFoundError(Exception):
    """The named tenant was never created, or was deleted."""

    def __init__(self, tenant: str) -> None:
        """Name the tenant that was not found."""
        super().__init__(f"Tenant {tenant!r} does not exist")


class ComponentNotEnabledError(Exception):
    """The tenant has no event-backed episodic memory to serve the request."""


class EpisodicMemoryDefaults(BaseModel):
    """The values a query or an expansion takes when the request omits them.

    Attributes:
        query_limit (int):
            The number of hits a query answers with (default: 10).
        expand_context (int):
            The number of neighbors around each hit a query renders
            into the hit's text (default: 2).
        rerank_candidates (int):
            The number of hits the vector search fetches for the reranker
            to score (default: 40).
        expand_before (int):
            The number of segments an expansion walks backward
            (default: 5).
        expand_after (int):
            The number of segments an expansion walks forward
            (default: 5).
    """

    query_limit: int = Field(
        10, gt=0, description="The number of hits a query answers with"
    )
    expand_context: int = Field(
        2,
        ge=0,
        description="The number of neighbors around each hit a query renders",
    )
    rerank_candidates: int = Field(
        40,
        gt=0,
        description="The number of hits the vector search fetches for the reranker",
    )
    expand_before: int = Field(
        5, ge=0, description="The number of segments an expansion walks backward"
    )
    expand_after: int = Field(
        5, ge=0, description="The number of segments an expansion walks forward"
    )


@dataclass(frozen=True)
class TenantEventMemory:
    """One tenant's event memory and what serving a request against it needs."""

    event_memory: EventMemory
    reranker: Reranker | None
    defaults: EpisodicMemoryDefaults


class TenantEventMemories(ABC):
    """Resolves a v1 tenant name to the event memory holding that tenant's data.

    A tenant is an opaque name. Sessions carry no lifecycle here: they are
    the `session_id` on events, so the only lifecycle is the tenant's.
    """

    @abstractmethod
    async def create(self, tenant: str) -> bool:
        """Create the tenant, and return whether this call is what created it.

        Creating a tenant that exists changes nothing and returns False.

        Raises:
            ComponentNotEnabledError:
                If the deployment has no event-backed episodic memory to
                give the tenant.
        """
        raise NotImplementedError

    @abstractmethod
    async def exists(self, tenant: str) -> bool:
        """Whether the tenant was created and is not deleted."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, tenant: str) -> None:
        """Delete the tenant and the data it holds.

        Deleting a tenant that does not exist changes nothing.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve(self, tenant: str) -> TenantEventMemory:
        """The tenant's event memory, its reranker, and its defaults.

        Raises:
            TenantNotFoundError:
                If the tenant was never created, or was deleted.
            ComponentNotEnabledError:
                If the tenant has no event-backed episodic memory.
        """
        raise NotImplementedError
