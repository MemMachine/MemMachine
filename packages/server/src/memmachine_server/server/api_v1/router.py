"""The v1 event-memory routes: tenants, queries, expansion, events.

Every route addresses a tenant by name and reaches that tenant's
`EventMemory` through `TenantEventMemories`, so what a name resolves to
is the deployment's concern and never a route's.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Final

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response

from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.common.reranker import Reranker
from memmachine_server.episodic_memory.event_memory.data_types import (
    DateTimeFormat,
    QueryHit,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    IdKind,
)

from .episodic_session_tenants import EpisodicSessionTenantEventMemories
from .errors import ErrorEnvelopeRoute, ErrorResponse
from .models import (
    EventSpec,
    ExpandRequest,
    ExpandResponse,
    ForgetEventsRequest,
    QueryHitBody,
    QueryRequest,
    QueryResponse,
    RerankSpec,
    SegmentBody,
    StoredEvents,
    TenantBody,
    TenantName,
)
from .tenant_event_memories import (
    TenantEventMemories,
    TenantEventMemory,
    TenantNotFoundError,
)

logger = logging.getLogger(__name__)

RENDERED_IDS: Final[tuple[IdKind, ...]] = ("session", "segment")
"""The markers every rendered window carries: its session, and each line's segments."""


async def get_tenant_event_memories(request: Request) -> TenantEventMemories:
    """The resolver this deployment serves v1 tenants from."""
    return EpisodicSessionTenantEventMemories(request.app.state.mem_machine)


TenantEventMemoriesDependency = Annotated[
    TenantEventMemories, Depends(get_tenant_event_memories)
]

router = APIRouter(
    route_class=ErrorEnvelopeRoute,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

CREATE_TENANT_DESCRIPTION: Final[str] = """
Create the tenant, and the memory its events are held in.

A tenant is named by an opaque string, not a UUID: the server resolves
the name, and a later release resolves it through the tenant registry
without a client changing what it sends. Creation is idempotent: 201
when this request created the tenant, 200 when it already existed.

Sessions have no lifecycle of their own. A session is the `session_id`
an event carries, so a tenant holds as many sessions as its events name.
""".strip()


@router.put(
    "/tenants/{tenant}",
    status_code=201,
    responses={200: {"model": TenantBody, "description": "The tenant already existed"}},
    description=CREATE_TENANT_DESCRIPTION,
    tags=["v1 Tenants"],
)
async def create_tenant(
    tenant: TenantName,
    memories: TenantEventMemoriesDependency,
    response: Response,
) -> TenantBody:
    """Create the tenant if it does not exist."""
    if not await memories.create(tenant):
        response.status_code = 200
    return TenantBody(name=tenant)


@router.get(
    "/tenants/{tenant}",
    description=(
        "The tenant, if it exists. This deployment holds no tenant registry, "
        "so a tenant is its name and the memory behind it, and nothing else "
        "is reported."
    ),
    tags=["v1 Tenants"],
)
async def get_tenant(
    tenant: TenantName,
    memories: TenantEventMemoriesDependency,
) -> TenantBody:
    """Answer with the tenant, or 404."""
    if not await memories.exists(tenant):
        raise TenantNotFoundError(tenant)
    return TenantBody(name=tenant)


@router.delete(
    "/tenants/{tenant}",
    status_code=204,
    description=(
        "Delete the tenant and everything it holds. Deleting a tenant that "
        "does not exist changes nothing and answers the same way."
    ),
    tags=["v1 Tenants"],
)
async def delete_tenant(
    tenant: TenantName,
    memories: TenantEventMemoriesDependency,
) -> None:
    """Delete the tenant, whether or not it exists."""
    await memories.delete(tenant)


QUERY_DESCRIPTION: Final[str] = """
Query the tenant's memory, and render the window around every hit.

The vector search runs first, then the reranker when the request asks
for one and the tenant has one, and the hits are cut to `limit`. Each
hit's `text` is its window rendered with the id markers a further
request continues from: `[session:"<id>"]` heads each block, and
`[segment:<hex32>]` -- or `[segments:<first>..<last>]` for a line built
from more than one segment -- starts each line. `seed` is the index in
`segments` of the segment the query matched.

A segment carries no `position`: this deployment has no event store, so
there is no ingestion order to report. `filter` is an expression in the
server's filter grammar rather than a JSON tree, for the same reason:
the grammar is what this deployment parses.
""".strip()


@router.post(
    "/tenants/{tenant}/episodic-memory/query",
    description=QUERY_DESCRIPTION,
    tags=["v1 Episodic Memory"],
)
async def query_episodic_memory(
    tenant: TenantName,
    spec: QueryRequest,
    memories: TenantEventMemoriesDependency,
) -> QueryResponse:
    """Query the tenant's event memory."""
    resolved = await memories.resolve(tenant)
    defaults = resolved.defaults
    limit = spec.limit if spec.limit is not None else defaults.query_limit
    expand_context = (
        spec.expand_context
        if spec.expand_context is not None
        else defaults.expand_context
    )
    reranking = _reranking(spec.rerank, resolved, limit=limit)
    datetime_format = spec.datetime_format.to_datetime_format()
    hits = await resolved.event_memory.query(
        spec.query,
        vector_search_limit=reranking.candidates if reranking is not None else limit,
        min_cosine_similarity=spec.min_cosine_similarity,
        expand_context=expand_context,
        since=spec.since,
        until=spec.until,
        session_ids=spec.session_ids,
        source_ids=spec.source_ids,
        block_kinds=spec.block_kinds,
        property_filter=parse_filter(spec.filter),
    )
    if reranking is not None:
        hits = await EventMemory.rerank(
            spec.query,
            hits,
            reranker=reranking.reranker,
            datetime_format=datetime_format,
        )
        if reranking.min_score is not None:
            hits = [hit for hit in hits if hit.score >= reranking.min_score]
    return QueryResponse(
        hits=[
            _hit_body(hit, datetime_format=datetime_format, parts=spec.parts)
            for hit in hits[:limit]
        ]
    )


@dataclass(frozen=True)
class _Reranking:
    """The reranking stage of one query: what scores it, and how widely."""

    reranker: Reranker
    candidates: int
    min_score: float | None


def _reranking(
    spec: RerankSpec | None,
    resolved: TenantEventMemory,
    *,
    limit: int,
) -> _Reranking | None:
    """How this query reranks, or None when nothing reranks it.

    The candidates are never fewer than the hits the query answers with.

    Raises:
        ValueError:
            If the request names a reranker. The tenant's own reranker is
            the only one this server scores with.
    """
    if spec is None:
        return None
    if spec.reranker is not None:
        raise ValueError(
            "rerank.reranker names a reranker this server does not resolve; "
            "it scores with the tenant's own reranker"
        )
    if resolved.reranker is None:
        return None
    candidates = (
        spec.candidates
        if spec.candidates is not None
        else resolved.defaults.rerank_candidates
    )
    return _Reranking(
        reranker=resolved.reranker,
        candidates=max(candidates, limit),
        min_score=spec.min_score,
    )


def _hit_body(
    hit: QueryHit,
    *,
    datetime_format: DateTimeFormat,
    parts: Iterable[str],
) -> QueryHitBody:
    """One hit: its window, its seed's place in it, and the rendered text."""
    window = hit.window()
    return QueryHitBody(
        score=hit.score,
        seed=len(hit.neighborhood.before),
        segments=[SegmentBody.from_segment(segment) for segment in window],
        text=EventMemory.render_segments(
            window,
            datetime_format=datetime_format,
            parts=parts,
            ids=RENDERED_IDS,
        ),
    )


EXPAND_DESCRIPTION: Final[str] = """
Walk the timeline out from one segment, backward and forward.

The anchor is a segment uuid, the hex a `[segment:...]` marker carries.
The walk stays inside the anchor's session and never returns the anchor
itself; a side that comes back empty means the session ran out that way.
Each side is rendered with the same id markers a query renders, so its
edges carry the handles a further step continues from.
""".strip()


@router.post(
    "/tenants/{tenant}/episodic-memory/expand",
    description=EXPAND_DESCRIPTION,
    tags=["v1 Episodic Memory"],
)
async def expand_episodic_memory(
    tenant: TenantName,
    spec: ExpandRequest,
    memories: TenantEventMemoriesDependency,
) -> ExpandResponse:
    """Expand around a segment of the tenant's event memory."""
    resolved = await memories.resolve(tenant)
    defaults = resolved.defaults
    datetime_format = spec.datetime_format.to_datetime_format()
    neighborhood = await resolved.event_memory.expand(
        spec.anchor,
        before=spec.before if spec.before is not None else defaults.expand_before,
        after=spec.after if spec.after is not None else defaults.expand_after,
        since=spec.since,
        until=spec.until,
        source_ids=spec.source_ids,
        block_kinds=spec.block_kinds,
        property_filter=parse_filter(spec.filter),
    )
    return ExpandResponse(
        before=[SegmentBody.from_segment(segment) for segment in neighborhood.before],
        after=[SegmentBody.from_segment(segment) for segment in neighborhood.after],
        before_text=EventMemory.render_segments(
            neighborhood.before,
            datetime_format=datetime_format,
            parts=spec.parts,
            ids=RENDERED_IDS,
        ),
        after_text=EventMemory.render_segments(
            neighborhood.after,
            datetime_format=datetime_format,
            parts=spec.parts,
            ids=RENDERED_IDS,
        ),
    )


ADD_EVENTS_DESCRIPTION: Final[str] = """
Remember a batch of events.

An event's `id` is the client's dedupe key: a client that retries a
batch supplies the ids it used, and a batch without them is stored
again. A tenant holds an event once: a batch naming an id the tenant
already holds is rejected whole with `event_exists`, and nothing in it
is stored. Forget the event to store it again.

Ingest is synchronous: the segments and the vector records are written
before this answers. `wait` is accepted so that a client written
against the asynchronous contract works unchanged, and is ignored.
""".strip()


@router.post(
    "/tenants/{tenant}/events",
    description=ADD_EVENTS_DESCRIPTION,
    responses={409: {"model": ErrorResponse}},
    tags=["v1 Events"],
)
async def add_events(
    tenant: TenantName,
    events: list[EventSpec],
    memories: TenantEventMemoriesDependency,
    wait: Annotated[
        float | None,
        Query(
            description="Seconds to wait for the batch; ignored, ingest is synchronous"
        ),
    ] = None,
) -> StoredEvents:
    """Encode a batch of events into the tenant's memory.

    A batch naming an event the tenant already holds is rejected whole,
    as `event_exists`, and stores nothing.
    """
    if wait is not None:
        logger.debug("Ignoring wait=%s seconds: v1 ingest is synchronous", wait)
    resolved = await memories.resolve(tenant)
    built = [spec.to_event() for spec in events]
    await resolved.event_memory.encode_events(built)
    return StoredEvents(stored=[event.uuid for event in built])


@router.post(
    "/tenants/{tenant}/events/delete",
    status_code=200,
    description=(
        "Forget events by their uuids, with everything derived from them. "
        "An id no event is held under forgets nothing and fails nothing."
    ),
    tags=["v1 Events"],
)
async def forget_events(
    tenant: TenantName,
    spec: ForgetEventsRequest,
    memories: TenantEventMemoriesDependency,
) -> None:
    """Forget the named events."""
    resolved = await memories.resolve(tenant)
    await resolved.event_memory.forget_events(spec.ids)


def load_v1_api_router(app: FastAPI) -> APIRouter:
    """Mount the v1 router, whose routes are all under `/v1`."""
    app.include_router(router, prefix="/v1")
    return router
