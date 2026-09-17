"""The two recall tools a coding agent reaches the v1 event memory through.

`memory_query` finds the windows a cue evokes and `memory_expand` walks
the timeline out from a segment one of them rendered. Both read one
tenant's memory: the name arrives in the `X-MemMachine-Tenant` header of
the MCP request, and `TenantContextMiddleware` puts it in a context
variable of this module's own, so this app and the v2 MCP app never read
each other's values. The tools resolve the name through the seam the v1
routes resolve it through and call the memory in process, so nothing here
depends on the HTTP routes.

Neither tool takes a count: how many hits a query answers with, how much
context is rendered around each, and how far a step walks are the
deployment's `EpisodicMemoryDefaults`, and the model chooses a place and a
direction instead of an amount.
"""

import contextvars
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Final, Literal
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.http import StarletteWithLifespan
from pydantic import Field
from starlette.applications import Starlette
from starlette.types import Lifespan, Receive, Scope, Send

from memmachine_server.episodic_memory.event_memory.data_types import (
    DateTimeFormat,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.server.api_v2 import mcp as api_v2_mcp

from .episodic_session_tenants import EpisodicSessionTenantEventMemories
from .router import RENDERED_IDS
from .tenant_event_memories import (
    ComponentNotEnabledError,
    TenantEventMemories,
    TenantEventMemory,
    TenantNotFoundError,
)

TENANT_HEADER: Final[str] = "X-MemMachine-Tenant"
"""The request header naming the tenant whose memory the tools read."""

RENDERED_DATETIME_FORMAT: Final[DateTimeFormat] = DateTimeFormat(
    date_style="medium",
    time_style="long",
    locale="en_US",
)
"""How every timestamp the tools render is written.

The date is what a reader places a memory by, so it is written out
("Jun 1, 2026") rather than as digits, and the weekday it would carry at
`"full"` is dropped from every line of every window. The time keeps its
zone, so two sessions recorded at different offsets stay comparable.
"""

NOTHING_MATCHED: Final[str] = "Nothing in this tenant's memory matched that cue."
"""What a query that matched no segment answers with."""

NOTHING_EARLIER: Final[str] = "Nothing earlier: the session starts here."
"""What the backward side of an expansion that ran out answers with."""

NOTHING_LATER: Final[str] = "Nothing later: the session ends here."
"""What the forward side of an expansion that ran out answers with."""

NO_BUDGET: Final[str] = "This deployment's expansion step is zero segments."
"""What an expansion answers with when the tenant's defaults walk nowhere."""

tenant_context_variable: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "v1_mcp_tenant", default=None
)
"""The tenant the request being served names, or None when it names none."""


class TenantContextMiddleware:
    """Reads the tenant header of every request into `tenant_context_variable`.

    Wrapping the MCP app rather than the whole server sets the variable
    for the requests the v1 tools serve and for no others.
    """

    def __init__(
        self, app: StarletteWithLifespan, header_name: str = TENANT_HEADER
    ) -> None:
        """Serve `app`, taking the tenant from the `header_name` header."""
        self.app = app
        self._header_name = header_name.lower()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one request with the tenant it names in the context variable."""
        tenant: str | None = None
        if scope.get("type") == "http":
            headers = {
                name.decode().lower(): value.decode()
                for name, value in scope.get("headers", [])
            }
            tenant = headers.get(self._header_name)
        token = tenant_context_variable.set(tenant)
        try:
            await self.app(scope, receive, send)
        finally:
            tenant_context_variable.reset(token)

    @property
    def lifespan(self) -> Lifespan[Starlette]:
        """The wrapped application's lifespan handler."""
        return self.app.lifespan


mcp: Final[FastMCP] = FastMCP("MemMachine event memory")
"""The MCP server holding the v1 recall tools."""

mcp_app: Final[TenantContextMiddleware] = TenantContextMiddleware(
    mcp.http_app(path="/")
)
"""The ASGI application serving those tools, mounted at `/v1/mcp`."""


QUERY_DESCRIPTION: Final[str] = """
Query this tenant's memory of past sessions; read the window around every hit.

A cue re-evokes the context a memory was encoded in, not just the name of a
thing in it. Describe the moment -- what was being done, said or decided, and
why -- or quote the line as it was said, with the speaker. A bare entity ("the
parser", "docker") is too diffuse to pin an episode. The user's own wording is
a fine cue; a question and a statement both work, and trying the literal
wording and then a sharper one costs one more call. When the target cannot be
pinned, query what surrounded it and walk to it with memory_expand.
Following a lead you just read is another query with a new cue.

Hits come back best first, a blank line between them, each a window of one
session's timeline. Read the markers:
  [session:"<id>"]      heads a window; that id is what `within` takes.
  [segment:<hex32>]     starts a line; that hex is what memory_expand takes.
  [segments:<a>..<b>]   a line built from a run of segments; either end is a
                        handle memory_expand takes.
A window is an append you read once: it does not come back, so take what you
need from it now.

How many hits come back, and how much of the timeline is rendered around each,
is the deployment's choice rather than a parameter.
""".strip()


@mcp.tool(name="memory_query", description=QUERY_DESCRIPTION)
async def memory_query(
    cue: Annotated[
        str, Field(description="The context to look for, in the model's own words")
    ],
    within: Annotated[
        str | None,
        Field(description='The session to look inside, the id in [session:"..."]'),
    ] = None,
    kinds: Annotated[
        list[str] | None,
        Field(description="The block kinds to keep; null keeps every kind"),
    ] = None,
    since: Annotated[
        str | None,
        Field(
            description=(
                "Inclusive lower bound on the memories' timestamps, ISO 8601 "
                "with an offset"
            )
        ),
    ] = None,
    until: Annotated[
        str | None,
        Field(
            description=(
                "Exclusive upper bound on the memories' timestamps, ISO 8601 "
                "with an offset"
            )
        ),
    ] = None,
) -> str:
    """The windows this tenant's memory matches the cue with, best first."""
    resolved = await resolve_tenant()
    defaults = resolved.defaults
    limit = defaults.query_limit
    reranker = resolved.reranker
    hits = await resolved.event_memory.query(
        cue,
        vector_search_limit=(
            max(defaults.rerank_candidates, limit) if reranker is not None else limit
        ),
        expand_context=defaults.expand_context,
        since=_timestamp(since, name="since"),
        until=_timestamp(until, name="until"),
        session_ids=None if within is None else [within],
        block_kinds=kinds,
    )
    if reranker is not None:
        hits = await EventMemory.rerank(
            cue,
            hits,
            reranker=reranker,
            datetime_format=RENDERED_DATETIME_FORMAT,
        )
    windows = [
        EventMemory.render_segments(
            hit.window(),
            datetime_format=RENDERED_DATETIME_FORMAT,
            ids=RENDERED_IDS,
        )
        for hit in hits[:limit]
    ]
    if not windows:
        return NOTHING_MATCHED
    return "\n\n".join(windows)


EXPAND_DESCRIPTION: Final[str] = """
Read the timeline around a segment you already hold, one step in one direction.

`id` is a handle a rendered line carries: `[segment:<hex32>]`, or the bare
32-hex uuid, with or without hyphens. A `[segments:<a>..<b>]` line is a run of
segments, so give one of its two ends rather than the range.

A step is a fixed number of segments the deployment chooses. You choose the
place, the direction, and whether to step again:
  around   a quarter of the step backward and the rest forward (the default):
           the rest of the message the handle points into lies forward, and
           what a moment turned out to mean arrives in later turns.
  earlier  the whole step backward from the given segment.
  later    the whole step forward.

The segment you named is not repeated -- you already hold it. Each side comes
back with the markers a query renders, so its edges are the handles a further
step continues from, and a side that says the session ran out is the end of
that session that way. What comes back is an append you read once.
""".strip()


@mcp.tool(name="memory_expand", description=EXPAND_DESCRIPTION)
async def memory_expand(
    # The tool's parameter is `id`, the name the marker grammar suggests;
    # the Python parameter is named around the builtin.
    segment: Annotated[
        str,
        Field(
            alias="id",
            description="The segment to step from, its hex or its [segment:...] marker",
        ),
    ],
    direction: Annotated[
        Literal["around", "earlier", "later"],
        Field(description="Which way the step is spent"),
    ] = "around",
) -> str:
    """The timeline on one or both sides of a segment, rendered with its ids."""
    resolved = await resolve_tenant()
    anchor = _anchor(segment)
    budget = resolved.defaults.expand_before + resolved.defaults.expand_after
    match direction:
        case "around":
            before = budget // 4
            after = budget - before
        case "earlier":
            before, after = budget, 0
        case "later":
            before, after = 0, budget
    try:
        neighborhood = await resolved.event_memory.expand(
            anchor, before=before, after=after
        )
    except LookupError as error:
        raise ToolError(str(error)) from error
    sides: list[str] = []
    if before > 0:
        sides.append(_side(neighborhood.before, NOTHING_EARLIER))
    if after > 0:
        sides.append(_side(neighborhood.after, NOTHING_LATER))
    if not sides:
        return NO_BUDGET
    return "\n\n".join(sides)


async def resolve_tenant() -> TenantEventMemory:
    """The event memory of the tenant this request names, and its defaults.

    Raises:
        ToolError:
            If the request named no tenant, if no tenant of that name
            exists, or if the tenant has no event memory to read.
    """
    tenant = tenant_context_variable.get()
    if not tenant:
        raise ToolError(f"This request carries no {TENANT_HEADER} header")
    try:
        return await tenant_event_memories().resolve(tenant)
    except (TenantNotFoundError, ComponentNotEnabledError) as error:
        raise ToolError(str(error)) from error


def tenant_event_memories() -> TenantEventMemories:
    """The resolver this deployment serves v1 tenants from.

    The `MemMachine` is the one this process started, the instance the v2
    MCP app and the v1 routes read, so a tenant reached through a tool and
    the same tenant reached through a route are one memory.

    Raises:
        ToolError: If the process has not started its memory.
    """
    memmachine = api_v2_mcp.mem_machine
    if memmachine is None:
        raise ToolError("This server has not started its memory")
    return EpisodicSessionTenantEventMemories(memmachine)


def _timestamp(value: str | None, *, name: str) -> datetime | None:
    """The instant a bound names, or None when the request left it out.

    Raises:
        ToolError:
            If `value` is not an ISO 8601 timestamp, or carries no offset
            from UTC, which leaves the instant it names undecided.
    """
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ToolError(f"{name} is not an ISO 8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise ToolError(
            f"{name} carries no offset from UTC: {value!r}. Write one, as in "
            "2026-09-01T09:00:00-07:00 or 2026-09-01T16:00:00Z"
        )
    return parsed


def _anchor(handle: str) -> UUID:
    """The segment uuid a handle names.

    Raises:
        ToolError:
            If the handle is a range of segments, or names no uuid at all.
    """
    text = handle.strip()
    marker = _SEGMENT_MARKER.fullmatch(text)
    if marker is not None:
        text = marker.group("uuid")
    else:
        run = _SEGMENT_RUN_MARKER.fullmatch(text)
        if run is not None:
            raise ToolError(
                f"{handle} is a run of segments; step from one of its ends, "
                f"{run.group('first')} or {run.group('last')}"
            )
    try:
        return UUID(text)
    except ValueError as error:
        raise ToolError(
            f"{handle!r} names no segment; give the hex of a [segment:<hex>] "
            "marker, with or without hyphens"
        ) from error


_SEGMENT_MARKER: Final[re.Pattern[str]] = re.compile(
    r"\[segment:(?P<uuid>[0-9a-fA-F]{32})\]"
)
"""The marker a rendered line built from one segment starts with."""

_SEGMENT_RUN_MARKER: Final[re.Pattern[str]] = re.compile(
    r"\[segments:(?P<first>[0-9a-fA-F]{32})\.\.(?P<last>[0-9a-fA-F]{32})\]"
)
"""The marker a rendered line built from several segments starts with."""


def _side(segments: Sequence[Segment], ran_out: str) -> str:
    """One side of an expansion: its rendered segments, or why it has none."""
    if not segments:
        return ran_out
    return EventMemory.render_segments(
        segments,
        datetime_format=RENDERED_DATETIME_FORMAT,
        ids=RENDERED_IDS,
    )
