"""Tests for the v1 MCP tools: what they render, and what they refuse."""

import re
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx2
import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

from memmachine_server.episodic_memory.event_memory.data_types import (
    Author,
    Context,
    Event,
    TextBlock,
)
from memmachine_server.server.api_v1 import mcp as mcp_module
from memmachine_server.server.api_v1.mcp import (
    NO_BUDGET,
    NOTHING_EARLIER,
    NOTHING_LATER,
    NOTHING_MATCHED,
    TENANT_HEADER,
    mcp,
    mcp_app,
    tenant_context_variable,
)
from memmachine_server.server.api_v1.tenant_event_memories import (
    EpisodicMemoryDefaults,
)
from memmachine_server.server.app import MemMachineAPI
from server_tests.memmachine_server.episodic_memory.event_memory.conftest import (
    AngleEmbedder,
    FakeReranker,
)

_TENANT = "alice"
_T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
_SEGMENT_MARKER = re.compile(r"\[segment:([0-9a-f]{32})\]")


def _event(text, *, session_id="s1", minute=0, author=None, source_id=None):
    return Event(
        uuid=uuid4(),
        timestamp=_T0 + timedelta(minutes=minute),
        session_id=session_id,
        source_id=source_id,
        context=Context(Author(name=author)) if author is not None else Context(),
        blocks=[TextBlock(text=text)],
    )


async def _ingest(memories, events, tenant=_TENANT):
    """Remember events in the tenant's memory, creating the tenant first."""
    await memories.create(tenant)
    resolved = await memories.resolve(tenant)
    await resolved.event_memory.encode_events(events)


def _markers(text):
    """The segment uuids the rendered text carries, in the order it carries them."""
    return _SEGMENT_MARKER.findall(text)


async def _call(client, name, arguments):
    """One tool call's text."""
    result = await client.call_tool(name=name, arguments=arguments)
    return result.data


@pytest.fixture
def angle_memories(memories):
    """Tenants whose texts rank by the token they carry."""
    memories.embedder = AngleEmbedder({"near": 0.0, "far": 1.2, "query": 0.0})
    return memories


@pytest.fixture
def tenant_context():
    token = tenant_context_variable.set(_TENANT)
    yield
    tenant_context_variable.reset(token)


@pytest.fixture
def resolver(memories, monkeypatch):
    """Serve the tools from the fake resolver rather than from a MemMachine."""
    monkeypatch.setattr(mcp_module, "tenant_event_memories", lambda: memories)
    return memories


@pytest_asyncio.fixture
async def mcp_client(resolver, tenant_context):
    async with Client(mcp) as client:
        yield client


# ===================================================================
# the tool surface
# ===================================================================


class TestToolSurface:
    @pytest.mark.asyncio
    async def test_both_tools_are_served_with_their_descriptions(self, mcp_client):
        tools = {tool.name: tool for tool in await mcp_client.list_tools()}
        assert set(tools) == {"memory_query", "memory_expand"}
        assert "cue re-evokes the context" in tools["memory_query"].description
        assert '[session:"<id>"]' in tools["memory_query"].description
        assert "one step in one direction" in tools["memory_expand"].description
        assert "not repeated" in tools["memory_expand"].description

    @pytest.mark.asyncio
    async def test_neither_tool_takes_a_count(self, mcp_client):
        tools = {tool.name: tool for tool in await mcp_client.list_tools()}
        assert set(tools["memory_query"].input_schema["properties"]) == {
            "cue",
            "within",
            "kinds",
            "since",
            "until",
        }
        assert set(tools["memory_expand"].input_schema["properties"]) == {
            "id",
            "direction",
        }


# ===================================================================
# memory_query
# ===================================================================


class TestQuery:
    @pytest.mark.asyncio
    async def test_a_hit_renders_the_window_with_its_markers(
        self, mcp_client, angle_memories
    ):
        # An expansion budget of 3 is one neighbor back and two forward.
        angle_memories.defaults = EpisodicMemoryDefaults(
            query_limit=1, expand_context=3
        )
        await _ingest(
            angle_memories,
            [
                _event("far one", minute=0, author="Alice"),
                _event("near two", minute=1, author="Bob"),
                _event("far three", minute=2, author="Alice"),
            ],
        )
        text = await _call(mcp_client, "memory_query", {"cue": "query"})
        lines = text.splitlines()
        assert lines[0] == '[session:"s1"]'
        assert len(lines) == 4
        assert [line.split("] ", 1)[1] for line in lines[1:]] == [
            '[Jun 1, 2026, 12:00:00\u202fPM UTC] Alice: "far one"',
            '[Jun 1, 2026, 12:01:00\u202fPM UTC] Bob: "near two"',
            '[Jun 1, 2026, 12:02:00\u202fPM UTC] Alice: "far three"',
        ]
        assert len(_markers(text)) == 3

    @pytest.mark.asyncio
    async def test_hits_come_back_best_first_a_blank_line_between_them(
        self, mcp_client, angle_memories
    ):
        angle_memories.defaults = EpisodicMemoryDefaults(expand_context=0)
        await _ingest(
            angle_memories,
            [_event("far one", minute=0), _event("near two", minute=1)],
        )
        text = await _call(mcp_client, "memory_query", {"cue": "query"})
        windows = text.split("\n\n")
        assert len(windows) == 2
        assert "near two" in windows[0]
        assert "far one" in windows[1]

    @pytest.mark.asyncio
    async def test_the_defaults_decide_the_hits_and_the_context(
        self, mcp_client, angle_memories
    ):
        angle_memories.defaults = EpisodicMemoryDefaults(
            query_limit=2, expand_context=0
        )
        await _ingest(
            angle_memories,
            [_event(f"near {index}", minute=index) for index in range(4)],
        )
        text = await _call(mcp_client, "memory_query", {"cue": "query"})
        assert len(text.split("\n\n")) == 2
        assert len(_markers(text)) == 2

    @pytest.mark.asyncio
    async def test_a_query_that_matches_nothing_says_so(self, mcp_client, memories):
        await _ingest(memories, [])
        text = await _call(mcp_client, "memory_query", {"cue": "query"})
        assert text == NOTHING_MATCHED

    @pytest.mark.asyncio
    async def test_within_confines_the_query_to_one_session(
        self, mcp_client, angle_memories
    ):
        angle_memories.defaults = EpisodicMemoryDefaults(expand_context=0)
        await _ingest(
            angle_memories,
            [
                _event("near one", session_id="s1"),
                _event("near two", session_id="s2", minute=1),
            ],
        )
        text = await _call(mcp_client, "memory_query", {"cue": "query", "within": "s2"})
        assert '[session:"s2"]' in text
        assert "near one" not in text

    @pytest.mark.asyncio
    async def test_kinds_select_the_block_kinds(self, mcp_client, angle_memories):
        await _ingest(angle_memories, [_event("near one")])
        kept = await _call(
            mcp_client, "memory_query", {"cue": "query", "kinds": ["text"]}
        )
        assert "near one" in kept

        dropped = await _call(mcp_client, "memory_query", {"cue": "query", "kinds": []})
        assert dropped == NOTHING_MATCHED

    @pytest.mark.asyncio
    async def test_since_and_until_bound_the_timestamps(
        self, mcp_client, angle_memories
    ):
        angle_memories.defaults = EpisodicMemoryDefaults(expand_context=0)
        await _ingest(
            angle_memories,
            [
                _event("near one", minute=0),
                _event("near two", minute=1),
                _event("near three", minute=2),
            ],
        )
        text = await _call(
            mcp_client,
            "memory_query",
            {
                "cue": "query",
                "since": "2026-06-01T12:01:00+00:00",
                "until": "2026-06-01T12:02:00Z",
            },
        )
        assert "near two" in text
        assert "near one" not in text
        assert "near three" not in text

    @pytest.mark.asyncio
    async def test_a_timestamp_without_an_offset_is_a_tool_error(
        self, mcp_client, memories
    ):
        await _ingest(memories, [])
        with pytest.raises(ToolError, match="since carries no offset from UTC"):
            await _call(
                mcp_client,
                "memory_query",
                {"cue": "query", "since": "2026-06-01T12:00:00"},
            )

    @pytest.mark.asyncio
    async def test_a_timestamp_that_is_not_iso_8601_is_a_tool_error(
        self, mcp_client, memories
    ):
        await _ingest(memories, [])
        with pytest.raises(ToolError, match="until is not an ISO 8601 timestamp"):
            await _call(
                mcp_client, "memory_query", {"cue": "query", "until": "last tuesday"}
            )


# ===================================================================
# reranking
# ===================================================================


class TestReranking:
    @pytest.mark.asyncio
    async def test_the_reranker_reorders_the_hits(self, mcp_client, angle_memories):
        angle_memories.reranker = FakeReranker()
        angle_memories.defaults = EpisodicMemoryDefaults(expand_context=0)
        await _ingest(
            angle_memories,
            [_event("near short", minute=0), _event("far " + "long " * 20, minute=1)],
        )
        text = await _call(mcp_client, "memory_query", {"cue": "query"})
        assert text.index("long") < text.index("near short")

    @pytest.mark.asyncio
    async def test_the_candidates_are_fetched_then_cut_to_the_limit(
        self, mcp_client, angle_memories
    ):
        angle_memories.reranker = FakeReranker()
        angle_memories.defaults = EpisodicMemoryDefaults(
            query_limit=1, rerank_candidates=3, expand_context=0
        )
        await _ingest(
            angle_memories,
            [
                _event("near one", minute=0),
                _event("near two", minute=1),
                _event("far " + "long " * 20, minute=2),
            ],
        )
        text = await _call(mcp_client, "memory_query", {"cue": "query"})
        # The vector search ranks the two "near" texts above the "far" one; the
        # reranker scores by length, and the one hit the query answers with is
        # the one the reranker put first.
        assert len(text.split("\n\n")) == 1
        assert "long" in text

    @pytest.mark.asyncio
    async def test_a_tenant_without_a_reranker_ranks_by_similarity(
        self, mcp_client, angle_memories
    ):
        angle_memories.defaults = EpisodicMemoryDefaults(expand_context=0)
        await _ingest(
            angle_memories,
            [_event("near short", minute=0), _event("far " + "long " * 20, minute=1)],
        )
        text = await _call(mcp_client, "memory_query", {"cue": "query"})
        assert text.index("near short") < text.index("long")


# ===================================================================
# memory_expand
# ===================================================================


class TestExpand:
    ANCHOR_INDEX = 10

    async def _ingest_around_an_anchor(self, memories):
        """Twenty-one segments in one session, the anchor in the middle."""
        memories.embedder = AngleEmbedder({"anchor": 0.0, "filler": 1.2, "query": 0.0})
        memories.defaults = EpisodicMemoryDefaults(query_limit=1, expand_context=0)
        await _ingest(
            memories,
            [
                _event(
                    "anchor here" if index == self.ANCHOR_INDEX else f"filler {index}",
                    minute=index,
                )
                for index in range(21)
            ],
        )

    @staticmethod
    async def _anchor(client):
        """The uuid of the one segment a query for the anchor returns."""
        text = await _call(client, "memory_query", {"cue": "query"})
        markers = _markers(text)
        assert len(markers) == 1
        return markers[0]

    @pytest.mark.asyncio
    async def test_around_spends_a_quarter_backward_and_the_rest_forward(
        self, mcp_client, memories
    ):
        await self._ingest_around_an_anchor(memories)
        anchor = await self._anchor(mcp_client)
        text = await _call(mcp_client, "memory_expand", {"id": anchor})
        before, after = text.split("\n\n")
        assert len(_markers(before)) == 2
        assert len(_markers(after)) == 8
        assert "filler 8" in before
        assert "filler 9" in before
        assert "filler 11" in after
        assert "filler 18" in after

    @pytest.mark.asyncio
    async def test_earlier_spends_the_whole_step_backward(self, mcp_client, memories):
        await self._ingest_around_an_anchor(memories)
        anchor = await self._anchor(mcp_client)
        text = await _call(
            mcp_client, "memory_expand", {"id": anchor, "direction": "earlier"}
        )
        assert len(_markers(text)) == 10
        assert "filler 0" in text
        assert "filler 11" not in text

    @pytest.mark.asyncio
    async def test_later_spends_the_whole_step_forward(self, mcp_client, memories):
        await self._ingest_around_an_anchor(memories)
        anchor = await self._anchor(mcp_client)
        text = await _call(
            mcp_client, "memory_expand", {"id": anchor, "direction": "later"}
        )
        assert len(_markers(text)) == 10
        assert "filler 20" in text
        assert "filler 9" not in text

    @pytest.mark.asyncio
    async def test_the_anchor_is_not_returned(self, mcp_client, memories):
        await self._ingest_around_an_anchor(memories)
        anchor = await self._anchor(mcp_client)
        text = await _call(mcp_client, "memory_expand", {"id": anchor})
        assert "anchor here" not in text
        assert anchor not in _markers(text)

    @pytest.mark.asyncio
    async def test_a_side_that_runs_out_says_the_session_ran_out(
        self, mcp_client, memories
    ):
        memories.defaults = EpisodicMemoryDefaults(expand_before=5, expand_after=5)
        await _ingest(memories, [_event("only one")])
        anchor = _markers(await _call(mcp_client, "memory_query", {"cue": "query"}))[0]

        around = await _call(mcp_client, "memory_expand", {"id": anchor})
        assert around == f"{NOTHING_EARLIER}\n\n{NOTHING_LATER}"

        earlier = await _call(
            mcp_client, "memory_expand", {"id": anchor, "direction": "earlier"}
        )
        assert earlier == NOTHING_EARLIER

        later = await _call(
            mcp_client, "memory_expand", {"id": anchor, "direction": "later"}
        )
        assert later == NOTHING_LATER

    @pytest.mark.asyncio
    async def test_a_step_of_no_segments_says_so(self, mcp_client, memories):
        memories.defaults = EpisodicMemoryDefaults(expand_before=0, expand_after=0)
        await _ingest(memories, [_event("only one")])
        anchor = _markers(await _call(mcp_client, "memory_query", {"cue": "query"}))[0]
        assert await _call(mcp_client, "memory_expand", {"id": anchor}) == NO_BUDGET

    @pytest.mark.asyncio
    async def test_every_form_of_a_segment_id_names_the_same_segment(
        self, mcp_client, memories
    ):
        await self._ingest_around_an_anchor(memories)
        anchor = await self._anchor(mcp_client)
        hyphenated = (
            f"{anchor[:8]}-{anchor[8:12]}-{anchor[12:16]}-{anchor[16:20]}-{anchor[20:]}"
        )
        texts = [
            await _call(mcp_client, "memory_expand", {"id": form, "direction": "later"})
            for form in (anchor, f"[segment:{anchor}]", hyphenated)
        ]
        assert texts[0] == texts[1] == texts[2]
        assert "filler 11" in texts[0]

    @pytest.mark.asyncio
    async def test_a_run_of_segments_is_refused_with_its_two_ends(
        self, mcp_client, memories
    ):
        await self._ingest_around_an_anchor(memories)
        anchor = await self._anchor(mcp_client)
        other = "f" * 32
        with pytest.raises(ToolError, match=f"step from one of its ends, {anchor}"):
            await _call(
                mcp_client, "memory_expand", {"id": f"[segments:{anchor}..{other}]"}
            )

    @pytest.mark.asyncio
    async def test_an_id_that_names_no_segment_is_a_tool_error(
        self, mcp_client, memories
    ):
        await _ingest(memories, [])
        with pytest.raises(ToolError, match="names no segment"):
            await _call(mcp_client, "memory_expand", {"id": "the third message"})

    @pytest.mark.asyncio
    async def test_a_segment_the_tenant_does_not_hold_is_a_tool_error(
        self, mcp_client, memories
    ):
        await self._ingest_around_an_anchor(memories)
        with pytest.raises(ToolError, match="is not in this memory"):
            await _call(mcp_client, "memory_expand", {"id": "0" * 32})


# ===================================================================
# the tenant
# ===================================================================


class TestTenant:
    @pytest.mark.asyncio
    async def test_a_request_without_the_header_is_a_tool_error(self, resolver):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=f"no {TENANT_HEADER} header"):
                await _call(client, "memory_query", {"cue": "query"})
            with pytest.raises(ToolError, match=f"no {TENANT_HEADER} header"):
                await _call(client, "memory_expand", {"id": "0" * 32})

    @pytest.mark.asyncio
    async def test_a_tenant_that_was_never_created_is_a_tool_error(self, mcp_client):
        with pytest.raises(ToolError, match="'alice' does not exist"):
            await _call(mcp_client, "memory_query", {"cue": "query"})

    @pytest.mark.asyncio
    async def test_a_tenant_without_an_event_memory_is_a_tool_error(
        self, mcp_client, memories
    ):
        await _ingest(memories, [_event("near one")])
        memories.enabled = False
        with pytest.raises(ToolError, match="Episodic memory is not enabled"):
            await _call(mcp_client, "memory_query", {"cue": "query"})

    @pytest.mark.asyncio
    async def test_the_tools_read_the_tenant_the_header_names(
        self, memories, monkeypatch
    ):
        """The mount, the middleware and the tools, over a real MCP handshake.

        Starlette's `TestClient` cannot run the streamable HTTP handshake,
        so the application is served in process through an ASGI transport.
        The tenant is on the request and in no context variable of this
        test, so a tool that answers at all read the header.
        """
        monkeypatch.setattr(mcp_module, "tenant_event_memories", lambda: memories)
        await _ingest(memories, [_event("near one", author="Alice")])
        application = MemMachineAPI()

        def client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx2.Timeout | None = None,
            auth: httpx2.Auth | None = None,
            follow_redirects: bool = True,
        ) -> httpx2.AsyncClient:
            return httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=application),
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
                auth=auth,
                follow_redirects=follow_redirects,
            )

        async with mcp_app.lifespan(application):
            transport = StreamableHttpTransport(
                url="http://testserver/v1/mcp/",
                headers={TENANT_HEADER: _TENANT},
                httpx_client_factory=client_factory,
            )
            async with Client(transport) as client:
                text = await _call(client, "memory_query", {"cue": "query"})
        assert "near one" in text
        assert '[session:"s1"]' in text
