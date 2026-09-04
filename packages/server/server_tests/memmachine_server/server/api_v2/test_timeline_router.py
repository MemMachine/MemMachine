"""HTTP surface for the timeline endpoints.

The store walk itself is covered against a real backend in
`test_timeline_access`; what is asserted here is the wiring: that each route
reaches the right memory operation, that addresses are resolved on the way in,
and that an address naming nothing -- or more than one segment -- is reported
as such rather than surfacing as an empty result.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from memmachine_server.episodic_memory.event_memory.data_types import (
    ProducerContext,
    QueryResult,
    ScoredSegmentContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segment_store import EventHeader
from memmachine_server.server.api_v2.router import get_memmachine
from memmachine_server.server.app import MemMachineAPI

BASE_TIME = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
SEED_UUID = UUID("aaaaaaaa-0000-0000-0000-000000000001")
OTHER_UUID = UUID("aaaaaaaa-0000-0000-0000-000000000002")
EVENT_UUID = UUID("bbbbbbbb-0000-0000-0000-000000000001")

PROJECT = {"org_id": "org", "project_id": "proj"}


def _segment(uuid: UUID, *, offset: int = 0, text: str = "hello") -> Segment:
    return Segment(
        uuid=uuid,
        event_uuid=EVENT_UUID,
        index=0,
        offset=offset,
        timestamp=BASE_TIME,
        block=TextBlock(text=text),
        context=ProducerContext(producer="alice"),
        properties={},
    )


@pytest.fixture
def long_term_memory():
    memory = AsyncMock()
    memory.resolve_segment_address.return_value = [SEED_UUID]
    memory.abbreviate_segment_addresses.return_value = {
        SEED_UUID: "aaaaaaaa0",
        OTHER_UUID: "aaaaaaaa1",
    }
    memory.get_timeline_segments.return_value = {SEED_UUID: _segment(SEED_UUID)}
    memory.expand_timeline.return_value = [_segment(OTHER_UUID, offset=1, text="next")]
    memory.search_timeline.return_value = QueryResult(
        scored_segment_contexts=[
            ScoredSegmentContext(
                score=0.75,
                seed_segment_uuid=SEED_UUID,
                segments=[_segment(SEED_UUID)],
            )
        ]
    )
    memory.outline_timeline.return_value = [
        EventHeader(
            event_uuid=EVENT_UUID,
            timestamp=BASE_TIME,
            first_segment_uuid=SEED_UUID,
            segment_count=3,
            encoded_length=120,
        )
    ]
    return memory


@pytest.fixture
def client(long_term_memory):
    memmachine = AsyncMock()

    @asynccontextmanager
    async def _open_timeline(_session_data):
        yield long_term_memory

    memmachine.open_timeline = _open_timeline

    app = MemMachineAPI()
    app.dependency_overrides[get_memmachine] = lambda: memmachine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides = {}


def test_search_returns_addressable_matches(client):
    response = client.post(
        "/api/v2/memories/timeline/search",
        json={**PROJECT, "query": "what did we decide", "limit": 5},
    )

    assert response.status_code == 200
    [match] = response.json()["matches"]
    assert match["score"] == 0.75
    assert match["seed"]["handle"] == "aaaaaaaa0"
    assert match["seed"]["segment_uid"] == SEED_UUID.hex
    assert match["seed"]["producer"] == "alice"
    assert "hello" in match["rendered"]


def test_search_passes_the_query_vector_through(client, long_term_memory):
    client.post(
        "/api/v2/memories/timeline/search",
        json={**PROJECT, "query": "cue", "query_vector": [0.1, 0.2]},
    )

    assert long_term_memory.search_timeline.await_args.kwargs["query_vector"] == [
        0.1,
        0.2,
    ]


def test_expand_shows_the_seed_among_its_neighbours(client):
    response = client.post(
        "/api/v2/memories/timeline/expand",
        json={**PROJECT, "handle": "aaaaaaaa0", "before": 1, "after": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["seed"]["handle"] == "aaaaaaaa0"
    # A window whose centre is missing cannot be read as a timeline.
    assert "hello" in body["rendered"]
    assert "next" in body["rendered"]


def test_expand_counts_events_when_asked(client, long_term_memory):
    client.post(
        "/api/v2/memories/timeline/expand",
        json={**PROJECT, "handle": "aaaaaaaa0", "unit": "events"},
    )

    assert long_term_memory.expand_timeline.await_args.kwargs["unit"] == "events"


def test_unknown_address_is_a_404(client, long_term_memory):
    long_term_memory.resolve_segment_address.return_value = []

    response = client.post(
        "/api/v2/memories/timeline/expand",
        json={**PROJECT, "handle": "deadbeef"},
    )

    assert response.status_code == 404


def test_ambiguous_address_is_a_404_naming_the_candidates(client, long_term_memory):
    long_term_memory.resolve_segment_address.return_value = [SEED_UUID, OTHER_UUID]

    response = client.post(
        "/api/v2/memories/timeline/expand",
        json={**PROJECT, "handle": "aaaaaaaa"},
    )

    assert response.status_code == 404
    assert SEED_UUID.hex in str(response.json())


def test_outline_reports_event_shape(client):
    response = client.post(
        "/api/v2/memories/timeline/outline",
        json={**PROJECT, "before": 2, "after": 2},
    )

    assert response.status_code == 200
    [event] = response.json()["events"]
    assert event["handle"] == "aaaaaaaa0"
    assert event["segment_count"] == 3
    assert event["encoded_length"] == 120


def test_outline_around_an_anchor_walks_both_ways(client, long_term_memory):
    client.post(
        "/api/v2/memories/timeline/outline",
        json={**PROJECT, "handle": "aaaaaaaa0", "before": 3, "after": 4},
    )

    calls = long_term_memory.outline_timeline.await_args_list
    assert len(calls) == 2
    backward, forward = calls
    # Each side gets the count it asked for, plus the anchor's own event, which
    # both walks include and the service then folds together.
    assert backward.kwargs["descending"] is True
    assert backward.kwargs["end"] is not None
    assert backward.kwargs["limit"] == 4
    assert "start" in forward.kwargs
    assert forward.kwargs["limit"] == 5


def test_resolve_reports_a_unique_address(client):
    response = client.post(
        "/api/v2/memories/timeline/resolve",
        json={**PROJECT, "handle": "aaaaaaaa0"},
    )

    assert response.status_code == 200
    assert response.json() == {"segment_uid": SEED_UUID.hex, "candidates": []}


def test_resolve_reports_candidates_when_ambiguous(client, long_term_memory):
    long_term_memory.resolve_segment_address.return_value = [SEED_UUID, OTHER_UUID]

    response = client.post(
        "/api/v2/memories/timeline/resolve",
        json={**PROJECT, "handle": "aaaaaaaa"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["segment_uid"] is None
    assert body["candidates"] == [SEED_UUID.hex, OTHER_UUID.hex]


def test_invalid_filter_is_a_422(client):
    response = client.post(
        "/api/v2/memories/timeline/search",
        json={**PROJECT, "query": "cue", "filter": "this is not a filter"},
    )

    assert response.status_code == 422
