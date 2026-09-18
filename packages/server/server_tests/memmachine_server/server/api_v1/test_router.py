"""Tests for the v1 routes: tenants, events, queries and expansion."""

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from memmachine_server.server.api_v1.tenant_event_memories import (
    EpisodicMemoryDefaults,
)
from memmachine_server.server.api_v2.service import get_memmachine
from server_tests.memmachine_server.episodic_memory.event_memory.conftest import (
    AngleEmbedder,
    FakeReranker,
)

from .conftest import FakeTenantEventMemories

_TENANT = "alice"
_BASE_TIMESTAMP = "2025-06-01T12:0{minute}:00+00:00"


def _event(
    text,
    *,
    session_id="s1",
    minute=0,
    author=None,
    properties=None,
    event_id=None,
    source_id=None,
):
    body = {
        "session_id": session_id,
        "timestamp": _BASE_TIMESTAMP.format(minute=minute),
        "blocks": [{"kind": "text", "text": text}],
    }
    if author is not None:
        body["context"] = {"author": {"name": author}}
    if properties is not None:
        body["properties"] = properties
    if event_id is not None:
        body["id"] = event_id
    if source_id is not None:
        body["source_id"] = source_id
    return body


def _create(client, tenant=_TENANT):
    response = client.put(f"/v1/tenants/{tenant}")
    assert response.status_code == 201
    return response


def _ingest(client, events, tenant=_TENANT):
    response = client.post(f"/v1/tenants/{tenant}/events", json=events)
    assert response.status_code == 200, response.text
    return response.json()["stored"]


def _query(client, tenant=_TENANT, **body):
    body.setdefault("query", "query")
    return client.post(f"/v1/tenants/{tenant}/episodic-memory/query", json=body)


def _error(response):
    return response.json()["error"]


# ===================================================================
# tenants
# ===================================================================


class TestTenants:
    def test_create_answers_201_then_200(self, client):
        created = client.put(f"/v1/tenants/{_TENANT}")
        assert created.status_code == 201
        assert created.json() == {"name": _TENANT}

        again = client.put(f"/v1/tenants/{_TENANT}")
        assert again.status_code == 200
        assert again.json() == {"name": _TENANT}

    def test_get_answers_404_until_the_tenant_is_created(self, client):
        missing = client.get(f"/v1/tenants/{_TENANT}")
        assert missing.status_code == 404
        assert _error(missing)["code"] == "tenant_not_found"

        _create(client)
        assert client.get(f"/v1/tenants/{_TENANT}").json() == {"name": _TENANT}

    def test_delete_is_idempotent(self, client):
        assert client.delete(f"/v1/tenants/{_TENANT}").status_code == 204

        _create(client)
        assert client.delete(f"/v1/tenants/{_TENANT}").status_code == 204
        assert client.get(f"/v1/tenants/{_TENANT}").status_code == 404

    def test_a_name_longer_than_the_bound_is_rejected(self, client):
        response = client.put(f"/v1/tenants/{'n' * 256}")
        assert response.status_code == 422
        assert _error(response)["code"] == "invalid_request"

    def test_a_tenant_name_holding_a_slash_addresses_no_route(self, client):
        assert client.put("/v1/tenants/one/two").status_code == 404


# ===================================================================
# events
# ===================================================================


class TestEvents:
    def test_ingest_keeps_the_given_id_and_mints_the_missing_one(self, client):
        _create(client)
        given = str(uuid4())
        stored = _ingest(
            client,
            [_event("near one", event_id=given), _event("near two", minute=1)],
        )
        assert stored[0] == given
        assert UUID(stored[1]) != UUID(given)

    def test_ingest_is_reachable_by_a_query(self, client):
        _create(client)
        _ingest(client, [_event("near one", author="Alice")])
        hits = _query(client).json()["hits"]
        assert len(hits) == 1
        assert "near one" in hits[0]["text"]

    def test_wait_is_accepted_and_ignored(self, client):
        _create(client)
        response = client.post(
            f"/v1/tenants/{_TENANT}/events?wait=5", json=[_event("near one")]
        )
        assert response.status_code == 200
        assert len(response.json()["stored"]) == 1

    def test_reingesting_a_held_id_answers_409_naming_it(self, client):
        _create(client)
        held = str(uuid4())
        _ingest(client, [_event("near one", event_id=held)])

        response = client.post(
            f"/v1/tenants/{_TENANT}/events",
            json=[
                _event("near two", event_id=held, minute=1),
                _event("near three", minute=2),
            ],
        )

        assert response.status_code == 409
        error = _error(response)
        assert error["code"] == "event_exists"
        assert held in error["message"]
        # The batch was rejected whole: the event beside it is not stored.
        texts = [hit["text"] for hit in _query(client).json()["hits"]]
        assert not any("near three" in text for text in texts)

    def test_ingest_into_an_unknown_tenant_answers_404(self, client):
        response = client.post(f"/v1/tenants/{_TENANT}/events", json=[_event("hi")])
        assert response.status_code == 404
        assert _error(response)["code"] == "tenant_not_found"

    def test_a_reserved_property_key_is_rejected(self, client):
        _create(client)
        response = client.post(
            f"/v1/tenants/{_TENANT}/events",
            json=[_event("near one", properties={"memmachine_em_session": "x"})],
        )
        assert response.status_code == 422
        error = _error(response)
        assert error["code"] == "invalid_request"
        assert "reserved" in error["message"]

    def test_a_naive_timestamp_is_rejected(self, client):
        _create(client)
        body = _event("near one")
        body["timestamp"] = "2025-06-01T12:00:00"
        response = client.post(f"/v1/tenants/{_TENANT}/events", json=[body])
        assert response.status_code == 422
        assert _error(response)["code"] == "invalid_request"

    def test_an_unregistered_block_kind_is_rejected(self, client):
        _create(client)
        body = _event("near one")
        body["blocks"] = [{"kind": "hologram", "text": "near one"}]
        response = client.post(f"/v1/tenants/{_TENANT}/events", json=[body])
        assert response.status_code == 422
        assert _error(response)["code"] == "invalid_request"

    def test_an_empty_session_id_is_rejected(self, client):
        _create(client)
        response = client.post(
            f"/v1/tenants/{_TENANT}/events", json=[_event("near one", session_id="")]
        )
        assert response.status_code == 422
        assert _error(response)["code"] == "invalid_request"

    def test_forget_removes_the_event_from_a_query(self, client):
        _create(client)
        stored = _ingest(client, [_event("near one")])
        forgotten = client.post(
            f"/v1/tenants/{_TENANT}/events/delete", json={"ids": stored}
        )
        assert forgotten.status_code == 200
        assert _query(client).json()["hits"] == []

    def test_forgetting_an_unheld_id_changes_nothing(self, client):
        _create(client)
        _ingest(client, [_event("near one")])
        response = client.post(
            f"/v1/tenants/{_TENANT}/events/delete", json={"ids": [str(uuid4())]}
        )
        assert response.status_code == 200
        assert len(_query(client).json()["hits"]) == 1


# ===================================================================
# query
# ===================================================================


class TestQuery:
    def test_a_hit_carries_its_window_its_seed_and_the_id_markers(
        self, client, memories
    ):
        memories.embedder = AngleEmbedder({"near": 0.0, "far": 1.2, "query": 0.0})
        _create(client)
        _ingest(
            client,
            [
                _event("far one", minute=0, author="Alice"),
                _event("near two", minute=1, author="Bob"),
                _event("far three", minute=2, author="Alice"),
            ],
        )
        # An expansion budget of 3 is one neighbor back and two forward.
        hits = _query(client, limit=1, expand_context=3).json()["hits"]
        assert len(hits) == 1
        hit = hits[0]
        assert [segment["block"]["text"] for segment in hit["segments"]] == [
            "far one",
            "near two",
            "far three",
        ]
        assert hit["seed"] == 1
        seed = hit["segments"][hit["seed"]]
        assert seed["session_id"] == "s1"
        assert seed["source_id"] is None
        assert seed["index"] == 0
        assert seed["offset"] == 0
        assert seed["timestamp"].startswith("2025-06-01T12:01:00")
        assert seed["context"] == {"author": {"name": "Bob"}}
        assert seed["block"] == {"kind": "text", "text": "near two"}
        assert seed["properties"] == {}
        assert '[session:"s1"]' in hit["text"]
        assert hit["text"].count("[segment:") == 3
        assert "Alice: " in hit["text"]
        assert "Bob: " in hit["text"]

    def test_parts_choose_what_the_header_carries(self, client):
        _create(client)
        _ingest(client, [_event("near one", author="Alice")])
        hits = _query(client, parts=[]).json()["hits"]
        assert "Alice" not in hits[0]["text"]

    def test_limit_caps_the_hits(self, client):
        _create(client)
        _ingest(
            client,
            [_event(f"near {index}", minute=index) for index in range(4)],
        )
        assert len(_query(client, limit=2).json()["hits"]) == 2

    def test_session_ids_and_source_ids_select(self, client):
        _create(client)
        _ingest(
            client,
            [
                _event("near one", session_id="s1", source_id="claude-code"),
                _event("near two", session_id="s2", source_id="codex", minute=1),
            ],
        )
        one = _query(client, session_ids=["s1"]).json()["hits"]
        assert [hit["segments"][hit["seed"]]["session_id"] for hit in one] == ["s1"]

        two = _query(client, source_ids=["codex"]).json()["hits"]
        assert [hit["segments"][hit["seed"]]["source_id"] for hit in two] == ["codex"]

    def test_the_filter_expression_selects_by_property(self, client):
        _create(client)
        _ingest(
            client,
            [
                _event("near one", properties={"color": "red"}),
                _event("near two", minute=1, properties={"color": "blue"}),
            ],
        )
        hits = _query(client, filter='m.color = "red"').json()["hits"]
        assert len(hits) == 1
        assert hits[0]["segments"][hits[0]["seed"]]["properties"] == {"color": "red"}

    def test_since_and_until_bound_the_timestamps(self, client):
        _create(client)
        _ingest(
            client,
            [_event("near one", minute=0), _event("near two", minute=2)],
        )
        hits = _query(client, since="2025-06-01T12:01:00+00:00").json()["hits"]
        assert len(hits) == 1
        assert "near two" in hits[0]["text"]

    def test_an_unknown_timezone_is_rejected(self, client):
        _create(client)
        response = _query(client, datetime_format={"timezone": "Mars/Olympus"})
        assert response.status_code == 422
        assert "Mars/Olympus" in _error(response)["message"]

    def test_an_unknown_tenant_answers_404(self, client):
        response = _query(client)
        assert response.status_code == 404
        assert _error(response)["code"] == "tenant_not_found"

    def test_a_tenant_without_the_component_answers_404(self, client, memories):
        _create(client)
        memories.enabled = False
        response = _query(client)
        assert response.status_code == 404
        assert _error(response)["code"] == "component_not_enabled"

    def test_an_unexpected_failure_answers_500_without_a_traceback(
        self, client, memories
    ):
        memories.failure = RuntimeError("the store fell over")
        response = _query(client)
        assert response.status_code == 500
        error = _error(response)
        assert error["code"] == "internal"
        assert "Traceback" not in error["message"]
        assert "fell over" not in error["message"]


# ===================================================================
# reranking
# ===================================================================


class TestReranking:
    @pytest.fixture
    def reranking_client(self, client, memories):
        memories.embedder = AngleEmbedder({"near": 0.0, "far": 1.2, "query": 0.0})
        memories.reranker = FakeReranker()
        memories.defaults = EpisodicMemoryDefaults(query_limit=10)
        return client

    @staticmethod
    def _ingest_two(client):
        _create(client)
        _ingest(
            client,
            [
                _event("near short", minute=0),
                _event("far " + "long " * 20, minute=1),
            ],
        )

    def test_the_reranker_reorders_the_hits(self, reranking_client):
        self._ingest_two(reranking_client)
        hits = _query(reranking_client).json()["hits"]
        assert "far" in hits[0]["text"]
        assert hits[0]["score"] > hits[1]["score"]

    def test_a_null_rerank_leaves_the_vector_order(self, reranking_client):
        self._ingest_two(reranking_client)
        hits = _query(reranking_client, rerank=None).json()["hits"]
        assert "near short" in hits[0]["text"]

    def test_min_score_drops_the_hits_below_it(self, reranking_client):
        self._ingest_two(reranking_client)
        scored = _query(reranking_client).json()["hits"]
        assert len(scored) == 2
        between = (scored[0]["score"] + scored[1]["score"]) / 2

        hits = _query(reranking_client, rerank={"min_score": between}).json()["hits"]
        assert len(hits) == 1
        assert "far" in hits[0]["text"]

    def test_a_tenant_without_a_reranker_ranks_by_similarity(self, client, memories):
        memories.embedder = AngleEmbedder({"near": 0.0, "far": 1.2, "query": 0.0})
        self._ingest_two(client)
        hits = _query(client).json()["hits"]
        assert "near short" in hits[0]["text"]

    def test_naming_a_reranker_is_rejected(self, reranking_client):
        self._ingest_two(reranking_client)
        response = _query(reranking_client, rerank={"reranker": "cohere"})
        assert response.status_code == 422
        assert "reranker" in _error(response)["message"]


# ===================================================================
# expansion
# ===================================================================


class TestExpand:
    @staticmethod
    def _ingest_five(client):
        _create(client)
        return _ingest(
            client,
            [_event(f"near {index}", minute=index) for index in range(5)],
        )

    @staticmethod
    def _anchor(client, text):
        hits = _query(client, limit=5, expand_context=0).json()["hits"]
        for hit in hits:
            segment = hit["segments"][hit["seed"]]
            if segment["block"]["text"] == text:
                return segment["uuid"]
        raise AssertionError(f"no hit on {text!r}")

    def test_both_sides_come_back_in_order_and_rendered(self, client):
        self._ingest_five(client)
        anchor = self._anchor(client, "near 2")
        response = client.post(
            f"/v1/tenants/{_TENANT}/episodic-memory/expand",
            json={"anchor": anchor, "before": 1, "after": 2},
        )
        assert response.status_code == 200
        body = response.json()
        assert [segment["block"]["text"] for segment in body["before"]] == ["near 1"]
        assert [segment["block"]["text"] for segment in body["after"]] == [
            "near 3",
            "near 4",
        ]
        assert '[session:"s1"]' in body["before_text"]
        assert "[segment:" in body["after_text"]
        assert "near 1" in body["before_text"]
        assert "near 3" in body["after_text"]

    def test_the_defaults_fill_in_the_counts(self, client, memories):
        memories.defaults = EpisodicMemoryDefaults(expand_before=1, expand_after=1)
        self._ingest_five(client)
        anchor = self._anchor(client, "near 2")
        body = client.post(
            f"/v1/tenants/{_TENANT}/episodic-memory/expand",
            json={"anchor": anchor},
        ).json()
        assert [segment["block"]["text"] for segment in body["before"]] == ["near 1"]
        assert [segment["block"]["text"] for segment in body["after"]] == ["near 3"]

    def test_a_side_that_runs_out_comes_back_empty(self, client):
        self._ingest_five(client)
        anchor = self._anchor(client, "near 0")
        body = client.post(
            f"/v1/tenants/{_TENANT}/episodic-memory/expand",
            json={"anchor": anchor, "before": 2, "after": 1},
        ).json()
        assert body["before"] == []
        assert body["before_text"] == ""
        assert [segment["block"]["text"] for segment in body["after"]] == ["near 1"]

    def test_an_anchor_of_no_segment_is_rejected(self, client):
        self._ingest_five(client)
        response = client.post(
            f"/v1/tenants/{_TENANT}/episodic-memory/expand",
            json={"anchor": str(uuid4())},
        )
        assert response.status_code == 422
        assert _error(response)["code"] == "invalid_request"

    def test_an_anchor_that_is_not_a_uuid_is_rejected(self, client):
        self._ingest_five(client)
        response = client.post(
            f"/v1/tenants/{_TENANT}/episodic-memory/expand",
            json={"anchor": "not-a-uuid"},
        )
        assert response.status_code == 422
        assert _error(response)["code"] == "invalid_request"

    def test_a_negative_count_is_rejected(self, client):
        self._ingest_five(client)
        response = client.post(
            f"/v1/tenants/{_TENANT}/episodic-memory/expand",
            json={"anchor": str(uuid4()), "before": -1},
        )
        assert response.status_code == 422
        assert _error(response)["code"] == "invalid_request"


def test_a_v2_route_keeps_its_own_error_shape(client):
    """The v1 envelope is the v1 routes' alone: v2 answers as it always did."""
    client.app.dependency_overrides[get_memmachine] = AsyncMock
    response = client.post("/api/v2/projects", json={"org_id": "a/b"})
    assert response.status_code == 422
    body = response.json()
    assert "error" not in body
    assert "detail" in body


def test_the_fake_resolver_is_the_one_the_routes_use(client, memories):
    assert isinstance(memories, FakeTenantEventMemories)
    _create(client)
    assert list(memories.memories) == [_TENANT]
