"""Tests for EventMemory."""

import datetime
import json
import math
import random
from datetime import UTC
from typing import Any, ClassVar, override
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    map_filter_fields,
)
from memmachine_server.common.vector_store.data_types import Record
from memmachine_server.episodic_memory.event_memory.data_types import (
    Author,
    Context,
    ContextPart,
    DateTimeFormat,
    Event,
    EvictionOptions,
    QueryHit,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    BLOCK_KIND_KEY,
    EVENT_SOURCE_KEY,
    EVENT_TIMESTAMP_KEY,
    EventMemory,
    EventMemoryPartition,
    _system_predicates,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from server_tests.memmachine_server.common.filter.random_filters import (
    random_filter,
    random_properties,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_partition import (
    evaluate_filter,
)

from .conftest import (
    AngleEmbedder,
    FakeReranker,
    InMemoryEventMemoryStorePartition,
    InMemoryVectorStorePartition,
    make_partition,
    open_event_memory_partition,
)

_async = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime.datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
_SHORT_TIME = DateTimeFormat(time_style="short")


def _record_properties(record: Record) -> dict[str, PropertyValue]:
    """Get properties from a record, asserting they are not None."""
    assert record.properties is not None
    return record.properties


def _record_query_filters(
    partition: InMemoryVectorStorePartition, monkeypatch: pytest.MonkeyPatch
) -> list[FilterExpr | None]:
    """Record the filter of every query the partition gets, in order."""
    seen: list[FilterExpr | None] = []
    query = partition.query

    async def spy(*args, **kwargs):
        seen.append(kwargs["property_filter"])
        return await query(*args, **kwargs)

    monkeypatch.setattr(partition, "query", spy)
    return seen


def _author(name: str) -> Context:
    return Context(Author(name=name))


def _make_event(
    text: str,
    *,
    timestamp: datetime.datetime = _T0,
    source_id: str = "src",
    context: Context | None = None,
    properties=None,
) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=timestamp,
        source_id=source_id,
        context=context if context is not None else Context(),
        blocks=[TextBlock(text=text)],
        properties=properties or {},
    )


def _ts(minutes: int) -> datetime.datetime:
    """Return _T0 + minutes."""
    return _T0 + datetime.timedelta(minutes=minutes)


def _texts(hits: list[QueryHit]) -> set[str]:
    return {
        seg.block.text
        for hit in hits
        for seg in hit.window()
        if isinstance(seg.block, TextBlock)
    }


async def _build(
    embedder: FakeEmbedder,
    *,
    partition: InMemoryEventMemoryStorePartition | None = None,
    vector_store_partition: InMemoryVectorStorePartition | None = None,
    deriver: Deriver | None = None,
    eviction: EvictionOptions | None = None,
) -> EventMemoryPartition:
    params: dict[str, Any] = {
        "event_memory_store_partition": partition
        or InMemoryEventMemoryStorePartition(),
        "vector_store_partition": vector_store_partition or make_partition(),
        "segmenter": Segmenter([TextSegmenter()]),
        "deriver": deriver or Deriver([WholeTextDeriver()]),
        "embedder": embedder,
        "eviction": eviction,
    }
    return await open_event_memory_partition(**params)


# ===================================================================
# schema
# ===================================================================


class TestSchema:
    def test_expected_vector_store_collection_schema_declares_the_reserved_keys(self):
        assert EventMemory.expected_vector_store_collection_schema() == {
            EVENT_TIMESTAMP_KEY: datetime.datetime,
            EVENT_SOURCE_KEY: str,
            BLOCK_KIND_KEY: str,
        }
        assert all(
            key.startswith("memmachine_")
            for key in EventMemory.expected_vector_store_collection_schema()
        )


# ===================================================================
# encode_events
# ===================================================================


@_async
class TestEncodeEvents:
    async def test_single_text_event(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
    ):
        event = _make_event("hello world")
        await event_memory_partition.encode_events([event])

        # One event memory stored.
        assert len(fake_event_memory_store_partition.segments) == 1
        segment = next(iter(fake_event_memory_store_partition.segments.values()))
        assert segment.event_uuid == event.uuid
        assert segment.index == 0
        assert segment.offset == 0
        assert segment.block == TextBlock(text="hello world")

        # One derivative record in vector store, carrying every filterable
        # system value under its reserved key and nothing else.
        assert len(fake_vector_store_partition.records) == 1
        record = next(iter(fake_vector_store_partition.records.values()))
        props = _record_properties(record)
        assert props == {
            EVENT_TIMESTAMP_KEY: event.timestamp,
            BLOCK_KIND_KEY: "text",
            EVENT_SOURCE_KEY: "src",
        }
        # The event memory store maps the derivative to its segment.
        assert (
            await fake_event_memory_store_partition.get_segment_uuids_by_derivative_uuids(
                [record.uuid]
            )
            == {record.uuid: segment.uuid}
        )

    async def test_source_is_recorded(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
    ):
        event = _make_event("hi", source_id="alice")
        await event_memory_partition.encode_events([event])

        record = next(iter(fake_vector_store_partition.records.values()))
        assert _record_properties(record)[EVENT_SOURCE_KEY] == "alice"
        segment = next(iter(fake_event_memory_store_partition.segments.values()))
        assert segment.source_id == "alice"

    async def test_context_is_not_a_property(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
    ):
        event = _make_event("hi", context=_author("Alice"))
        await event_memory_partition.encode_events([event])

        record = next(iter(fake_vector_store_partition.records.values()))
        props = _record_properties(record)
        assert not any("author" in key or "context" in key for key in props)

    async def test_long_text_chunking(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        long_text = "word " * 1000  # ~5000 chars
        event = _make_event(long_text.strip())
        await event_memory_partition.encode_events([event])

        segments = list(fake_event_memory_store_partition.segments.values())
        assert len(segments) > 1

        # All segments share the same event_uuid and have incrementing offsets.
        offsets = sorted(s.offset for s in segments)
        assert offsets == list(range(len(segments)))
        for segment in segments:
            assert segment.event_uuid == event.uuid
            assert isinstance(segment.block, TextBlock)
            assert len(segment.block.text) <= 2000

    async def test_multiple_content_items(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        event = Event(
            source_id="src",
            uuid=uuid4(),
            timestamp=_T0,
            blocks=[TextBlock(text="first"), TextBlock(text="second")],
        )
        await event_memory_partition.encode_events([event])

        segments = sorted(
            fake_event_memory_store_partition.segments.values(),
            key=lambda s: s.index,
        )
        assert len(segments) == 2
        assert segments[0].index == 0
        assert segments[1].index == 1
        assert segments[0].offset == 0
        assert segments[1].offset == 0

    async def test_undeclared_properties_stay_on_the_segment(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        """An undeclared property is the event memory store's; the vector record carries the declared keys."""
        event = _make_event(
            "hi", properties={"color": "red", "_episode_uid": "episode-1"}
        )
        await event_memory_partition.encode_events([event])

        record = next(iter(fake_vector_store_partition.records.values()))
        props = _record_properties(record)
        assert "color" not in props
        assert props["_episode_uid"] == "episode-1"
        segment = next(iter(fake_event_memory_store_partition.segments.values()))
        assert segment.properties["color"] == "red"

    async def test_reserved_property_key_is_rejected(
        self, event_memory_partition: EventMemoryPartition
    ):
        event = _make_event("hi", properties={EVENT_SOURCE_KEY: "spoofed"})
        with pytest.raises(ValueError, match="reserved"):
            await event_memory_partition.encode_events([event])

    async def test_illegal_property_key_is_rejected(
        self, event_memory_partition: EventMemoryPartition
    ):
        event = _make_event("hi", properties={"Color": "red"})
        with pytest.raises(ValueError, match=r"\[a-z0-9_\]"):
            await event_memory_partition.encode_events([event])

    async def test_init_raises_on_missing_reserved_field(self, fake_embedder):
        schema = EventMemory.expected_vector_store_collection_schema()
        del schema[EVENT_SOURCE_KEY]
        vector_store_partition = InMemoryVectorStorePartition(indexed_properties=schema)
        with pytest.raises(
            ValueError,
            match="Vector store schema missing fields required by EventMemory",
        ):
            await open_event_memory_partition(
                vector_store_partition=vector_store_partition,
                event_memory_store_partition=InMemoryEventMemoryStorePartition(),
                segmenter=Segmenter([TextSegmenter()]),
                embedder=fake_embedder,
                deriver=Deriver([WholeTextDeriver()]),
            )

    async def test_empty_events(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
    ):
        await event_memory_partition.encode_events([])
        assert len(fake_event_memory_store_partition.segments) == 0
        assert len(fake_vector_store_partition.records) == 0

    async def test_derive_sentences(
        self,
        event_memory_partition_with_sentences: EventMemoryPartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        event = _make_event("Hello there. How are you? I am fine.")
        await event_memory_partition_with_sentences.encode_events([event])

        # One segment, but multiple derivatives (one per sentence).
        assert len(fake_event_memory_store_partition.segments) == 1
        assert len(fake_vector_store_partition.records) > 1


# ===================================================================
# query
# ===================================================================


@_async
class TestQuery:
    async def test_basic_query(self, event_memory_partition: EventMemoryPartition):
        e1 = _make_event("short", timestamp=_ts(0))
        e2 = _make_event("a longer sentence here", timestamp=_ts(1))
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query("test query")
        assert len(hits) == 2
        for hit in hits:
            assert isinstance(hit, QueryHit)
            assert hit.seed.uuid in {s.uuid for s in hit.window()}

    async def test_limit_is_a_maximum(
        self, event_memory_partition: EventMemoryPartition
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory_partition.encode_events(events)

        assert (
            len(await event_memory_partition.query("test", vector_search_limit=2)) <= 2
        )

    async def test_expand_context_marks_the_seed(
        self, event_memory_partition: EventMemoryPartition
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory_partition.encode_events(events)

        hits = await event_memory_partition.query(
            "test query", vector_search_limit=1, expand_context=6
        )

        # With expand_context=6: before=2, after=4.
        [hit] = hits
        assert len(hit.window()) > 1
        assert len({seg.event_uuid for seg in hit.window()}) > 1
        assert hit.seed.uuid == hit.seed.uuid
        timestamps = [seg.timestamp for seg in hit.window()]
        assert timestamps == sorted(timestamps)

    async def test_empty_memory(self, event_memory_partition: EventMemoryPartition):
        assert await event_memory_partition.query("anything") == []

    async def test_a_score_is_the_partitions_score(
        self, event_memory_partition: EventMemoryPartition
    ):
        await event_memory_partition.encode_events([_make_event("hello")])
        [hit] = await event_memory_partition.query("hello")
        # FakeEmbedder: all vectors same direction, and the fixture's
        # partition scores by cosine, so the score is 1.0.
        assert hit.score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("similarity_metric", list(SimilarityMetric))
    async def test_hits_come_best_first_and_a_threshold_keeps_the_better_ones(
        self, similarity_metric: SimilarityMetric
    ):
        # Angles from the query: "near" 0.1 rad, "far" 1.2 rad, the query 0.
        embedder = AngleEmbedder({"near": 0.1, "far": 1.2, "query": 0.0})
        memory = await _build(
            embedder, vector_store_partition=make_partition(similarity_metric)
        )
        far = _make_event("far", timestamp=_ts(0))
        near = _make_event("near", timestamp=_ts(1))
        await memory.encode_events([far, near])

        ranked = await memory.query("query")
        # Halfway between the two scores, whichever way the metric points.
        threshold = (ranked[0].score + ranked[1].score) / 2
        thresholded = await memory.query("query", score_threshold=threshold)

        assert [hit.seed.event_uuid for hit in ranked] == [
            near.uuid,
            far.uuid,
        ]
        assert [hit.seed.event_uuid for hit in thresholded] == [near.uuid]

    async def test_a_neighbor_in_the_window_is_not_a_hit(self):
        embedder = AngleEmbedder({"match": 0.0, "other": 1.4, "query": 0.0})
        memory = await _build(embedder)
        other = _make_event("other", timestamp=_ts(0))
        match = _make_event("match", timestamp=_ts(1))
        await memory.encode_events([other, match])

        hits = await memory.query("query", vector_search_limit=1, expand_context=3)

        [hit] = hits
        assert hit.seed.event_uuid == match.uuid
        assert {seg.event_uuid for seg in hit.window()} == {other.uuid, match.uuid}


@_async
class TestQuerySystemFilters:
    async def test_source_ids_select_events(
        self, event_memory_partition: EventMemoryPartition
    ):
        alice = _make_event("alice says", timestamp=_ts(0), source_id="alice")
        bob = _make_event("bob says", timestamp=_ts(1), source_id="bob")
        await event_memory_partition.encode_events([alice, bob])

        hits = await event_memory_partition.query(
            "says", source_ids=["bob"], expand_context=6
        )

        assert _texts(hits) == {"bob says"}

    async def test_since_and_until_bound_hits_and_windows(
        self, event_memory_partition: EventMemoryPartition
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory_partition.encode_events(events)

        hits = await event_memory_partition.query(
            "event", since=_ts(1), until=_ts(3), expand_context=6
        )

        assert _texts(hits) == {"event 1", "event 2"}

    async def test_block_kinds_select_segments(
        self, event_memory_partition: EventMemoryPartition
    ):
        await event_memory_partition.encode_events([_make_event("hi")])

        assert len(await event_memory_partition.query("hi", block_kinds=["text"])) == 1
        assert await event_memory_partition.query("hi", block_kinds=["image"]) == []

    async def test_timestamp_filter_field_is_the_event_memory_store_column(
        self, event_memory_partition: EventMemoryPartition
    ):
        """A caller's bare `timestamp` names the event timestamp at the event memory store."""
        early = _make_event("early", timestamp=_ts(0))
        late = _make_event("late", timestamp=_ts(10))
        await event_memory_partition.encode_events([early, late])

        hits = await event_memory_partition.query(
            "x",
            property_filter=Comparison(field="timestamp", op=">=", value=_ts(5)),
            expand_context=6,
        )

        assert _texts(hits) == {"late"}

    async def test_an_undeclared_property_stays_off_the_vector_stage(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_vector_store_partition,
        monkeypatch,
    ):
        """The vector stage gets the typed filters; a filter on an undeclared property is the store's."""
        red = _make_event("red", timestamp=_ts(0), properties={"color": "red"})
        blue = _make_event("blue", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory_partition.encode_events([red, blue])
        seen: list[object] = []
        original_query = fake_vector_store_partition.query

        async def recording_query(**kwargs):
            seen.append(kwargs.get("property_filter"))
            return await original_query(**kwargs)

        monkeypatch.setattr(fake_vector_store_partition, "query", recording_query)

        hits = await event_memory_partition.query(
            "x",
            source_ids=["src"],
            property_filter=Comparison(field="m.color", op="=", value="blue"),
        )

        assert _texts(hits) == {"blue"}
        assert seen == [In(field=EVENT_SOURCE_KEY, values=["src"])]


# ===================================================================
# expand
# ===================================================================


@_async
class TestExpand:
    async def test_expand_around_a_segment(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory_partition.encode_events(events)
        [seed_uuid] = fake_event_memory_store_partition.event_to_segments[
            events[2].uuid
        ]

        neighborhood = await event_memory_partition.expand(seed_uuid, before=1, after=2)

        assert [s.event_uuid for s in neighborhood.before] == [events[1].uuid]
        assert [s.event_uuid for s in neighborhood.after] == [
            events[3].uuid,
            events[4].uuid,
        ]
        assert seed_uuid not in {
            s.uuid for s in neighborhood.before + neighborhood.after
        }

    async def test_expand_filters_neighbors_but_not_the_seed(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        red = _make_event("red", timestamp=_ts(0), properties={"color": "red"})
        blue = _make_event("blue", timestamp=_ts(1), properties={"color": "blue"})
        green = _make_event("green", timestamp=_ts(2), properties={"color": "green"})
        await event_memory_partition.encode_events([red, blue, green])
        [seed_uuid] = fake_event_memory_store_partition.event_to_segments[blue.uuid]

        neighborhood = await event_memory_partition.expand(
            seed_uuid,
            before=5,
            after=5,
            property_filter=Comparison(field="m.color", op="=", value="green"),
        )

        assert neighborhood.before == []
        assert [s.event_uuid for s in neighborhood.after] == [green.uuid]

    async def test_negative_counts_are_rejected(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        event = _make_event("x", timestamp=_ts(0))
        await event_memory_partition.encode_events([event])
        [seed_uuid] = fake_event_memory_store_partition.event_to_segments[event.uuid]

        with pytest.raises(ValueError, match="before must be nonnegative"):
            await event_memory_partition.expand(seed_uuid, before=-1)
        with pytest.raises(ValueError, match="expand_context must be nonnegative"):
            await event_memory_partition.query("x", expand_context=-1)

    async def test_expand_walks_further_from_an_edge(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory_partition.encode_events(events)

        [seed_uuid] = fake_event_memory_store_partition.event_to_segments[
            events[0].uuid
        ]
        first = await event_memory_partition.expand(seed_uuid, after=2)
        second = await event_memory_partition.expand(first.after[-1].uuid, after=2)

        assert [s.event_uuid for s in first.after] == [events[1].uuid, events[2].uuid]
        assert [s.event_uuid for s in second.after] == [events[3].uuid, events[4].uuid]

    async def test_unknown_seed_raises(
        self, event_memory_partition: EventMemoryPartition
    ):
        await event_memory_partition.encode_events([_make_event("hi")])
        with pytest.raises(LookupError):
            await event_memory_partition.expand(uuid4(), before=1, after=1)


# ===================================================================
# forget_events
# ===================================================================


@_async
class TestForgetEvents:
    async def test_forget_basic(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
    ):
        e1 = _make_event("keep me", timestamp=_ts(0))
        e2 = _make_event("forget me", timestamp=_ts(1))
        await event_memory_partition.encode_events([e1, e2])

        assert len(fake_event_memory_store_partition.segments) == 2
        assert len(fake_vector_store_partition.records) == 2

        await event_memory_partition.forget_events([e2.uuid])

        # Only e1's data remains.
        assert len(fake_event_memory_store_partition.segments) == 1
        remaining_segment = next(
            iter(fake_event_memory_store_partition.segments.values())
        )
        assert remaining_segment.event_uuid == e1.uuid
        assert len(fake_vector_store_partition.records) == 1

    async def test_forget_empty_set(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        await event_memory_partition.encode_events([_make_event("keep me")])
        await event_memory_partition.forget_events([])
        assert len(fake_event_memory_store_partition.segments) == 1

    async def test_forget_nonexistent(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        await event_memory_partition.encode_events([_make_event("keep me")])
        await event_memory_partition.forget_events([uuid4()])
        assert len(fake_event_memory_store_partition.segments) == 1


# ===================================================================
# render (static, sync)
# ===================================================================


def _make_segment(
    *,
    event_uuid=None,
    index: int = 0,
    offset: int = 0,
    timestamp: datetime.datetime = _T0,
    text: str = "text",
    context: Context | None = None,
) -> Segment:
    return Segment(
        source_id="src",
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=timestamp,
        block=TextBlock(text=text),
        context=context if context is not None else Context(),
    )


class TestRender:
    def test_no_context(self):
        segment = _make_segment(text="hello world")
        result = EventMemory.render_segments([segment], datetime_format=_SHORT_TIME)
        assert json.dumps("hello world") in result
        assert "[" in result  # Timestamp bracket.

    def test_author_renders_its_name(self):
        segment = _make_segment(text="hi", context=_author("Alice"))
        result = EventMemory.render_segments([segment], datetime_format=_SHORT_TIME)
        assert "Alice:" in result
        assert json.dumps("hi") in result

    def test_adjacent_pieces_share_a_header(self):
        event_uuid = uuid4()
        s1 = _make_segment(event_uuid=event_uuid, index=0, offset=0, text="part1")
        s2 = _make_segment(event_uuid=event_uuid, index=0, offset=1, text="part2")
        s3 = _make_segment(event_uuid=event_uuid, index=1, offset=0, text="part3")
        result = EventMemory.render_segments([s1, s2, s3], datetime_format=_SHORT_TIME)
        # Text content is accumulated into one JSON string.
        assert json.dumps("part1part2part3") in result
        # Only one timestamp line.
        assert result.count("[") == 1

    def test_a_missing_piece_starts_a_new_header(self):
        event_uuid = uuid4()
        first = _make_segment(event_uuid=event_uuid, index=0, offset=0, text="A")
        third = _make_segment(event_uuid=event_uuid, index=2, offset=0, text="C")
        result = EventMemory.render_segments(
            [first, third], datetime_format=_SHORT_TIME
        )
        assert result.count("[") == 2
        assert json.dumps("AC") not in result

    def test_no_timestamp_when_both_styles_are_off(self):
        segment = _make_segment(text="hi", context=_author("Alice"))
        result = EventMemory.render_segments(
            [segment], datetime_format=DateTimeFormat(date_style=None, time_style=None)
        )
        assert result == 'Alice: "hi"'

    def test_empty_list(self):
        assert EventMemory.render_segments([], datetime_format=_SHORT_TIME) == ""


# ===================================================================
# rerank (static)
# ===================================================================


@_async
class TestRerank:
    async def test_scores_are_replaced_and_ordered(
        self, event_memory_partition: EventMemoryPartition
    ):
        e1 = _make_event("short", timestamp=_ts(0))
        e2 = _make_event("a much longer text", timestamp=_ts(1))
        await event_memory_partition.encode_events([e1, e2])
        hits = await event_memory_partition.query("anything")

        reranked = await EventMemory.rerank(
            "anything",
            hits,
            reranker=FakeReranker(),
            datetime_format=_SHORT_TIME,
        )

        # FakeReranker scores by rendered length: the longer text first.
        assert [hit.seed.event_uuid for hit in reranked] == [
            e2.uuid,
            e1.uuid,
        ]
        assert reranked[0].score > reranked[1].score > 1.0

    async def test_every_hit_is_returned_in_descending_score(
        self, event_memory_partition: EventMemoryPartition
    ):
        events = [_make_event("x" * (i + 1), timestamp=_ts(i)) for i in range(4)]
        await event_memory_partition.encode_events(events)
        hits = await event_memory_partition.query("anything")

        reranked = await EventMemory.rerank(
            "anything", hits, reranker=FakeReranker(), datetime_format=_SHORT_TIME
        )

        assert len(reranked) == len(hits) == 4
        scores = [hit.score for hit in reranked]
        assert scores == sorted(scores, reverse=True)

    async def test_empty(self):
        assert (
            await EventMemory.rerank(
                "q", [], reranker=FakeReranker(), datetime_format=_SHORT_TIME
            )
            == []
        )


# ===================================================================
# Round-trip tests (encode -> query/forget -> verify via public API)
# ===================================================================


@_async
class TestRoundTrips:
    async def test_encode_then_query_returns_encoded_content(
        self, event_memory_partition: EventMemoryPartition
    ):
        """Encoded events should be retrievable through query."""
        e1 = _make_event(
            "The quick brown fox",
            context=_author("Alice"),
            timestamp=_ts(0),
        )
        e2 = _make_event(
            "jumps over the lazy dog",
            context=_author("Bob"),
            timestamp=_ts(1),
        )
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query("test query")
        assert len(hits) == 2
        assert _texts(hits) == {"The quick brown fox", "jumps over the lazy dog"}

    async def test_encode_then_query_preserves_context(
        self, event_memory_partition: EventMemoryPartition
    ):
        """Query results should carry the original context."""
        event = _make_event("hello", context=_author("Alice"), timestamp=_ts(0))
        await event_memory_partition.encode_events([event])

        [hit] = await event_memory_partition.query("test")
        assert hit.seed.context.get("author") == Author(name="Alice")

    async def test_forget_then_query_excludes_forgotten(
        self, event_memory_partition: EventMemoryPartition
    ):
        """Forgotten events must not appear in query results."""
        e1 = _make_event("keep this one", timestamp=_ts(0))
        e2 = _make_event("forget this one", timestamp=_ts(1))
        await event_memory_partition.encode_events([e1, e2])

        await event_memory_partition.forget_events([e2.uuid])

        texts = _texts(await event_memory_partition.query("test query"))
        assert "keep this one" in texts
        assert "forget this one" not in texts

    async def test_forget_all_then_query_returns_empty(
        self, event_memory_partition: EventMemoryPartition
    ):
        e1 = _make_event("first", timestamp=_ts(0))
        e2 = _make_event("second", timestamp=_ts(1))
        await event_memory_partition.encode_events([e1, e2])

        await event_memory_partition.forget_events([e1.uuid, e2.uuid])

        assert await event_memory_partition.query("test") == []

    async def test_multiple_encode_calls_are_additive(
        self, event_memory_partition: EventMemoryPartition
    ):
        e1 = _make_event("batch one", timestamp=_ts(0))
        e2 = _make_event("batch two", timestamp=_ts(1))

        await event_memory_partition.encode_events([e1])
        await event_memory_partition.encode_events([e2])

        assert _texts(await event_memory_partition.query("test query")) == {
            "batch one",
            "batch two",
        }

    async def test_query_result_rendered_as_string(
        self, event_memory_partition: EventMemoryPartition
    ):
        """End-to-end: encode, query, render."""
        event = _make_event(
            "The mitochondria is the powerhouse of the cell.",
            context=_author("textbook"),
            timestamp=_ts(0),
        )
        await event_memory_partition.encode_events([event])

        [hit] = await event_memory_partition.query("biology")
        context_string = EventMemory.render_segments(
            hit.window(), datetime_format=_SHORT_TIME
        )

        assert "textbook:" in context_string
        assert "The mitochondria is the powerhouse of the cell." in context_string


# ===================================================================
# Filtering round-trip tests
# ===================================================================


@_async
class TestQueryWithFilter:
    async def test_an_undeclared_property_conjunct_reaches_only_the_event_memory_store(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The vector store gets the conjuncts on declared fields; the event memory store gets the whole filter."""
        seen = _record_query_filters(fake_vector_store_partition, monkeypatch)
        e1 = _make_event("red thing", properties={"color": "red", "_episode_uid": "e1"})
        e2 = _make_event(
            "blue thing", properties={"color": "blue", "_episode_uid": "e2"}
        )
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=And(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Comparison(field="episode_uid", op="=", value="e1"),
            ),
        )

        assert seen == [Comparison(field="_episode_uid", op="=", value="e1")]
        assert _texts(hits) == {"red thing"}

    async def test_an_undeclared_property_under_or_leaves_the_vector_search_unfiltered(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An Or with an arm on undeclared fields alone leaves nothing for the vector store."""
        seen = _record_query_filters(fake_vector_store_partition, monkeypatch)
        await event_memory_partition.encode_events(
            [_make_event("thing", timestamp=_ts(0))]
        )

        await event_memory_partition.query(
            "thing",
            property_filter=Or(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Comparison(field="episode_uid", op="=", value="e1"),
            ),
        )

        assert seen == [None]

    @pytest.mark.parametrize(
        ("property_filter", "vector_store_filter"),
        [
            pytest.param(
                Or(
                    left=And(
                        left=Comparison(field="m.color", op="=", value="red"),
                        right=Comparison(field="episode_uid", op="=", value="e1"),
                    ),
                    right=And(
                        left=Comparison(field="m.color", op="=", value="blue"),
                        right=Comparison(field="episode_uid", op="=", value="e2"),
                    ),
                ),
                Or(
                    left=Comparison(field="_episode_uid", op="=", value="e1"),
                    right=Comparison(field="_episode_uid", op="=", value="e2"),
                ),
                id="each arm of an or",
            ),
            pytest.param(
                Not(
                    expr=Or(
                        left=Comparison(field="m.color", op="=", value="red"),
                        right=Comparison(field="episode_uid", op="=", value="e1"),
                    )
                ),
                Not(expr=Comparison(field="_episode_uid", op="=", value="e1")),
                id="a negated or",
            ),
            pytest.param(
                Not(
                    expr=And(
                        left=Comparison(field="m.color", op="=", value="red"),
                        right=Comparison(field="episode_uid", op="=", value="e1"),
                    )
                ),
                None,
                id="a negated and",
            ),
            pytest.param(
                Not(
                    expr=Or(
                        left=Comparison(field="m.color", op="=", value="red"),
                        right=Not(
                            expr=Comparison(field="episode_uid", op="=", value="e1")
                        ),
                    )
                ),
                Not(
                    expr=Not(expr=Comparison(field="_episode_uid", op="=", value="e1"))
                ),
                id="a negation under a negated or",
            ),
        ],
    )
    async def test_the_vector_store_gets_the_part_of_the_filter_on_declared_fields(
        self,
        event_memory_partition: EventMemoryPartition,
        fake_vector_store_partition: InMemoryVectorStorePartition,
        monkeypatch: pytest.MonkeyPatch,
        property_filter: FilterExpr,
        vector_store_filter: FilterExpr | None,
    ):
        """Every record matching the filter matches the part the vector store gets."""
        seen = _record_query_filters(fake_vector_store_partition, monkeypatch)
        await event_memory_partition.encode_events(
            [_make_event("thing", timestamp=_ts(0))]
        )

        await event_memory_partition.query("thing", property_filter=property_filter)

        assert seen == [vector_store_filter]

    async def test_a_declared_user_property_reaches_the_vector_store(
        self,
        fake_embedder: FakeEmbedder,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A user key the partition declares is on the record, and its condition reaches the search under its bare name."""
        collection = InMemoryVectorStorePartition(
            similarity_metric=fake_embedder.similarity_metric,
            indexed_properties={
                **EventMemory.expected_vector_store_collection_schema(),
                "color": str,
            },
        )
        event_memory_partition = await open_event_memory_partition(
            event_memory_store_partition=fake_event_memory_store_partition,
            vector_store_partition=collection,
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([WholeTextDeriver()]),
            embedder=fake_embedder,
        )
        seen = _record_query_filters(collection, monkeypatch)
        await event_memory_partition.encode_events(
            [_make_event("red thing", properties={"color": "red", "size": 1})]
        )

        await event_memory_partition.query(
            "thing",
            property_filter=And(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Comparison(field="m.size", op="=", value=1),
            ),
        )

        record = next(iter(collection.records.values()))
        assert _record_properties(record)["color"] == "red"
        assert "size" not in _record_properties(record)
        assert seen == [Comparison(field="color", op="=", value="red")]

    async def test_a_seed_the_filter_rejects_is_left_out_though_its_context_matches(
        self,
        event_memory_partition: EventMemoryPartition,
    ):
        await event_memory_partition.encode_events(
            [
                _make_event(
                    "blue thing", timestamp=_ts(0), properties={"color": "blue"}
                ),
                _make_event("red thing", timestamp=_ts(1), properties={"color": "red"}),
            ]
        )

        result = await event_memory_partition.query(
            "thing",
            vector_search_limit=2,
            expand_context=1,
            property_filter=Comparison(field="m.color", op="=", value="red"),
        )

        seed_blocks = [hit.seed.block for hit in result]
        assert seed_blocks == [TextBlock(text="red thing")]

    async def test_equality_filter(self, event_memory_partition: EventMemoryPartition):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=Comparison(field="m.color", op="=", value="red"),
        )
        assert _texts(hits) == {"red thing"}

    async def test_inequality_filter(
        self, event_memory_partition: EventMemoryPartition
    ):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=Comparison(field="m.color", op="!=", value="red"),
        )
        assert _texts(hits) == {"blue thing"}

    async def test_in_filter(self, event_memory_partition: EventMemoryPartition):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        e3 = _make_event("green thing", timestamp=_ts(2), properties={"color": "green"})
        await event_memory_partition.encode_events([e1, e2, e3])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=In(field="m.color", values=["red", "green"]),
        )
        assert _texts(hits) == {"red thing", "green thing"}

    async def test_is_null_filter(self, event_memory_partition: EventMemoryPartition):
        e1 = _make_event("has color", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("no color", timestamp=_ts(1))
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=IsNull(field="m.color"),
        )
        assert _texts(hits) == {"no color"}

    async def test_and_filter(self, event_memory_partition: EventMemoryPartition):
        e1 = _make_event("red small", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue small", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=And(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Not(expr=IsNull(field="m.color")),
            ),
        )
        assert _texts(hits) == {"red small"}

    async def test_or_filter(self, event_memory_partition: EventMemoryPartition):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        e3 = _make_event("green thing", timestamp=_ts(2), properties={"color": "green"})
        await event_memory_partition.encode_events([e1, e2, e3])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=Or(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Comparison(field="m.color", op="=", value="blue"),
            ),
        )
        assert _texts(hits) == {"red thing", "blue thing"}

    async def test_not_filter(self, event_memory_partition: EventMemoryPartition):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory_partition.encode_events([e1, e2])

        hits = await event_memory_partition.query(
            "thing",
            property_filter=Not(expr=Comparison(field="m.color", op="=", value="red")),
        )
        assert _texts(hits) == {"blue thing"}

    async def test_filter_returns_empty_when_nothing_matches(
        self, event_memory_partition: EventMemoryPartition
    ):
        await event_memory_partition.encode_events(
            [_make_event("red thing", timestamp=_ts(0), properties={"color": "red"})]
        )

        hits = await event_memory_partition.query(
            "thing",
            property_filter=Comparison(field="m.color", op="=", value="purple"),
        )
        assert hits == []

    async def test_context_filter_returns_no_results(
        self, event_memory_partition: EventMemoryPartition
    ):
        """Context is never filterable."""
        await event_memory_partition.encode_events(
            [_make_event("hi", context=_author("Alice"))]
        )

        hits = await event_memory_partition.query(
            "hi",
            property_filter=Comparison(field="context.author", op="=", value="Alice"),
        )
        assert hits == []


# ===================================================================
# Queries with and without the vector store's part of the filter
# ===================================================================

_FIELDS_ON_THE_RECORD = ("d1", "d2", "d3")
_FIELDS_ON_THE_SEGMENT_ONLY = ("m.u1", "m.u2")


@pytest_asyncio.fixture
async def declaring_event_memory(
    fake_embedder, fake_event_memory_store_partition
) -> EventMemoryPartition:
    """Event memory whose vector store declares `_d1`, `_d2`, and `_d3` but not `u1` or `u2`."""
    collection = InMemoryVectorStorePartition(
        similarity_metric=fake_embedder.similarity_metric,
        indexed_properties={
            **EventMemory.expected_vector_store_collection_schema(),
            "_d1": int,
            "_d2": int,
            "_d3": int,
        },
    )
    return await open_event_memory_partition(
        event_memory_store_partition=fake_event_memory_store_partition,
        vector_store_partition=collection,
        segmenter=Segmenter([TextSegmenter()]),
        deriver=Deriver([WholeTextDeriver()]),
        embedder=fake_embedder,
    )


async def _encode_random_events(
    event_memory_partition: EventMemoryPartition, rng: random.Random, count: int
) -> dict[str, dict[str, PropertyValue]]:
    """Encode `count` events with random properties; their properties by text."""
    properties_by_text = {
        f"event {index}": random_properties(rng, ("_d1", "_d2", "_d3", "u1", "u2"))
        for index in range(count)
    }
    await event_memory_partition.encode_events(
        [
            _make_event(text, timestamp=_ts(index), properties=properties)
            for index, (text, properties) in enumerate(properties_by_text.items())
        ]
    )
    return properties_by_text


async def _query_texts(
    event_memory_partition: EventMemoryPartition,
    property_filter: FilterExpr,
    limit: int,
    query: str = "event",
) -> set[str]:
    result = await event_memory_partition.query(
        query, vector_search_limit=limit, property_filter=property_filter
    )
    return {
        segment.block.text
        for hit in result
        for segment in hit.window()
        if isinstance(segment.block, TextBlock)
    }


def _without_the_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the vector store no filter at all."""
    monkeypatch.setattr(
        "memmachine_server.episodic_memory.event_memory.event_memory.extract_filter",
        lambda expr, *, allowed_fields: None,
    )


@_async
class TestQueryWithAndWithoutThePart:
    async def test_both_answer_the_filter_when_the_search_reaches_every_record(
        self,
        declaring_event_memory: EventMemoryPartition,
        monkeypatch: pytest.MonkeyPatch,
    ):
        rng = random.Random(1705)
        properties_by_text = await _encode_random_events(
            declaring_event_memory, rng, 24
        )
        fields = (*_FIELDS_ON_THE_RECORD, *_FIELDS_ON_THE_SEGMENT_ONLY)
        for _ in range(300):
            property_filter = random_filter(rng, fields, depth=4)
            with_part = await _query_texts(
                declaring_event_memory, property_filter, limit=24
            )
            with monkeypatch.context() as patch:
                _without_the_part(patch)
                without_part = await _query_texts(
                    declaring_event_memory, property_filter, limit=24
                )
            segment_filter = map_filter_fields(
                property_filter,
                InMemoryEventMemoryStorePartition._normalize_segment_field,
            )
            expected = {
                text
                for text, properties in properties_by_text.items()
                if evaluate_filter(segment_filter, properties)
            }
            assert with_part == without_part == expected, property_filter

    async def test_the_part_only_adds_hits_when_the_search_reaches_fewer_records(
        self,
        declaring_event_memory: EventMemoryPartition,
        monkeypatch: pytest.MonkeyPatch,
    ):
        rng = random.Random(1706)
        await _encode_random_events(declaring_event_memory, rng, 24)
        fields = (*_FIELDS_ON_THE_RECORD, *_FIELDS_ON_THE_SEGMENT_ONLY)
        for _ in range(300):
            property_filter = random_filter(rng, fields, depth=4)
            with_part = await _query_texts(
                declaring_event_memory, property_filter, limit=4
            )
            with monkeypatch.context() as patch:
                _without_the_part(patch)
                without_part = await _query_texts(
                    declaring_event_memory, property_filter, limit=4
                )
            assert with_part >= without_part, property_filter


class _ColorEmbedder(Embedder):
    """Embeds a text by how often it says blue and red, so the two rank apart."""

    async def _ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        return [self._embed(str(text)) for text in inputs]

    async def _search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        return [self._embed(str(query)) for query in queries]

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [float(text.count("blue")), float(text.count("red"))]

    @property
    def model_id(self) -> str:
        return "color-model"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


@_async
class TestQueryWithAConditionOnAnUndeclaredField:
    async def test_the_condition_applies_to_the_seeds_the_search_returns(
        self,
        fake_vector_store_partition: InMemoryVectorStorePartition,
        fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    ):
        """A seed the search ranks below the limit is not found, unless the condition is on a declared field."""
        event_memory_partition = await open_event_memory_partition(
            event_memory_store_partition=fake_event_memory_store_partition,
            vector_store_partition=fake_vector_store_partition,
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([WholeTextDeriver()]),
            embedder=_ColorEmbedder(),
        )
        await event_memory_partition.encode_events(
            [
                *(
                    _make_event(
                        f"blue thing {index}",
                        timestamp=_ts(index),
                        properties={"color": "blue", "_episode_uid": f"blue-{index}"},
                    )
                    for index in range(10)
                ),
                _make_event(
                    "red thing",
                    timestamp=_ts(10),
                    properties={"color": "red", "_episode_uid": "red"},
                ),
            ]
        )

        on_the_segment_only = await _query_texts(
            event_memory_partition,
            Comparison(field="m.color", op="=", value="red"),
            limit=3,
            query="blue thing",
        )
        on_the_segment_only_with_every_seed = await _query_texts(
            event_memory_partition,
            Comparison(field="m.color", op="=", value="red"),
            limit=11,
            query="blue thing",
        )
        on_the_record = await _query_texts(
            event_memory_partition,
            Comparison(field="episode_uid", op="=", value="red"),
            limit=3,
            query="blue thing",
        )

        assert on_the_segment_only == set()
        assert on_the_segment_only_with_every_seed == {"red thing"}
        assert on_the_record == {"red thing"}


# ===================================================================
# Query deduplication
# ===================================================================


@_async
class TestQueryDeduplication:
    async def test_multiple_derivatives_deduplicate_to_one_segment(
        self,
        event_memory_partition_with_sentences: EventMemoryPartition,
    ):
        event = _make_event(
            "First sentence. Second sentence. Third sentence.",
            timestamp=_ts(0),
        )
        await event_memory_partition_with_sentences.encode_events([event])

        hits = await event_memory_partition_with_sentences.query("sentence")

        # All derivatives map to the same segment, so deduplication
        # should collapse them into a single hit.
        assert len(hits) == 1
        assert len(hits[0].window()) == 1

    async def test_derivatives_from_different_segments_not_collapsed(
        self,
        event_memory_partition_with_sentences: EventMemoryPartition,
    ):
        e1 = _make_event("Alpha sentence. Beta sentence.", timestamp=_ts(0))
        e2 = _make_event("Gamma sentence. Delta sentence.", timestamp=_ts(1))
        await event_memory_partition_with_sentences.encode_events([e1, e2])

        assert len(await event_memory_partition_with_sentences.query("sentence")) == 2

    async def test_dedup_uses_best_derivative_score(self):
        """When multiple derivatives map to one segment, the best score wins."""
        embedder = AngleEmbedder({"Close": 0.1, "Distant": 1.0, "query": 0.0})
        memory = await open_event_memory_partition(
            event_memory_store_partition=InMemoryEventMemoryStorePartition(),
            vector_store_partition=make_partition(),
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([SentenceTextDeriver()]),
            embedder=embedder,
        )
        await memory.encode_events(
            [_make_event("Distant sentence. Close sentence.", timestamp=_ts(0))]
        )

        [hit] = await memory.query("query")

        assert hit.score == pytest.approx(math.cos(0.1))


# ===================================================================
# The deriver's format options (timestamp baked into the embedding)
# ===================================================================


class _RecordingEmbedder(FakeEmbedder):
    """FakeEmbedder that records the texts passed to ingest_embed."""

    def __init__(self):
        super().__init__()
        self.ingested: list[str] = []

    async def _ingest_embed(self, inputs, max_attempts=1):
        self.ingested.extend(inputs)
        return await super()._ingest_embed(inputs, max_attempts=max_attempts)


@_async
class TestDeriverDateTimeFormat:
    async def test_default_bakes_full_date_into_embedding(self):
        embedder = _RecordingEmbedder()
        event_memory_partition = await _build(embedder)

        await event_memory_partition.encode_events([_make_event("hello world")])

        # _T0 is 2025-06-01; the ingest default is a full date with no time,
        # and the message text is JSON-dumped (ensure_ascii=False).
        assert embedder.ingested == ['[Sunday, June 1, 2025] "hello world"']

    async def test_the_deriver_decides_the_format(self):
        embedder = _RecordingEmbedder()
        event_memory_partition = await _build(
            embedder,
            deriver=Deriver(
                [WholeTextDeriver(DateTimeFormat(date_style=None, time_style=None))]
            ),
        )

        await event_memory_partition.encode_events([_make_event("hello world")])

        assert embedder.ingested == ['"hello world"']

    async def test_author_is_embedded_by_name(self):
        embedder = _RecordingEmbedder()
        event_memory_partition = await _build(
            embedder,
            deriver=Deriver(
                [WholeTextDeriver(DateTimeFormat(date_style=None, time_style=None))]
            ),
        )

        await event_memory_partition.encode_events(
            [_make_event("hello", context=_author("Alice"), source_id="u-1")]
        )

        assert embedder.ingested == ['Alice: "hello"']

    async def test_segment_text_stays_structured(self):
        """The baked timestamp lives only in the embedding, not the segment."""
        embedder = _RecordingEmbedder()
        partition = InMemoryEventMemoryStorePartition()
        event_memory_partition = await _build(embedder, partition=partition)

        await event_memory_partition.encode_events([_make_event("hello world")])

        (segment,) = partition.segments.values()
        assert segment.block == TextBlock(text="hello world")


# ===================================================================
# system predicates (typed filters as a filter tree)
# ===================================================================

_T1 = _T0 + datetime.timedelta(days=1)


def _conjuncts(expr):
    if isinstance(expr, And):
        return [*_conjuncts(expr.left), *_conjuncts(expr.right)]
    return [expr]


def test_no_system_filter_is_no_tree():
    assert _system_predicates() is None


def test_predicates_name_the_reserved_keys():
    tree = _system_predicates(
        since=_T0,
        until=_T1,
        source_ids=["alice", "bob"],
        block_kinds=["text"],
    )
    assert _conjuncts(tree) == [
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T0),
        Comparison(field=EVENT_TIMESTAMP_KEY, op="<", value=_T1),
        In(field=EVENT_SOURCE_KEY, values=["alice", "bob"]),
        In(field=BLOCK_KIND_KEY, values=["text"]),
    ]


def test_empty_ids_admit_nothing_and_none_admits_everything():
    assert _system_predicates(source_ids=None) is None
    assert _system_predicates(source_ids=[]) == In(field=EVENT_SOURCE_KEY, values=[])


# ===================================================================
# eviction
# ===================================================================


async def _stored_minutes(collection: InMemoryVectorStorePartition) -> list[int]:
    """Minute-offsets (from _T0) of every derivative currently in the store."""
    [result] = await collection.query(query_vectors=[[1.0, 0.0]], limit=1000)
    minutes: list[int] = []
    for match in result.matches:
        timestamp = _record_properties(collection.records[match.record_uuid])[
            EVENT_TIMESTAMP_KEY
        ]
        assert isinstance(timestamp, datetime.datetime)
        minutes.append(int((timestamp - _T0).total_seconds() // 60))
    return sorted(minutes)


def _linked(partition: InMemoryEventMemoryStorePartition) -> set[UUID]:
    return {
        uid for linked in partition.segment_to_derivatives.values() for uid in linked
    }


class TestEviction:
    """Eviction caps a near-duplicate cluster at `target_size` by dropping its
    temporally middle members.

    The FakeEmbedder maps every text onto one direction (cosine 1.0), so
    on a cosine partition each batch below forms a single cluster
    regardless of the text.
    """

    @_async
    async def test_disabled_keeps_all_and_issues_no_query(
        self, event_memory_partition, fake_vector_store_partition, monkeypatch
    ):
        queries: list[int] = []
        original_query = fake_vector_store_partition.query

        async def counting_query(**kwargs):
            queries.append(1)
            return await original_query(**kwargs)

        monkeypatch.setattr(fake_vector_store_partition, "query", counting_query)
        events = [_make_event(f"msg {i}", timestamp=_ts(i)) for i in range(20)]

        await event_memory_partition.encode_events(events)

        assert len(await _stored_minutes(fake_vector_store_partition)) == 20
        assert queries == [1]  # the one query _stored_minutes made

    @_async
    async def test_cluster_within_target_keeps_all(
        self, event_memory_partition_with_eviction, fake_vector_store_partition
    ):
        events = [_make_event(f"msg {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory_partition_with_eviction.encode_events(events)
        assert await _stored_minutes(fake_vector_store_partition) == [0, 1, 2, 3, 4]

    @_async
    async def test_intra_batch_caps_and_keeps_temporal_extremes(
        self, event_memory_partition_with_eviction, fake_vector_store_partition
    ):
        events = [_make_event(f"msg {i}", timestamp=_ts(i)) for i in range(20)]
        await event_memory_partition_with_eviction.encode_events(events)
        # target=5 -> keep 2 earliest + 3 latest, evict the middle.
        assert await _stored_minutes(fake_vector_store_partition) == [0, 1, 17, 18, 19]

    @_async
    async def test_batch_order_mimics_serial_ingestion(
        self, fake_vector_store_partition
    ):
        """A batch evicts what encoding its events one at a time would."""
        events = [_make_event(f"msg {i}", timestamp=_ts(i)) for i in range(12)]
        eviction = EvictionOptions(score_threshold=0.5, search_limit=100, target_size=5)
        batched_collection = make_partition()
        serial_collection = make_partition()
        batched = await _build(
            FakeEmbedder(), vector_store_partition=batched_collection, eviction=eviction
        )
        serial = await _build(
            FakeEmbedder(), vector_store_partition=serial_collection, eviction=eviction
        )

        await batched.encode_events(list(reversed(events)))
        for event in events:
            await serial.encode_events([event])

        assert await _stored_minutes(batched_collection) == await _stored_minutes(
            serial_collection
        )

    @_async
    async def test_clusters_are_trimmed_independently(self):
        embedder = AngleEmbedder({"alpha": 0.0, "beta": math.pi / 2})
        collection = make_partition()
        memory = await _build(
            embedder,
            vector_store_partition=collection,
            eviction=EvictionOptions(
                score_threshold=0.5, search_limit=100, target_size=5
            ),
        )
        alphas = [_make_event(f"alpha {i}", timestamp=_ts(i)) for i in range(7)]
        betas = [_make_event(f"beta {i}", timestamp=_ts(10 + i)) for i in range(2)]

        await memory.encode_events([*alphas, *betas])

        assert await _stored_minutes(collection) == [
            0,
            1,
            4,
            5,
            6,
            10,
            11,
        ]

    @_async
    @pytest.mark.parametrize(
        ("similarity_metric", "score_threshold"),
        [
            (SimilarityMetric.COSINE, 0.5),
            (SimilarityMetric.DOT, 0.5),
            (SimilarityMetric.EUCLIDEAN, 1.2),
            (SimilarityMetric.MANHATTAN, 1.5),
        ],
    )
    async def test_a_batch_and_serial_ingestion_trim_alike_under_every_metric(
        self, similarity_metric: SimilarityMetric, score_threshold: float
    ):
        """A batch decides from its own scores, serial ingestion from the store's.

        The two trim the same clusters only when both score by the
        partition's metric. Texts sharing a token score 1.0 by cosine and
        dot product and 0.0 by distance; texts a quarter turn apart score
        0.0 by cosine and dot product, sqrt(2) by Euclidean and 2.0 by
        Manhattan distance. A distance threshold above 1.0 tells the two
        apart only when the batch scores by distance, not cosine.
        """
        embedder = AngleEmbedder({"alpha": 0.0, "beta": math.pi / 2})
        eviction = EvictionOptions(
            score_threshold=score_threshold, search_limit=100, target_size=5
        )
        alphas = [_make_event(f"alpha {i}", timestamp=_ts(i)) for i in range(7)]
        betas = [_make_event(f"beta {i}", timestamp=_ts(10 + i)) for i in range(2)]
        batched_collection = make_partition(similarity_metric)
        serial_collection = make_partition(similarity_metric)
        batched = await _build(
            embedder, vector_store_partition=batched_collection, eviction=eviction
        )
        serial = await _build(
            embedder, vector_store_partition=serial_collection, eviction=eviction
        )

        await batched.encode_events([*alphas, *betas])
        for event in [*alphas, *betas]:
            await serial.encode_events([event])

        expected = [0, 1, 4, 5, 6, 10, 11]
        assert await _stored_minutes(batched_collection) == expected
        assert await _stored_minutes(serial_collection) == expected

    @_async
    async def test_cross_batch_displaces_stored_records_and_keeps_their_links(
        self,
        event_memory_partition_with_eviction,
        fake_vector_store_partition,
        fake_event_memory_store_partition,
    ):
        await event_memory_partition_with_eviction.encode_events(
            [_make_event(f"a{i}", timestamp=_ts(i)) for i in range(5)]
        )
        assert await _stored_minutes(fake_vector_store_partition) == [0, 1, 2, 3, 4]
        stored_before = set(fake_vector_store_partition.records)

        # A second batch grows the cluster past target: stored records from the
        # temporal middle leave the vector store; their links stay.
        await event_memory_partition_with_eviction.encode_events(
            [_make_event(f"b{i}", timestamp=_ts(5 + i)) for i in range(5)]
        )

        assert await _stored_minutes(fake_vector_store_partition) == [0, 1, 7, 8, 9]
        displaced = stored_before - set(fake_vector_store_partition.records)
        assert displaced
        assert _linked(fake_event_memory_store_partition) == (
            set(fake_vector_store_partition.records) | displaced
        )
        # Segments are untouched: every event is still expandable.
        assert len(fake_event_memory_store_partition.segments) == 10

    @_async
    async def test_skipped_derivatives_are_never_linked(
        self, event_memory_partition_with_eviction, fake_event_memory_store_partition
    ):
        events = [_make_event(f"msg {i}", timestamp=_ts(i)) for i in range(20)]
        await event_memory_partition_with_eviction.encode_events(events)

        assert len(_linked(fake_event_memory_store_partition)) == 5
        assert len(fake_event_memory_store_partition.segments) == 20


class _Mood(ContextPart):
    """A part kind a library user would add; contributes its word."""

    kind: ClassVar[str] = "mood"
    word: str

    @override
    def render(self, datetime_format: DateTimeFormat) -> str | None:
        return self.word


@_async
class TestComposition:
    """Timestamp, then the parts `parts` names in that order, then the
    content; parts carry no order of their own."""

    def _segment(self, context) -> Segment:
        return Segment(
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime.datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            context=context,
            block=TextBlock(text="hi"),
        )

    def test_default_order_is_timestamp_then_author_then_content(self):
        segment = self._segment(Context(Author(name="Alice")))
        rendered = EventMemory.render_segments([segment], datetime_format=_SHORT_TIME)
        # Babel puts a narrow no-break space before the meridiem.
        assert rendered.replace("\u202f", " ") == (
            '[Thursday, January 15, 2026, 10:30 AM] Alice: "hi"'
        )

    def test_listed_parts_render_in_the_listed_order(self):
        context = Context(Author(name="Alice"), _Mood(word="happy"))
        no_time = DateTimeFormat(time_style=None)
        assert (
            EventMemory.render_segments(
                [self._segment(context)],
                datetime_format=no_time,
                parts=("mood", "author"),
            )
            == '[Thursday, January 15, 2026] happy: Alice: "hi"'
        )
        assert (
            EventMemory.render_segments(
                [self._segment(context)],
                datetime_format=no_time,
                parts=("author", "mood"),
            )
            == '[Thursday, January 15, 2026] Alice: happy: "hi"'
        )

    def test_a_part_not_listed_contributes_nothing(self):
        context = Context(Author(name="Alice"), _Mood(word="happy"))
        no_time = DateTimeFormat(time_style=None)
        assert (
            EventMemory.render_segments(
                [self._segment(context)], datetime_format=no_time, parts=("author",)
            )
            == '[Thursday, January 15, 2026] Alice: "hi"'
        )
        assert (
            EventMemory.render_segments(
                [self._segment(context)], datetime_format=no_time, parts=()
            )
            == '[Thursday, January 15, 2026] "hi"'
        )

    def test_parts_must_be_kind_names(self):
        with pytest.raises(ValueError, match="Part kind"):
            EventMemory.render_segments(
                [], datetime_format=_SHORT_TIME, parts=("Not A Kind",)
            )
        with pytest.raises(ValueError, match="Part kind"):
            WholeTextDeriver(parts=("Not A Kind",))

    async def test_the_handler_composes_the_parts_it_names(self):
        context = Context(Author(name="Alice"), _Mood(word="happy"))
        handler = WholeTextDeriver(
            DateTimeFormat(date_style=None, time_style=None), parts=("mood",)
        )
        [derivative] = await Deriver([handler]).derive(self._segment(context))
        assert derivative.text == 'happy: "hi"'

    async def test_the_embedded_text_is_the_same_composition_over_the_derived_text(
        self,
    ):
        segment = self._segment(Context(Author(name="Alice")))
        [derivative] = await Deriver([WholeTextDeriver()]).derive(segment)
        # The handler's default format: a full date and no time.
        assert derivative.text == '[Thursday, January 15, 2026] Alice: "hi"'

    async def test_embedded_text_without_author_or_timestamp(self):
        segment = self._segment(Context())
        bare = WholeTextDeriver(DateTimeFormat(date_style=None, time_style=None))
        [derivative] = await Deriver([bare]).derive(segment)
        assert derivative.text == '"hi"'
