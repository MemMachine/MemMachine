"""Tests for EventMemory."""

import datetime
import json
import math
from datetime import UTC
from typing import Any
from uuid import uuid4

import pytest

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStoreCollectionConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    DateTimeFormat,
    Event,
    NullContext,
    ProducerContext,
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
    EVENT_SESSION_KEY,
    EVENT_SOURCE_KEY,
    EVENT_TIMESTAMP_KEY,
    EventMemory,
    EventMemoryParams,
    _system_predicates,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)

from .conftest import (
    AngleEmbedder,
    FakeReranker,
    InMemorySegmentStorePartition,
    InMemoryVectorStoreCollection,
    make_collection,
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


def _author(name: str) -> Context:
    return ProducerContext(producer=name)


def _make_event(
    text: str,
    *,
    timestamp: datetime.datetime = _T0,
    session_id: str = "s",
    source_id: str = "src",
    context: Context | None = None,
    properties=None,
) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=timestamp,
        session_id=session_id,
        source_id=source_id,
        context=context if context is not None else NullContext(),
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


def _build(
    embedder: FakeEmbedder,
    *,
    partition: InMemorySegmentStorePartition | None = None,
    collection: InMemoryVectorStoreCollection | None = None,
    deriver: Deriver | None = None,
) -> EventMemory:
    params: dict[str, Any] = {
        "segment_store_partition": partition or InMemorySegmentStorePartition(),
        "vector_store_collection": collection or make_collection(embedder),
        "segmenter": TextSegmenter(),
        "deriver": deriver or WholeTextDeriver(),
        "embedder": embedder,
    }
    return EventMemory(EventMemoryParams(**params))


# ===================================================================
# schema
# ===================================================================


class TestSchema:
    def test_expected_vector_store_collection_schema_declares_the_reserved_keys(self):
        assert EventMemory.expected_vector_store_collection_schema() == {
            EVENT_TIMESTAMP_KEY: datetime.datetime,
            EVENT_SESSION_KEY: str,
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
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("hello world")
        await event_memory.encode_events([event])

        # One segment stored.
        assert len(fake_segment_store_partition.segments) == 1
        segment = next(iter(fake_segment_store_partition.segments.values()))
        assert segment.event_uuid == event.uuid
        assert segment.index == 0
        assert segment.offset == 0
        assert segment.block == TextBlock(text="hello world")

        # One derivative record in vector store, carrying every filterable
        # system value under its reserved key and nothing else.
        assert len(fake_vector_store_collection.records) == 1
        record = next(iter(fake_vector_store_collection.records.values()))
        props = _record_properties(record)
        assert props == {
            EVENT_TIMESTAMP_KEY: event.timestamp,
            BLOCK_KIND_KEY: "text",
            EVENT_SESSION_KEY: "s",
            EVENT_SOURCE_KEY: "src",
        }
        # The derivative's segment is not copied here; the segment store owns
        # that mapping and answers it from the derivative's own row.
        assert await fake_segment_store_partition.get_segment_uuids_by_derivative_uuids(
            [record.uuid]
        ) == {record.uuid: segment.uuid}

    async def test_session_and_source_are_recorded(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("hi", session_id="s1", source_id="alice")
        await event_memory.encode_events([event])

        record = next(iter(fake_vector_store_collection.records.values()))
        props = _record_properties(record)
        assert props[EVENT_SESSION_KEY] == "s1"
        assert props[EVENT_SOURCE_KEY] == "alice"
        segment = next(iter(fake_segment_store_partition.segments.values()))
        assert (segment.session_id, segment.source_id) == ("s1", "alice")

    async def test_context_is_not_a_property(
        self,
        event_memory: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("hi", context=_author("Alice"))
        await event_memory.encode_events([event])

        record = next(iter(fake_vector_store_collection.records.values()))
        props = _record_properties(record)
        assert not any("author" in key or "context" in key for key in props)

    async def test_long_text_chunking(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        long_text = "word " * 1000  # ~5000 chars
        event = _make_event(long_text.strip())
        await event_memory.encode_events([event])

        segments = list(fake_segment_store_partition.segments.values())
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
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        event = Event(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            timestamp=_T0,
            blocks=[TextBlock(text="first"), TextBlock(text="second")],
        )
        await event_memory.encode_events([event])

        segments = sorted(
            fake_segment_store_partition.segments.values(),
            key=lambda s: s.index,
        )
        assert len(segments) == 2
        assert segments[0].index == 0
        assert segments[1].index == 1
        assert segments[0].offset == 0
        assert segments[1].offset == 0

    async def test_user_properties_stay_with_the_segment(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("hi", properties={"color": "red"})
        await event_memory.encode_events([event])

        segment = next(iter(fake_segment_store_partition.segments.values()))
        assert segment.properties == {"color": "red"}
        record = next(iter(fake_vector_store_collection.records.values()))
        assert "color" not in _record_properties(record)

    async def test_reserved_property_key_is_rejected(self, event_memory: EventMemory):
        event = _make_event("hi", properties={EVENT_SESSION_KEY: "spoofed"})
        with pytest.raises(ValueError, match="reserved"):
            await event_memory.encode_events([event])

    async def test_illegal_property_key_is_rejected(self, event_memory: EventMemory):
        event = _make_event("hi", properties={"Color": "red"})
        with pytest.raises(ValueError, match=r"\[a-z0-9_\]"):
            await event_memory.encode_events([event])

    async def test_init_raises_on_missing_reserved_field(self, fake_embedder):
        schema = EventMemory.expected_vector_store_collection_schema()
        del schema[EVENT_SESSION_KEY]
        collection = InMemoryVectorStoreCollection(
            VectorStoreCollectionConfig(
                vector_dimensions=2, indexed_properties_schema=schema
            )
        )
        with pytest.raises(
            ValueError,
            match="Collection schema missing fields required by EventMemory",
        ):
            EventMemory(
                EventMemoryParams(
                    vector_store_collection=collection,
                    segment_store_partition=InMemorySegmentStorePartition(),
                    segmenter=TextSegmenter(),
                    embedder=fake_embedder,
                    deriver=WholeTextDeriver(),
                )
            )

    async def test_empty_events(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        await event_memory.encode_events([])
        assert len(fake_segment_store_partition.segments) == 0
        assert len(fake_vector_store_collection.records) == 0

    async def test_derive_sentences(
        self,
        event_memory_with_sentences: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        event = _make_event("Hello there. How are you? I am fine.")
        await event_memory_with_sentences.encode_events([event])

        # One segment, but multiple derivatives (one per sentence).
        assert len(fake_segment_store_partition.segments) == 1
        assert len(fake_vector_store_collection.records) > 1


# ===================================================================
# query
# ===================================================================


@_async
class TestQuery:
    async def test_basic_query(self, event_memory: EventMemory):
        e1 = _make_event("short", timestamp=_ts(0))
        e2 = _make_event("a longer sentence here", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        hits = await event_memory.query("test query")
        assert len(hits) == 2
        for hit in hits:
            assert isinstance(hit, QueryHit)
            assert hit.seed.uuid in {s.uuid for s in hit.window()}

    async def test_limit_is_a_maximum(self, event_memory: EventMemory):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory.encode_events(events)

        assert len(await event_memory.query("test", vector_search_limit=2)) <= 2

    async def test_expand_context_marks_the_seed(self, event_memory: EventMemory):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory.encode_events(events)

        hits = await event_memory.query(
            "test query", vector_search_limit=1, expand_context=6
        )

        # With expand_context=6: before=2, after=4.
        [hit] = hits
        assert len(hit.window()) > 1
        assert len({seg.event_uuid for seg in hit.window()}) > 1
        assert hit.seed.uuid == hit.seed.uuid
        timestamps = [seg.timestamp for seg in hit.window()]
        assert timestamps == sorted(timestamps)

    async def test_empty_memory(self, event_memory: EventMemory):
        assert await event_memory.query("anything") == []

    async def test_scores_are_cosine_similarities(self, event_memory: EventMemory):
        await event_memory.encode_events([_make_event("hello")])
        [hit] = await event_memory.query("hello")
        # FakeEmbedder: all vectors same direction -> cosine 1.0.
        assert hit.score == pytest.approx(1.0, abs=0.01)

    async def test_hits_rank_by_cosine_similarity_and_a_threshold_excludes(self):
        # Angles from the query: "near" 0.1 rad, "far" 1.2 rad, the query 0.
        embedder = AngleEmbedder({"near": 0.1, "far": 1.2, "query": 0.0})
        memory = _build(embedder)
        far = _make_event("far", timestamp=_ts(0))
        near = _make_event("near", timestamp=_ts(1))
        await memory.encode_events([far, near])

        ranked = await memory.query("query")
        thresholded = await memory.query("query", min_cosine_similarity=0.9)

        assert [hit.seed.event_uuid for hit in ranked] == [
            near.uuid,
            far.uuid,
        ]
        assert [hit.score for hit in ranked] == sorted(
            (hit.score for hit in ranked), reverse=True
        )
        assert [hit.seed.event_uuid for hit in thresholded] == [near.uuid]

    async def test_a_neighbor_in_the_window_is_not_a_hit(self):
        embedder = AngleEmbedder({"match": 0.0, "other": 1.4, "query": 0.0})
        memory = _build(embedder)
        other = _make_event("other", timestamp=_ts(0))
        match = _make_event("match", timestamp=_ts(1))
        await memory.encode_events([other, match])

        hits = await memory.query("query", vector_search_limit=1, expand_context=3)

        [hit] = hits
        assert hit.seed.event_uuid == match.uuid
        assert {seg.event_uuid for seg in hit.window()} == {other.uuid, match.uuid}


@_async
class TestQuerySystemFilters:
    async def test_session_ids_select_events_and_confine_windows(
        self, event_memory: EventMemory
    ):
        a0 = _make_event("a0", timestamp=_ts(0), session_id="a")
        b0 = _make_event("b0", timestamp=_ts(1), session_id="b")
        a1 = _make_event("a1", timestamp=_ts(2), session_id="a")
        n0 = _make_event("n0", timestamp=_ts(3))
        await event_memory.encode_events([a0, b0, a1, n0])

        hits = await event_memory.query("x", session_ids=["a"], expand_context=6)

        assert {hit.seed.event_uuid for hit in hits} == {
            a0.uuid,
            a1.uuid,
        }
        for hit in hits:
            assert {seg.session_id for seg in hit.window()} == {"a"}

    async def test_source_ids_select_events(self, event_memory: EventMemory):
        alice = _make_event("alice says", timestamp=_ts(0), source_id="alice")
        bob = _make_event("bob says", timestamp=_ts(1), source_id="bob")
        await event_memory.encode_events([alice, bob])

        hits = await event_memory.query("says", source_ids=["bob"], expand_context=6)

        assert _texts(hits) == {"bob says"}

    async def test_since_and_until_bound_hits_and_windows(
        self, event_memory: EventMemory
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory.encode_events(events)

        hits = await event_memory.query(
            "event", since=_ts(1), until=_ts(3), expand_context=6
        )

        assert _texts(hits) == {"event 1", "event 2"}

    async def test_block_kinds_select_segments(self, event_memory: EventMemory):
        await event_memory.encode_events([_make_event("hi")])

        assert len(await event_memory.query("hi", block_kinds=["text"])) == 1
        assert await event_memory.query("hi", block_kinds=["image"]) == []

    async def test_timestamp_filter_field_is_the_segment_store_column(
        self, event_memory: EventMemory
    ):
        """A caller's bare `timestamp` names the event timestamp at the segment store."""
        early = _make_event("early", timestamp=_ts(0))
        late = _make_event("late", timestamp=_ts(10))
        await event_memory.encode_events([early, late])

        hits = await event_memory.query(
            "x",
            property_filter=Comparison(field="timestamp", op=">=", value=_ts(5)),
            expand_context=6,
        )

        assert _texts(hits) == {"late"}

    async def test_the_property_filter_never_reaches_the_vector_store(
        self, event_memory: EventMemory, fake_vector_store_collection, monkeypatch
    ):
        """The vector stage evaluates the typed filters only; the property filter is the store's."""
        red = _make_event("red", timestamp=_ts(0), properties={"color": "red"})
        blue = _make_event("blue", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([red, blue])
        seen: list[object] = []
        original_query = fake_vector_store_collection.query

        async def recording_query(**kwargs):
            seen.append(kwargs.get("property_filter"))
            return await original_query(**kwargs)

        monkeypatch.setattr(fake_vector_store_collection, "query", recording_query)

        hits = await event_memory.query(
            "x",
            session_ids=["s"],
            property_filter=Comparison(field="m.color", op="=", value="blue"),
        )

        assert _texts(hits) == {"blue"}
        assert seen == [In(field=EVENT_SESSION_KEY, values=["s"])]


# ===================================================================
# expand
# ===================================================================


@_async
class TestExpand:
    async def test_expand_around_a_segment(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory.encode_events(events)
        [seed_uuid] = fake_segment_store_partition.event_to_segments[events[2].uuid]

        neighborhood = await event_memory.expand(seed_uuid, before=1, after=2)

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
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        red = _make_event("red", timestamp=_ts(0), properties={"color": "red"})
        blue = _make_event("blue", timestamp=_ts(1), properties={"color": "blue"})
        green = _make_event("green", timestamp=_ts(2), properties={"color": "green"})
        await event_memory.encode_events([red, blue, green])
        [seed_uuid] = fake_segment_store_partition.event_to_segments[blue.uuid]

        neighborhood = await event_memory.expand(
            seed_uuid,
            before=5,
            after=5,
            property_filter=Comparison(field="m.color", op="=", value="green"),
        )

        assert neighborhood.before == []
        assert [s.event_uuid for s in neighborhood.after] == [green.uuid]

    async def test_expand_session_ids_bound_what_the_seed_may_be_in(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        a0 = _make_event("a0", timestamp=_ts(0), session_id="a")
        b0 = _make_event("b0", timestamp=_ts(1), session_id="b")
        a1 = _make_event("a1", timestamp=_ts(2), session_id="a")
        await event_memory.encode_events([a0, b0, a1])
        [seed_uuid] = fake_segment_store_partition.event_to_segments[a0.uuid]

        neighborhood = await event_memory.expand(seed_uuid, after=5, session_ids=["a"])
        assert [s.event_uuid for s in neighborhood.after] == [a1.uuid]
        with pytest.raises(LookupError):
            await event_memory.expand(seed_uuid, after=5, session_ids=["b"])

    async def test_negative_counts_are_rejected(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        event = _make_event("x", timestamp=_ts(0))
        await event_memory.encode_events([event])
        [seed_uuid] = fake_segment_store_partition.event_to_segments[event.uuid]

        with pytest.raises(ValueError, match="before must be nonnegative"):
            await event_memory.expand(seed_uuid, before=-1)
        with pytest.raises(ValueError, match="expand_context must be nonnegative"):
            await event_memory.query("x", expand_context=-1)

    async def test_expand_walks_further_from_an_edge(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory.encode_events(events)

        [seed_uuid] = fake_segment_store_partition.event_to_segments[events[0].uuid]
        first = await event_memory.expand(seed_uuid, after=2)
        second = await event_memory.expand(first.after[-1].uuid, after=2)

        assert [s.event_uuid for s in first.after] == [events[1].uuid, events[2].uuid]
        assert [s.event_uuid for s in second.after] == [events[3].uuid, events[4].uuid]

    async def test_unknown_seed_raises(self, event_memory: EventMemory):
        await event_memory.encode_events([_make_event("hi")])
        with pytest.raises(LookupError):
            await event_memory.expand(uuid4(), before=1, after=1)


# ===================================================================
# forget_events
# ===================================================================


@_async
class TestForgetEvents:
    async def test_forget_basic(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        e1 = _make_event("keep me", timestamp=_ts(0))
        e2 = _make_event("forget me", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        assert len(fake_segment_store_partition.segments) == 2
        assert len(fake_vector_store_collection.records) == 2

        await event_memory.forget_events([e2.uuid])

        # Only e1's data remains.
        assert len(fake_segment_store_partition.segments) == 1
        remaining_segment = next(iter(fake_segment_store_partition.segments.values()))
        assert remaining_segment.event_uuid == e1.uuid
        assert len(fake_vector_store_collection.records) == 1

    async def test_forget_empty_set(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        await event_memory.encode_events([_make_event("keep me")])
        await event_memory.forget_events([])
        assert len(fake_segment_store_partition.segments) == 1

    async def test_forget_nonexistent(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        await event_memory.encode_events([_make_event("keep me")])
        await event_memory.forget_events([uuid4()])
        assert len(fake_segment_store_partition.segments) == 1


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
        session_id="s",
        source_id="src",
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=timestamp,
        block=TextBlock(text=text),
        context=context if context is not None else NullContext(),
    )


class TestRender:
    def test_no_context(self):
        segment = _make_segment(text="hello world")
        result = EventMemory.render_segments([segment], datetime_format=_SHORT_TIME)
        assert json.dumps("hello world") in result
        assert "[" in result  # Timestamp bracket.

    def test_producer_renders_its_name(self):
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
    async def test_scores_are_replaced_and_ordered(self, event_memory: EventMemory):
        e1 = _make_event("short", timestamp=_ts(0))
        e2 = _make_event("a much longer text", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])
        hits = await event_memory.query("anything")

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
        self, event_memory: EventMemory
    ):
        events = [_make_event("x" * (i + 1), timestamp=_ts(i)) for i in range(4)]
        await event_memory.encode_events(events)
        hits = await event_memory.query("anything")

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
        self, event_memory: EventMemory
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
        await event_memory.encode_events([e1, e2])

        hits = await event_memory.query("test query")
        assert len(hits) == 2
        assert _texts(hits) == {"The quick brown fox", "jumps over the lazy dog"}

    async def test_encode_then_query_preserves_context(self, event_memory: EventMemory):
        """Query results should carry the original context."""
        event = _make_event("hello", context=_author("Alice"), timestamp=_ts(0))
        await event_memory.encode_events([event])

        [hit] = await event_memory.query("test")
        assert hit.seed.context == ProducerContext(producer="Alice")

    async def test_forget_then_query_excludes_forgotten(
        self, event_memory: EventMemory
    ):
        """Forgotten events must not appear in query results."""
        e1 = _make_event("keep this one", timestamp=_ts(0))
        e2 = _make_event("forget this one", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        await event_memory.forget_events([e2.uuid])

        texts = _texts(await event_memory.query("test query"))
        assert "keep this one" in texts
        assert "forget this one" not in texts

    async def test_forget_all_then_query_returns_empty(self, event_memory: EventMemory):
        e1 = _make_event("first", timestamp=_ts(0))
        e2 = _make_event("second", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        await event_memory.forget_events([e1.uuid, e2.uuid])

        assert await event_memory.query("test") == []

    async def test_multiple_encode_calls_are_additive(self, event_memory: EventMemory):
        e1 = _make_event("batch one", timestamp=_ts(0))
        e2 = _make_event("batch two", timestamp=_ts(1))

        await event_memory.encode_events([e1])
        await event_memory.encode_events([e2])

        assert _texts(await event_memory.query("test query")) == {
            "batch one",
            "batch two",
        }

    async def test_query_result_rendered_as_string(self, event_memory: EventMemory):
        """End-to-end: encode, query, render."""
        event = _make_event(
            "The mitochondria is the powerhouse of the cell.",
            context=_author("textbook"),
            timestamp=_ts(0),
        )
        await event_memory.encode_events([event])

        [hit] = await event_memory.query("biology")
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
    async def test_equality_filter(self, event_memory: EventMemory):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        hits = await event_memory.query(
            "thing",
            property_filter=Comparison(field="m.color", op="=", value="red"),
        )
        assert _texts(hits) == {"red thing"}

    async def test_inequality_filter(self, event_memory: EventMemory):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        hits = await event_memory.query(
            "thing",
            property_filter=Comparison(field="m.color", op="!=", value="red"),
        )
        assert _texts(hits) == {"blue thing"}

    async def test_in_filter(self, event_memory: EventMemory):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        e3 = _make_event("green thing", timestamp=_ts(2), properties={"color": "green"})
        await event_memory.encode_events([e1, e2, e3])

        hits = await event_memory.query(
            "thing",
            property_filter=In(field="m.color", values=["red", "green"]),
        )
        assert _texts(hits) == {"red thing", "green thing"}

    async def test_is_null_filter(self, event_memory: EventMemory):
        e1 = _make_event("has color", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("no color", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        hits = await event_memory.query(
            "thing",
            property_filter=IsNull(field="m.color"),
        )
        assert _texts(hits) == {"no color"}

    async def test_and_filter(self, event_memory: EventMemory):
        e1 = _make_event("red small", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue small", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        hits = await event_memory.query(
            "thing",
            property_filter=And(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Not(expr=IsNull(field="m.color")),
            ),
        )
        assert _texts(hits) == {"red small"}

    async def test_or_filter(self, event_memory: EventMemory):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        e3 = _make_event("green thing", timestamp=_ts(2), properties={"color": "green"})
        await event_memory.encode_events([e1, e2, e3])

        hits = await event_memory.query(
            "thing",
            property_filter=Or(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Comparison(field="m.color", op="=", value="blue"),
            ),
        )
        assert _texts(hits) == {"red thing", "blue thing"}

    async def test_not_filter(self, event_memory: EventMemory):
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        hits = await event_memory.query(
            "thing",
            property_filter=Not(expr=Comparison(field="m.color", op="=", value="red")),
        )
        assert _texts(hits) == {"blue thing"}

    async def test_filter_returns_empty_when_nothing_matches(
        self, event_memory: EventMemory
    ):
        await event_memory.encode_events(
            [_make_event("red thing", timestamp=_ts(0), properties={"color": "red"})]
        )

        hits = await event_memory.query(
            "thing",
            property_filter=Comparison(field="m.color", op="=", value="purple"),
        )
        assert hits == []

    async def test_context_filter_returns_no_results(self, event_memory: EventMemory):
        """Context is never filterable."""
        await event_memory.encode_events([_make_event("hi", context=_author("Alice"))])

        hits = await event_memory.query(
            "hi",
            property_filter=Comparison(field="context.author", op="=", value="Alice"),
        )
        assert hits == []


# ===================================================================
# Query deduplication
# ===================================================================


@_async
class TestQueryDeduplication:
    async def test_multiple_derivatives_deduplicate_to_one_segment(
        self,
        event_memory_with_sentences: EventMemory,
    ):
        event = _make_event(
            "First sentence. Second sentence. Third sentence.",
            timestamp=_ts(0),
        )
        await event_memory_with_sentences.encode_events([event])

        hits = await event_memory_with_sentences.query("sentence")

        # All derivatives map to the same segment, so deduplication
        # should collapse them into a single hit.
        assert len(hits) == 1
        assert len(hits[0].window()) == 1

    async def test_derivatives_from_different_segments_not_collapsed(
        self,
        event_memory_with_sentences: EventMemory,
    ):
        e1 = _make_event("Alpha sentence. Beta sentence.", timestamp=_ts(0))
        e2 = _make_event("Gamma sentence. Delta sentence.", timestamp=_ts(1))
        await event_memory_with_sentences.encode_events([e1, e2])

        assert len(await event_memory_with_sentences.query("sentence")) == 2

    async def test_dedup_uses_best_derivative_score(self):
        """When multiple derivatives map to one segment, the best score wins."""
        embedder = AngleEmbedder({"Close": 0.1, "Distant": 1.0, "query": 0.0})
        memory = EventMemory(
            EventMemoryParams(
                segment_store_partition=InMemorySegmentStorePartition(),
                vector_store_collection=make_collection(embedder),
                segmenter=TextSegmenter(),
                deriver=SentenceTextDeriver(),
                embedder=embedder,
            )
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
        event_memory = _build(embedder)

        await event_memory.encode_events([_make_event("hello world")])

        # _T0 is 2025-06-01; the ingest default is a full date with no time,
        # and the message text is JSON-dumped (ensure_ascii=False).
        assert embedder.ingested == ['[Sunday, June 1, 2025] "hello world"']

    async def test_the_deriver_decides_the_format(self):
        embedder = _RecordingEmbedder()
        event_memory = _build(
            embedder,
            deriver=WholeTextDeriver(DateTimeFormat(date_style=None, time_style=None)),
        )

        await event_memory.encode_events([_make_event("hello world")])

        assert embedder.ingested == ['"hello world"']

    async def test_author_is_embedded_by_name(self):
        embedder = _RecordingEmbedder()
        event_memory = _build(
            embedder,
            deriver=WholeTextDeriver(DateTimeFormat(date_style=None, time_style=None)),
        )

        await event_memory.encode_events(
            [_make_event("hello", context=_author("Alice"), source_id="u-1")]
        )

        assert embedder.ingested == ['Alice: "hello"']

    async def test_segment_text_stays_structured(self):
        """The baked timestamp lives only in the embedding, not the segment."""
        embedder = _RecordingEmbedder()
        partition = InMemorySegmentStorePartition()
        event_memory = _build(embedder, partition=partition)

        await event_memory.encode_events([_make_event("hello world")])

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
        session_ids=["s1"],
        source_ids=["alice", "bob"],
        block_kinds=["text"],
    )
    assert _conjuncts(tree) == [
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T0),
        Comparison(field=EVENT_TIMESTAMP_KEY, op="<", value=_T1),
        In(field=EVENT_SESSION_KEY, values=["s1"]),
        In(field=EVENT_SOURCE_KEY, values=["alice", "bob"]),
        In(field=BLOCK_KIND_KEY, values=["text"]),
    ]


def test_empty_ids_admit_nothing_and_none_admits_everything():
    assert _system_predicates(session_ids=None) is None
    assert _system_predicates(session_ids=[]) == In(field=EVENT_SESSION_KEY, values=[])
