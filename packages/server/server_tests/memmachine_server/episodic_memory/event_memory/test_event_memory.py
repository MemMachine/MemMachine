"""Tests for EventMemory."""

import datetime
import json
from datetime import UTC
from uuid import uuid4

import pytest

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
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
    CitationContext,
    Content,
    Event,
    FileRef,
    MessageContext,
    NullContext,
    QueryResult,
    ReadFile,
    ScoredSegmentContext,
    Segment,
    Text,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)

from .conftest import (
    InMemorySegmentStorePartition,
    InMemoryVectorStoreCollection,
)

_async = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime.datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
_NULL_CONTEXT = NullContext()


def _record_properties(record: Record) -> dict[str, PropertyValue]:
    """Get properties from a record, asserting they are not None."""
    assert record.properties is not None
    return record.properties


def _make_event(
    text: str,
    *,
    timestamp: datetime.datetime = _T0,
    context: MessageContext | CitationContext | NullContext = _NULL_CONTEXT,
    properties=None,
) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=timestamp,
        body=Content(context=context, items=[Text(text=text)]),
        properties=properties or {},
    )


def _ts(minutes: int) -> datetime.datetime:
    """Return _T0 + minutes."""
    return _T0 + datetime.timedelta(minutes=minutes)


# ===================================================================
# schema
# ===================================================================


class TestSchema:
    def test_expected_vector_store_collection_schema_only_has_base_fields(self):
        assert EventMemory.expected_vector_store_collection_schema() == {
            "_segment_uuid": str,
            "_timestamp": datetime.datetime,
        }


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
        assert segment.block == Text(text="hello world")

        # One derivative record in vector store.
        assert len(fake_vector_store_collection.records) == 1
        record = next(iter(fake_vector_store_collection.records.values()))
        props = _record_properties(record)
        assert props["_segment_uuid"] == str(segment.uuid)
        assert props["_timestamp"] == event.timestamp

    async def test_multiple_events_sorted_by_timestamp(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        e1 = _make_event("first", timestamp=_ts(3))
        e2 = _make_event("second", timestamp=_ts(1))
        e3 = _make_event("third", timestamp=_ts(2))

        await event_memory.encode_events([e1, e2, e3])

        # Segments stored in chronological order.
        ordered_segments = [
            fake_segment_store_partition.segments[uid]
            for uid in fake_segment_store_partition.segment_order
        ]
        timestamps = [s.timestamp for s in ordered_segments]
        assert timestamps == sorted(timestamps)

    async def test_message_context(
        self,
        event_memory: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("hi", context=MessageContext(source="Alice"))
        await event_memory.encode_events([event])

        record = next(iter(fake_vector_store_collection.records.values()))
        props = _record_properties(record)
        assert "_context_type" not in props
        assert "_context_source" not in props

    async def test_citation_context(
        self,
        event_memory: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("abstract", context=CitationContext(source="paper.pdf"))
        await event_memory.encode_events([event])

        record = next(iter(fake_vector_store_collection.records.values()))
        props = _record_properties(record)
        assert "_context_type" not in props
        assert "_context_source" not in props

    async def test_no_context(
        self,
        event_memory: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("bare text")
        await event_memory.encode_events([event])

        record = next(iter(fake_vector_store_collection.records.values()))
        props = _record_properties(record)
        assert "_context_type" not in props
        assert "_context_source" not in props

    async def test_read_file_body(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = Event(
            uuid=uuid4(),
            timestamp=_T0,
            body=ReadFile(file=FileRef()),
        )
        await event_memory.encode_events([event])

        # 1 segment, 0 derivatives, 0 vector records.
        assert len(fake_segment_store_partition.segments) == 1
        segment = next(iter(fake_segment_store_partition.segments.values()))
        assert isinstance(segment.block, FileRef)
        assert fake_segment_store_partition.segment_to_derivatives[segment.uuid] == []
        assert len(fake_vector_store_collection.records) == 0

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
            assert isinstance(segment.block, Text)
            assert len(segment.block.text) <= 2000

    async def test_multiple_content_items(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        event = Event(
            uuid=uuid4(),
            timestamp=_T0,
            body=Content(items=[Text(text="first"), Text(text="second")]),
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

    async def test_user_properties_propagate(
        self,
        event_memory: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = _make_event("hi", properties={"color": "red"})
        await event_memory.encode_events([event])

        record = next(iter(fake_vector_store_collection.records.values()))
        assert _record_properties(record)["color"] == "red"

    async def test_missing_context_schema_fields_still_allows_ingest(
        self, fake_embedder
    ):
        # Collection without context fields.
        config = VectorStoreCollectionConfig(
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema={
                "_segment_uuid": str,
                "_timestamp": datetime.datetime,
            },
        )
        collection = InMemoryVectorStoreCollection(config)
        partition = InMemorySegmentStorePartition()
        em = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=partition,
                embedder=fake_embedder,
            )
        )
        event = _make_event("hi", context=MessageContext(source="Alice"))
        await em.encode_events([event])

        assert len(collection.records) == 1
        record = next(iter(collection.records.values()))
        props = _record_properties(record)
        assert "_context_type" not in props
        assert "_context_source" not in props

    async def test_init_raises_on_missing_base_field(self, fake_embedder):
        # Collection without _timestamp — base field required at init.
        config = VectorStoreCollectionConfig(
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema={
                "_segment_uuid": str,
            },
        )
        collection = InMemoryVectorStoreCollection(config)
        partition = InMemorySegmentStorePartition()
        with pytest.raises(
            ValueError,
            match="Collection schema missing fields required by EventMemory",
        ):
            EventMemory(
                EventMemoryParams(
                    vector_store_collection=collection,
                    segment_store_partition=partition,
                    embedder=fake_embedder,
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

        result = await event_memory.query("test query")
        assert isinstance(result, QueryResult)
        assert len(result.scored_segment_contexts) > 0

        # Each scored context has segments.
        for scored in result.scored_segment_contexts:
            assert len(scored.segments) > 0

    async def test_vector_search_limit(self, event_memory: EventMemory):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory.encode_events(events)

        result = await event_memory.query("test", vector_search_limit=2)
        assert len(result.scored_segment_contexts) <= 2

    async def test_expand_context(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
    ):
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory.encode_events(events)

        result = await event_memory.query("test query", expand_context=3)

        # With expand_context=3, backward=1, forward=2.
        # Context windows should include neighbors.
        for scored in result.scored_segment_contexts:
            assert len(scored.segments) >= 1

    async def test_empty_memory(self, event_memory: EventMemory):
        result = await event_memory.query("anything")
        assert result.scored_segment_contexts == []

    async def test_without_reranker_uses_embedding_scores(
        self, event_memory: EventMemory
    ):
        await event_memory.encode_events([_make_event("hello")])
        result = await event_memory.query("hello")

        # FakeEmbedder: all vectors same direction → cosine ≈ 1.0.
        for scored in result.scored_segment_contexts:
            assert scored.score == pytest.approx(1.0, abs=0.01)

    async def test_with_reranker(self, event_memory_with_reranker: EventMemory):
        e1 = _make_event("short", timestamp=_ts(0))
        e2 = _make_event("a much longer text", timestamp=_ts(1))
        await event_memory_with_reranker.encode_events([e1, e2])

        result = await event_memory_with_reranker.query("anything")

        # FakeReranker scores by string length (higher is better).
        # The result with the longer formatted context should come first.
        scores = [sc.score for sc in result.scored_segment_contexts]
        assert scores == sorted(scores, reverse=True)
        assert len(scores) == 2
        assert scores[0] > scores[1]


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

    async def test_forget_read_file(
        self,
        event_memory: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        event = Event(
            uuid=uuid4(),
            timestamp=_T0,
            body=ReadFile(file=FileRef()),
        )
        await event_memory.encode_events([event])
        assert len(fake_segment_store_partition.segments) == 1
        assert len(fake_vector_store_collection.records) == 0

        await event_memory.forget_events([event.uuid])
        assert len(fake_segment_store_partition.segments) == 0


# ===================================================================
# build_query_result_context (static, sync)
# ===================================================================


def _make_segment(
    *,
    event_uuid=None,
    index: int = 0,
    offset: int = 0,
    timestamp: datetime.datetime = _T0,
    text: str = "text",
    context: MessageContext | CitationContext | NullContext = _NULL_CONTEXT,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=timestamp,
        block=Text(text=text),
        context=context,
    )


class TestBuildQueryResultContext:
    def test_under_limit(self):
        s1 = _make_segment(timestamp=_ts(0))
        s2 = _make_segment(timestamp=_ts(1))
        s3 = _make_segment(timestamp=_ts(2))

        qr = QueryResult(
            scored_segment_contexts=[
                ScoredSegmentContext(
                    score=1.0, seed_segment_uuid=s1.uuid, segments=[s1]
                ),
                ScoredSegmentContext(
                    score=0.5, seed_segment_uuid=s2.uuid, segments=[s2, s3]
                ),
            ]
        )
        result = EventMemory.build_query_result_context(qr, max_num_segments=10)
        assert len(result) == 3
        # Sorted chronologically.
        assert result == sorted(
            result,
            key=lambda s: (s.timestamp, s.event_uuid, s.index, s.offset),
        )

    def test_over_limit_prioritizes_seed(self):
        # 5 segments, seed in the middle (index 2).
        event_uuid = uuid4()
        segments = [
            _make_segment(
                event_uuid=event_uuid, index=i, timestamp=_ts(i), text=f"seg{i}"
            )
            for i in range(5)
        ]
        seed = segments[2]

        qr = QueryResult(
            scored_segment_contexts=[
                ScoredSegmentContext(
                    score=1.0,
                    seed_segment_uuid=seed.uuid,
                    segments=segments,
                ),
            ]
        )
        result = EventMemory.build_query_result_context(qr, max_num_segments=3)
        assert len(result) == 3
        # Seed must be included.
        result_uuids = {s.uuid for s in result}
        assert seed.uuid in result_uuids

    def test_deduplicates_across_contexts(self):
        shared = _make_segment(timestamp=_ts(0))
        s1 = _make_segment(timestamp=_ts(1))
        s2 = _make_segment(timestamp=_ts(2))

        qr = QueryResult(
            scored_segment_contexts=[
                ScoredSegmentContext(
                    score=1.0,
                    seed_segment_uuid=shared.uuid,
                    segments=[shared, s1],
                ),
                ScoredSegmentContext(
                    score=0.5,
                    seed_segment_uuid=shared.uuid,
                    segments=[shared, s2],
                ),
            ]
        )
        result = EventMemory.build_query_result_context(qr, max_num_segments=10)
        uuids = [s.uuid for s in result]
        assert len(uuids) == len(set(uuids))  # No duplicates.
        assert len(result) == 3  # shared, s1, s2

    def test_empty_result(self):
        qr = QueryResult(scored_segment_contexts=[])
        result = EventMemory.build_query_result_context(qr, max_num_segments=10)
        assert result == []

    def test_budget_exhaustion_across_contexts(self):
        # First context: 3 segments. Second context: 4 segments. Budget: 5.
        ctx1_segs = [_make_segment(timestamp=_ts(i)) for i in range(3)]
        ctx2_segs = [_make_segment(timestamp=_ts(10 + i)) for i in range(4)]

        qr = QueryResult(
            scored_segment_contexts=[
                ScoredSegmentContext(
                    score=1.0,
                    seed_segment_uuid=ctx1_segs[1].uuid,
                    segments=ctx1_segs,
                ),
                ScoredSegmentContext(
                    score=0.5,
                    seed_segment_uuid=ctx2_segs[1].uuid,
                    segments=ctx2_segs,
                ),
            ]
        )
        result = EventMemory.build_query_result_context(qr, max_num_segments=5)
        assert len(result) == 5
        # First context fully included.
        result_uuids = {s.uuid for s in result}
        for seg in ctx1_segs:
            assert seg.uuid in result_uuids


# ===================================================================
# string_from_segment_context (static, sync)
# ===================================================================


class TestStringFromSegmentContext:
    def test_no_context(self):
        segment = _make_segment(text="hello world")
        result = EventMemory.string_from_segment_context([segment])
        assert json.dumps("hello world") in result
        assert "[" in result  # Timestamp bracket.

    def test_message_context(self):
        segment = _make_segment(text="hi", context=MessageContext(source="Alice"))
        result = EventMemory.string_from_segment_context([segment])
        assert "Alice:" in result
        assert json.dumps("hi") in result

    def test_citation_context(self):
        segment = _make_segment(
            text="abstract", context=CitationContext(source="paper.pdf")
        )
        result = EventMemory.string_from_segment_context([segment])
        assert "From 'paper.pdf':" in result
        assert json.dumps("abstract") in result

    def test_continuation_segments(self):
        event_uuid = uuid4()
        s1 = _make_segment(event_uuid=event_uuid, index=0, offset=0, text="part1")
        s2 = Segment(
            uuid=uuid4(),
            event_uuid=event_uuid,
            index=0,
            offset=1,
            timestamp=_T0,
            block=Text(text="part2"),
        )
        result = EventMemory.string_from_segment_context([s1, s2])
        # Text is accumulated into one JSON string.
        assert json.dumps("part1part2") in result
        # Only one timestamp line.
        assert result.count("[") == 1

    def test_empty_list(self):
        result = EventMemory.string_from_segment_context([])
        assert result == ""


# ===================================================================
# Round-trip tests (encode → query/forget → verify via public API)
# ===================================================================


@_async
class TestRoundTrips:
    async def test_encode_then_query_returns_encoded_content(
        self, event_memory: EventMemory
    ):
        """Encoded events should be retrievable through query."""
        e1 = _make_event(
            "The quick brown fox",
            context=MessageContext(source="Alice"),
            timestamp=_ts(0),
        )
        e2 = _make_event(
            "jumps over the lazy dog",
            context=MessageContext(source="Bob"),
            timestamp=_ts(1),
        )
        await event_memory.encode_events([e1, e2])

        result = await event_memory.query("test query")
        assert len(result.scored_segment_contexts) == 2

        # Verify the actual content is present in the returned segments.
        all_segments = [
            seg for scored in result.scored_segment_contexts for seg in scored.segments
        ]
        all_texts = {
            seg.block.text for seg in all_segments if isinstance(seg.block, Text)
        }
        assert "The quick brown fox" in all_texts
        assert "jumps over the lazy dog" in all_texts

    async def test_encode_then_query_preserves_context(self, event_memory: EventMemory):
        """Query results should carry the original context."""
        event = _make_event(
            "hello", context=MessageContext(source="Alice"), timestamp=_ts(0)
        )
        await event_memory.encode_events([event])

        result = await event_memory.query("test")
        segment = result.scored_segment_contexts[0].segments[0]
        assert isinstance(segment.context, MessageContext)
        assert segment.context.source == "Alice"

    async def test_forget_then_query_excludes_forgotten(
        self, event_memory: EventMemory
    ):
        """Forgotten events must not appear in query results."""
        e1 = _make_event("keep this one", timestamp=_ts(0))
        e2 = _make_event("forget this one", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        await event_memory.forget_events([e2.uuid])

        result = await event_memory.query("test query")
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "keep this one" in all_texts
        assert "forget this one" not in all_texts

    async def test_forget_all_then_query_returns_empty(self, event_memory: EventMemory):
        """After forgetting all events, query should return nothing."""
        e1 = _make_event("first", timestamp=_ts(0))
        e2 = _make_event("second", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        await event_memory.forget_events([e1.uuid, e2.uuid])

        result = await event_memory.query("test")
        assert result.scored_segment_contexts == []

    async def test_multiple_encode_calls_are_additive(self, event_memory: EventMemory):
        """Successive encode_events calls should accumulate data."""
        e1 = _make_event("batch one", timestamp=_ts(0))
        e2 = _make_event("batch two", timestamp=_ts(1))

        await event_memory.encode_events([e1])
        await event_memory.encode_events([e2])

        result = await event_memory.query("test query")
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "batch one" in all_texts
        assert "batch two" in all_texts

    async def test_expand_context_includes_neighbors(self, event_memory: EventMemory):
        """Query with expand_context should return neighboring segments."""
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(5)]
        await event_memory.encode_events(events)

        result = await event_memory.query(
            "test query", vector_search_limit=1, expand_context=6
        )

        # With expand_context=6: backward=2, forward=4.
        # Even with only 1 seed, context window should include neighbors.
        assert len(result.scored_segment_contexts) == 1
        context_segments = result.scored_segment_contexts[0].segments
        assert len(context_segments) > 1

        # Context segments should be from distinct events.
        event_uuids = {seg.event_uuid for seg in context_segments}
        assert len(event_uuids) > 1

    async def test_query_result_formatted_as_string(self, event_memory: EventMemory):
        """End-to-end: encode, query, format as string."""
        event = _make_event(
            "The mitochondria is the powerhouse of the cell.",
            context=MessageContext(source="textbook"),
            timestamp=_ts(0),
        )
        await event_memory.encode_events([event])

        result = await event_memory.query("biology")
        segments = EventMemory.build_query_result_context(result, max_num_segments=10)
        context_string = EventMemory.string_from_segment_context(segments)

        assert "textbook:" in context_string
        assert "The mitochondria is the powerhouse of the cell." in context_string


# ===================================================================
# Filtering round-trip tests
# ===================================================================


@_async
class TestQueryWithFilter:
    async def test_equality_filter(self, event_memory: EventMemory):
        """Filter by user property equality."""
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        result = await event_memory.query(
            "thing",
            property_filter=Comparison(field="m.color", op="=", value="red"),
        )
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "red thing" in all_texts
        assert "blue thing" not in all_texts

    async def test_inequality_filter(self, event_memory: EventMemory):
        """Filter by != excludes matching events."""
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        result = await event_memory.query(
            "thing",
            property_filter=Comparison(field="m.color", op="!=", value="red"),
        )
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "blue thing" in all_texts
        assert "red thing" not in all_texts

    async def test_in_filter(self, event_memory: EventMemory):
        """Filter by IN membership."""
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        e3 = _make_event("green thing", timestamp=_ts(2), properties={"color": "green"})
        await event_memory.encode_events([e1, e2, e3])

        result = await event_memory.query(
            "thing",
            property_filter=In(field="m.color", values=["red", "green"]),
        )
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "red thing" in all_texts
        assert "green thing" in all_texts
        assert "blue thing" not in all_texts

    async def test_is_null_filter(self, event_memory: EventMemory):
        """Filter by IS NULL matches events missing the property."""
        e1 = _make_event("has color", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("no color", timestamp=_ts(1))
        await event_memory.encode_events([e1, e2])

        result = await event_memory.query(
            "thing",
            property_filter=IsNull(field="m.color"),
        )
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "no color" in all_texts
        assert "has color" not in all_texts

    async def test_and_filter(self, event_memory: EventMemory):
        """Filter by AND conjunction."""
        e1 = _make_event("red small", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue small", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        result = await event_memory.query(
            "thing",
            property_filter=And(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Not(expr=IsNull(field="m.color")),
            ),
        )
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "red small" in all_texts
        assert "blue small" not in all_texts

    async def test_or_filter(self, event_memory: EventMemory):
        """Filter by OR disjunction."""
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        e3 = _make_event("green thing", timestamp=_ts(2), properties={"color": "green"})
        await event_memory.encode_events([e1, e2, e3])

        result = await event_memory.query(
            "thing",
            property_filter=Or(
                left=Comparison(field="m.color", op="=", value="red"),
                right=Comparison(field="m.color", op="=", value="blue"),
            ),
        )
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "red thing" in all_texts
        assert "blue thing" in all_texts
        assert "green thing" not in all_texts

    async def test_not_filter(self, event_memory: EventMemory):
        """Filter by NOT negation."""
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        e2 = _make_event("blue thing", timestamp=_ts(1), properties={"color": "blue"})
        await event_memory.encode_events([e1, e2])

        result = await event_memory.query(
            "thing",
            property_filter=Not(expr=Comparison(field="m.color", op="=", value="red")),
        )
        all_texts = {
            seg.block.text
            for scored in result.scored_segment_contexts
            for seg in scored.segments
            if isinstance(seg.block, Text)
        }
        assert "blue thing" in all_texts
        assert "red thing" not in all_texts

    async def test_filter_returns_empty_when_nothing_matches(
        self, event_memory: EventMemory
    ):
        """Filter that matches nothing returns empty results."""
        e1 = _make_event("red thing", timestamp=_ts(0), properties={"color": "red"})
        await event_memory.encode_events([e1])

        result = await event_memory.query(
            "thing",
            property_filter=Comparison(field="m.color", op="=", value="purple"),
        )
        assert result.scored_segment_contexts == []

    async def test_context_filter_returns_no_results(self, event_memory: EventMemory):
        """Context fields are no longer filterable."""
        event = _make_event("hi", context=MessageContext(source="Alice"))
        await event_memory.encode_events([event])

        result = await event_memory.query(
            "hi",
            property_filter=Comparison(
                field="context.source",
                op="=",
                value="Alice",
            ),
        )
        assert result.scored_segment_contexts == []


# ===================================================================
# Query deduplication
# ===================================================================


@_async
class TestQueryDeduplication:
    async def test_multiple_derivatives_deduplicate_to_one_segment(
        self,
        event_memory_with_sentences: EventMemory,
    ):
        """Multiple derivatives from the same segment should produce one scored context."""
        event = _make_event(
            "First sentence. Second sentence. Third sentence.",
            timestamp=_ts(0),
        )
        await event_memory_with_sentences.encode_events([event])

        result = await event_memory_with_sentences.query("sentence")

        # All derivatives map to the same segment, so deduplication
        # should collapse them into a single scored context.
        assert len(result.scored_segment_contexts) == 1
        assert len(result.scored_segment_contexts[0].segments) == 1

    async def test_derivatives_from_different_segments_not_collapsed(
        self,
        event_memory_with_sentences: EventMemory,
    ):
        """Derivatives from different segments should remain separate scored contexts."""
        e1 = _make_event(
            "Alpha sentence. Beta sentence.",
            timestamp=_ts(0),
        )
        e2 = _make_event(
            "Gamma sentence. Delta sentence.",
            timestamp=_ts(1),
        )
        await event_memory_with_sentences.encode_events([e1, e2])

        result = await event_memory_with_sentences.query("sentence")

        # Two events → two segments → two scored contexts.
        assert len(result.scored_segment_contexts) == 2

    async def test_dedup_uses_best_derivative_score(
        self,
        event_memory_with_sentences: EventMemory,
    ):
        """When multiple derivatives map to one segment, the best score wins."""
        event = _make_event(
            "Short. A much longer second sentence here.",
            timestamp=_ts(0),
        )
        await event_memory_with_sentences.encode_events([event])

        result = await event_memory_with_sentences.query("test")

        # Cosine sim ≈ 1.0 for all (FakeEmbedder), but the key point
        # is that we get exactly one context with a valid score.
        assert len(result.scored_segment_contexts) == 1
        assert result.scored_segment_contexts[0].score == pytest.approx(1.0, abs=0.01)


# ===================================================================
# Eviction
# ===================================================================


@_async
class TestEviction:
    async def test_eviction_disabled_by_default(
        self,
        event_memory: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        """Without eviction params, all derivatives are stored."""
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory.encode_events(events)
        assert len(fake_vector_store_collection.records) == 10

    async def test_eviction_removes_excess_derivatives(
        self,
        event_memory_with_eviction: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        """Eviction keeps vector store within target_size for a dense cluster."""
        # FakeEmbedder: all vectors cosine ≈ 1.0 > threshold 0.5, so all
        # derivatives land in one cluster. target_size=3.
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory_with_eviction.encode_events(events)
        assert len(fake_vector_store_collection.records) <= 3

    async def test_eviction_deletes_existing_derivatives(
        self,
        event_memory_with_eviction: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        """A second batch triggers eviction of records from the first batch."""
        # First batch: 2 events, under target_size=3.
        batch1 = [_make_event(f"first {i}", timestamp=_ts(i)) for i in range(2)]
        await event_memory_with_eviction.encode_events(batch1)
        assert len(fake_vector_store_collection.records) == 2

        # Second batch: 2 more events push cluster over target_size.
        batch2 = [_make_event(f"second {i}", timestamp=_ts(10 + i)) for i in range(2)]
        await event_memory_with_eviction.encode_events(batch2)
        assert len(fake_vector_store_collection.records) <= 3

    async def test_eviction_preserves_segments(
        self,
        event_memory_with_eviction: EventMemory,
        fake_segment_store_partition: InMemorySegmentStorePartition,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        """Segments are always stored even when their derivatives are evicted."""
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory_with_eviction.encode_events(events)

        # All 10 segments exist.
        assert len(fake_segment_store_partition.segments) == 10
        # But some derivatives were evicted/skipped.
        assert len(fake_vector_store_collection.records) < 10

    async def test_eviction_temporal_middle_priority(
        self,
        event_memory_with_eviction: EventMemory,
        fake_vector_store_collection: InMemoryVectorStoreCollection,
    ):
        """Eviction targets the temporal middle, preserving oldest and newest."""
        # Encode 10 events with distinct timestamps.
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory_with_eviction.encode_events(events)

        # Collect surviving timestamps from vector store.
        surviving_timestamps = set()
        for record in fake_vector_store_collection.records.values():
            assert record.properties is not None
            surviving_timestamps.add(record.properties["_timestamp"])

        # The oldest and newest events should survive (protected ends).
        assert _ts(0) in surviving_timestamps  # oldest
        assert _ts(9) in surviving_timestamps  # newest

    async def test_eviction_no_effect_below_threshold(
        self, fake_embedder, fake_vector_store_collection, fake_segment_store_partition
    ):
        """Threshold above max cosine similarity means no eviction occurs."""
        em = EventMemory(
            EventMemoryParams(
                vector_store_collection=fake_vector_store_collection,
                segment_store_partition=fake_segment_store_partition,
                embedder=fake_embedder,
                eviction_similarity_threshold=2.0,  # Above cosine max of 1.0
                eviction_target_size=3,
            )
        )
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await em.encode_events(events)
        # No vectors meet threshold, so all are kept.
        assert len(fake_vector_store_collection.records) == 10

    async def test_query_after_eviction(
        self,
        event_memory_with_eviction: EventMemory,
    ):
        """Query returns consistent results after eviction."""
        events = [_make_event(f"event {i}", timestamp=_ts(i)) for i in range(10)]
        await event_memory_with_eviction.encode_events(events)

        result = await event_memory_with_eviction.query("test")
        # Should return results without errors (no dangling references).
        assert len(result.scored_segment_contexts) > 0
        for scored in result.scored_segment_contexts:
            assert len(scored.segments) > 0
