"""End-to-end: temporal pieces plugged into a real EventMemory.

Wires ``TemporalSegmenter`` (plus a stock ``WholeTextDeriver``) into
``EventMemory`` (with the in-memory test embedder / vector store / segment
store), encodes events, queries, and ranks the result caller-side with
``TemporalScorer``. Proves the whole loop: EventMemory stores
``TimeRangesContext`` through its real encode path and returns it on query,
while staying unaware of the temporal scoring ``TemporalScorer`` then applies.
The deriver is deliberately a stock one -- only the segmenter is
temporal-specific.
"""

from datetime import UTC, datetime
from typing import override
from uuid import uuid4

import pytest

from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    CompositeContext,
    Context,
    Event,
    NullContext,
    ProducerContext,
    QueryResult,
    TextBlock,
    TimeRangesContext,
    find_contexts,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.temporal_segmenter import (
    TemporalSegmenter,
    TemporalSegmenterParams,
)
from memmachine_server.episodic_memory.event_memory.temporal_ranking import (
    temporal_rerank_query_results,
)
from memmachine_server.temporal.extractor import TemporalExtractor
from memmachine_server.temporal.query_planner import TemporalQueryPlan
from memmachine_server.temporal.scoring import TemporalScorer, TemporalScorerParams
from memmachine_server.temporal.time_range import TimeInterval, TimeRange
from server_tests.memmachine_server.common.reranker.fake_embedder import FakeEmbedder
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_collection import (
    InMemoryVectorStoreCollection,
)
from server_tests.memmachine_server.episodic_memory.event_memory.conftest import (
    InMemorySegmentStorePartition,
)


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def time_range(start: datetime, end: datetime) -> TimeRange:
    """A single-interval ``TimeRange``."""
    return TimeRange(intervals=[TimeInterval(start=start, end=end)])


class FakeExtractor(TemporalExtractor):
    @override
    async def extract(self, text, *, ref_time=None):
        if "2024" in text:
            return [time_range(dt(2024, 1), dt(2025, 1))]
        if "2020" in text:
            return [time_range(dt(2020, 1), dt(2021, 1))]
        return []


def make_event(text: str, context: Context | None = None) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        context=context or NullContext(),
        blocks=[TextBlock(text=text)],
    )


def make_memory() -> EventMemory:
    embedder = FakeEmbedder()
    config = VectorStoreCollectionConfig(
        vector_dimensions=embedder.dimensions,
        similarity_metric=embedder.similarity_metric,
        indexed_properties_schema=EventMemory.expected_vector_store_collection_schema(),
    )
    return EventMemory(
        EventMemoryParams(
            segment_store_partition=InMemorySegmentStorePartition(),
            vector_store_collection=InMemoryVectorStoreCollection(config),
            segmenter=TemporalSegmenter(
                TemporalSegmenterParams(temporal_extractor=FakeExtractor())
            ),
            deriver=WholeTextDeriver(),
            embedder=embedder,
        )
    )


@pytest.mark.asyncio
async def test_encode_query_and_caller_rank():
    memory = make_memory()
    # Equal-length texts (differ only in the year) so the length-based test
    # embedder gives identical vectors; the temporal match is the sole
    # discriminator at rank time.
    await memory.encode_events(
        [
            make_event("The launch happened in the year 2024."),
            make_event("The launch happened in the year 2020."),
        ]
    )

    result = await memory.query("when did the launch happen", vector_search_limit=10)

    assert len(result.scored_segment_contexts) == 2
    for scored in result.scored_segment_contexts:
        seed = next(s for s in scored.segments if s.uuid == scored.seed_segment_uuid)
        assert find_contexts(seed.context, TimeRangesContext)

    plan = TemporalQueryPlan(targets=[time_range(dt(2024, 1), dt(2025, 1))])
    reranked = temporal_rerank_query_results(
        TemporalScorer(TemporalScorerParams()), plan, result
    )

    # Transparent layer: QueryResult in, QueryResult out, same two contexts.
    assert isinstance(reranked, QueryResult)
    assert len(reranked.scored_segment_contexts) == 2
    top, other = reranked.scored_segment_contexts
    # The 2024 doc ranks first, carrying a higher (temporally-lifted) score.
    assert "2024" in top.segments[0].block.text
    assert top.score > other.score


@pytest.mark.asyncio
async def test_composed_producer_and_temporal_end_to_end():
    memory = make_memory()
    # WholeTextDeriver prefixes the producer into the embedded text, so the two
    # producer names are kept equal length: with the length-based FakeEmbedder
    # that keeps both docs' vectors identical (pool spread 0), leaving the
    # temporal match as the sole rank discriminator.
    await memory.encode_events(
        [
            make_event(
                "The launch happened in the year 2024.",
                context=ProducerContext(producer="Alice"),
            ),
            make_event(
                "The launch happened in the year 2020.",
                context=ProducerContext(producer="Carol"),
            ),
        ]
    )

    result = await memory.query("when did the launch happen", vector_search_limit=10)
    assert len(result.scored_segment_contexts) == 2
    for scored in result.scored_segment_contexts:
        seed = next(s for s in scored.segments if s.uuid == scored.seed_segment_uuid)
        # Both aspects survived storage, accessible uniformly.
        assert isinstance(seed.context, CompositeContext)
        assert find_contexts(seed.context, ProducerContext)
        assert find_contexts(seed.context, TimeRangesContext)

    plan = TemporalQueryPlan(targets=[time_range(dt(2024, 1), dt(2025, 1))])
    reranked = temporal_rerank_query_results(
        TemporalScorer(TemporalScorerParams()), plan, result
    )
    top, other = reranked.scored_segment_contexts
    assert "2024" in top.segments[0].block.text
    assert top.score > other.score
