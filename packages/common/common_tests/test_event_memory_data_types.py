"""Tests for common event memory data types, formatting, and unification."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from memmachine_common.api.event_memory.data_types import (
    EventMemoryCitationContext,
    EventMemoryFormatOptions,
    EventMemoryMessageContext,
    EventMemoryQueryResult,
    EventMemoryScoredSegmentContext,
    EventMemorySegment,
    EventMemoryText,
    format_segment_context,
    format_segment_contexts,
)


def _make_segment(
    *,
    event_uuid=None,
    index=0,
    offset=0,
    timestamp=None,
    text="hello",
    context=None,
    properties=None,
):
    return EventMemorySegment(
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=timestamp or datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
        context=context,
        block=EventMemoryText(text=text),
        properties=properties or {},
    )


class TestSegmentRoundTrip:
    def test_serialize_deserialize(self):
        seg = _make_segment(
            properties={"count": 42, "ts": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )
        seg2 = EventMemorySegment.model_validate(seg.model_dump(mode="json"))
        assert seg.uuid == seg2.uuid
        assert seg.properties == seg2.properties
        for key in seg.properties:
            assert type(seg.properties[key]) is type(seg2.properties[key])

    def test_empty_properties(self):
        seg = _make_segment()
        seg2 = EventMemorySegment.model_validate(seg.model_dump(mode="json"))
        assert seg2.properties == {}


class TestFormatSegmentContext:
    def test_message_context(self):
        seg = _make_segment(
            context=EventMemoryMessageContext(source="alice"),
            text="hi there",
        )
        result = format_segment_context([seg])
        assert "alice:" in result
        assert "hi there" in result

    def test_citation_context(self):
        seg = _make_segment(
            context=EventMemoryCitationContext(source="doc.pdf"),
            text="some quote",
        )
        result = format_segment_context([seg])
        assert "From 'doc.pdf':" in result
        assert "some quote" in result

    def test_no_context(self):
        seg = _make_segment(text="bare text")
        result = format_segment_context([seg])
        assert "bare text" in result

    def test_continuation_same_event_and_index(self):
        event_uuid = uuid4()
        seg1 = _make_segment(event_uuid=event_uuid, index=0, offset=0, text="part1")
        seg2 = _make_segment(event_uuid=event_uuid, index=0, offset=1, text="part2")
        result = format_segment_context([seg1, seg2])
        assert "part1part2" in result

    def test_timezone_formatting(self):
        tz = timezone(timedelta(hours=9))
        seg = _make_segment(
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
            text="test",
        )
        opts = EventMemoryFormatOptions(timezone=tz)
        result = format_segment_context([seg], format_options=opts)
        # 10:30 UTC == 19:30 in UTC+09:00
        assert "7:30" in result

    def test_styles_none_omits_timestamp(self):
        seg = _make_segment(text="test")
        opts = EventMemoryFormatOptions(date_style=None, time_style=None)
        result = format_segment_context([seg], format_options=opts)
        assert not result.startswith("[")

    def test_time_style_none_omits_time(self):
        seg = _make_segment(
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
            text="test",
        )
        opts = EventMemoryFormatOptions(time_style=None)
        result = format_segment_context([seg], format_options=opts)
        assert "10:30" not in result

    def test_date_style_none_omits_date(self):
        seg = _make_segment(
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
            text="test",
        )
        opts = EventMemoryFormatOptions(date_style=None)
        result = format_segment_context([seg], format_options=opts)
        assert "2026" not in result

    def test_locale_formatting(self):
        seg = _make_segment(
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
            text="test",
        )
        opts_en = EventMemoryFormatOptions(locale="en_US")
        opts_fr = EventMemoryFormatOptions(locale="fr_FR")
        assert format_segment_context([seg], format_options=opts_en) != (
            format_segment_context([seg], format_options=opts_fr)
        )


class TestScoredSegmentContextToString:
    def test_to_string(self):
        seg = _make_segment(
            context=EventMemoryMessageContext(source="bob"),
            text="hey",
        )
        scored = EventMemoryScoredSegmentContext(
            seed_segment_uuid=seg.uuid,
            score=0.9,
            segments=[seg],
        )
        result = scored.to_string()
        assert "bob:" in result
        assert "hey" in result


class TestBuildContext:
    def _make_scored_context(self, num_segments=3, score=0.5):
        seed_uuid = uuid4()
        event_uuid = uuid4()
        segments = [
            _make_segment(event_uuid=event_uuid, index=i, text=f"seg{i}")
            for i in range(num_segments)
        ]
        segments[0] = EventMemorySegment(
            uuid=seed_uuid,
            event_uuid=event_uuid,
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
            block=EventMemoryText(text="seed"),
            properties={},
        )
        return EventMemoryScoredSegmentContext(
            seed_segment_uuid=seed_uuid,
            score=score,
            segments=segments,
        )

    def test_all_fit(self):
        ctx = self._make_scored_context(num_segments=3, score=0.9)
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx])
        result = qr.build_context(max_num_segments=10)
        assert len(result) == 3

    def test_budget_respected(self):
        ctx = self._make_scored_context(num_segments=5, score=0.9)
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx])
        result = qr.build_context(max_num_segments=2)
        assert len(result) == 2

    def test_higher_score_first(self):
        ctx1 = self._make_scored_context(num_segments=2, score=0.3)
        ctx2 = self._make_scored_context(num_segments=2, score=0.9)
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx2, ctx1])
        result = qr.build_context(max_num_segments=2)
        # Should pick from ctx2 (higher score) first
        assert all(seg in ctx2.segments for seg in result)

    def test_deduplication(self):
        shared_seg = _make_segment(text="shared")
        ctx1 = EventMemoryScoredSegmentContext(
            seed_segment_uuid=shared_seg.uuid,
            score=0.9,
            segments=[shared_seg],
        )
        ctx2 = EventMemoryScoredSegmentContext(
            seed_segment_uuid=shared_seg.uuid,
            score=0.5,
            segments=[shared_seg],
        )
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx1, ctx2])
        result = qr.build_context(max_num_segments=10)
        assert len(result) == 1

    def test_chronological_order(self):
        seg1 = _make_segment(
            timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc), text="first"
        )
        seg2 = _make_segment(
            timestamp=datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc), text="second"
        )
        ctx = EventMemoryScoredSegmentContext(
            seed_segment_uuid=seg2.uuid,
            score=0.9,
            segments=[seg2, seg1],
        )
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx])
        result = qr.build_context(max_num_segments=10)
        assert result[0].timestamp < result[1].timestamp

    def test_empty_result(self):
        qr = EventMemoryQueryResult(scored_segment_contexts=[])
        result = qr.build_context(max_num_segments=10)
        assert result == []


class TestToString:
    def test_to_string(self):
        seg = _make_segment(
            context=EventMemoryMessageContext(source="alice"),
            text="hello",
        )
        ctx = EventMemoryScoredSegmentContext(
            seed_segment_uuid=seg.uuid,
            score=0.9,
            segments=[seg],
        )
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx])
        result = qr.to_string(max_num_segments=10)
        assert "alice:" in result
        assert "hello" in result

    def test_to_string_no_limit(self):
        seg = _make_segment(
            context=EventMemoryMessageContext(source="alice"),
            text="hello",
        )
        ctx = EventMemoryScoredSegmentContext(
            seed_segment_uuid=seg.uuid,
            score=0.9,
            segments=[seg],
        )
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx])
        result = qr.to_string()
        assert "alice:" in result
        assert "hello" in result

    def test_to_string_separates_disconnected_contexts(self):
        seg_a = _make_segment(
            context=EventMemoryMessageContext(source="alice"),
            text="first",
        )
        seg_b = _make_segment(
            context=EventMemoryMessageContext(source="bob"),
            text="second",
            timestamp=datetime(2026, 1, 16, 10, 30, tzinfo=timezone.utc),
        )
        ctx_a = EventMemoryScoredSegmentContext(
            seed_segment_uuid=seg_a.uuid, score=0.9, segments=[seg_a]
        )
        ctx_b = EventMemoryScoredSegmentContext(
            seed_segment_uuid=seg_b.uuid, score=0.8, segments=[seg_b]
        )
        qr = EventMemoryQueryResult(scored_segment_contexts=[ctx_a, ctx_b])
        result = qr.to_string()
        assert "\n\n" in result
        assert "alice:" in result
        assert "bob:" in result


class TestFormatSegmentContexts:
    def test_overlapping_contexts_merged(self):
        shared = _make_segment(text="shared")
        other = _make_segment(text="other")
        ctx1 = [shared, other]
        ctx2 = [shared]
        result = format_segment_contexts([ctx1, ctx2])
        # Single component since they share `shared`; no separator.
        assert "\n\n" not in result
        assert "shared" in result
        assert "other" in result

    def test_disconnected_contexts_separated(self):
        seg_a = _make_segment(text="alpha")
        seg_b = _make_segment(
            text="beta",
            timestamp=datetime(2026, 2, 1, 10, 30, tzinfo=timezone.utc),
        )
        result = format_segment_contexts([[seg_a], [seg_b]])
        assert "\n\n" in result

    def test_components_ordered_chronologically(self):
        seg_late = _make_segment(
            context=EventMemoryMessageContext(source="late"),
            text="L",
            timestamp=datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
        )
        seg_early = _make_segment(
            context=EventMemoryMessageContext(source="early"),
            text="E",
            timestamp=datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc),
        )
        result = format_segment_contexts([[seg_late], [seg_early]])
        assert result.index("early:") < result.index("late:")


class TestPropertiesRoundTrip:
    @pytest.mark.parametrize(
        "props",
        [
            {"count": 42},
            {"name": "test"},
            {"ratio": 3.14},
            {"flag": True},
            {"ts": datetime(2026, 1, 15, tzinfo=timezone.utc)},
            {
                "count": 42,
                "name": "test",
                "ts": datetime(2026, 1, 15, tzinfo=timezone.utc),
            },
        ],
    )
    def test_round_trip(self, props):
        seg = _make_segment(properties=props)
        seg2 = EventMemorySegment.model_validate(seg.model_dump(mode="json"))
        assert seg.properties == seg2.properties
