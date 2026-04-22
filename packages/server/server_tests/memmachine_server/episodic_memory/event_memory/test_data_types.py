"""Tests for event memory data type serialization round-trips."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memmachine_server.episodic_memory.event_memory.data_types import (
    CitationContext,
    Content,
    Derivative,
    Event,
    MessageContext,
    Segment,
    Text,
)

SAMPLE_PROPERTIES = {
    "dt": datetime(2026, 1, 15, tzinfo=UTC),
    "n": 42,
    "x": 3.14,
    "f": True,
    "s": "hello",
}


class TestSegmentRoundTrip:
    def test_all_property_types(self):
        seg = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block=Text(text="hello"),
            properties=SAMPLE_PROPERTIES,
        )
        seg2 = Segment.model_validate(seg.model_dump(mode="json"))
        assert seg.properties == seg2.properties
        for key in seg.properties:
            assert type(seg.properties[key]) is type(seg2.properties[key])

    def test_empty_properties(self):
        seg = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block=Text(text="hello"),
        )
        seg2 = Segment.model_validate(seg.model_dump(mode="json"))
        assert seg2.properties == {}

    def test_from_code_plain_values(self):
        seg = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block=Text(text="hello"),
            properties={"name": "foo", "count": 7},
        )
        assert seg.properties == {"name": "foo", "count": 7}

    def test_context_preserved(self):
        seg = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            context=MessageContext(source="user"),
            block=Text(text="hello"),
        )
        seg2 = Segment.model_validate(seg.model_dump(mode="json"))
        assert isinstance(seg2.context, MessageContext)
        assert seg2.context.source == "user"


class TestEventRoundTrip:
    def test_all_property_types(self):
        evt = Event(
            uuid=uuid4(),
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            body=Content(items=[Text(text="hi")]),
            properties=SAMPLE_PROPERTIES,
        )
        evt2 = Event.model_validate(evt.model_dump(mode="json"))
        assert evt.properties == evt2.properties
        for key in evt.properties:
            assert type(evt.properties[key]) is type(evt2.properties[key])


class TestDerivativeRoundTrip:
    def test_all_property_types(self):
        der = Derivative(
            uuid=uuid4(),
            segment_uuid=uuid4(),
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            text="hello",
            properties=SAMPLE_PROPERTIES,
        )
        der2 = Derivative.model_validate(der.model_dump(mode="json"))
        assert der.properties == der2.properties
        for key in der.properties:
            assert type(der.properties[key]) is type(der2.properties[key])


class TestDeserializationErrors:
    def test_rejects_malformed_tagged_dict(self):
        """A dict-of-dicts without type tags should fail validation."""
        data = {
            "uuid": str(uuid4()),
            "event_uuid": str(uuid4()),
            "index": 0,
            "offset": 0,
            "timestamp": "2026-01-15T10:30:00Z",
            "block": {"type": "text", "text": "hi"},
            "properties": {"key": {"not_tagged": "value"}},
        }
        with pytest.raises(ValidationError):
            Segment.model_validate(data)

    def test_rejects_extra_keys_in_entry(self):
        data = {
            "uuid": str(uuid4()),
            "event_uuid": str(uuid4()),
            "index": 0,
            "offset": 0,
            "timestamp": "2026-01-15T10:30:00Z",
            "block": {"type": "text", "text": "hi"},
            "properties": {"n": {"t": "int", "v": 42, "extra": "junk"}},
        }
        with pytest.raises(ValidationError):
            Segment.model_validate(data)

    def test_rejects_datetime_missing_tz(self):
        data = {
            "uuid": str(uuid4()),
            "event_uuid": str(uuid4()),
            "index": 0,
            "offset": 0,
            "timestamp": "2026-01-15T10:30:00Z",
            "block": {"type": "text", "text": "hi"},
            "properties": {
                "dt": {"t": "datetime", "v": "2026-01-15T00:00:00+00:00"},
            },
        }
        with pytest.raises(ValidationError):
            Segment.model_validate(data)


class TestContextModels:
    def test_context_models_do_not_declare_index_hints(self):
        assert not hasattr(MessageContext, "indexed_properties")
        assert not hasattr(CitationContext, "indexed_properties")
