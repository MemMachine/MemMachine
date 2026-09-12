"""Tests for event memory data type serialization round-trips."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memmachine_server.episodic_memory.event_memory.data_types import (
    ID_MAX_BYTES,
    Author,
    Derivative,
    Event,
    FormatOptions,
    Neighborhood,
    QueryHit,
    Segment,
    TextBlock,
    UnknownPart,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
    with_part,
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
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block=TextBlock(text="hello"),
            properties=SAMPLE_PROPERTIES,
        )
        seg2 = Segment.model_validate(seg.model_dump(mode="json"))
        assert seg.properties == seg2.properties
        for key in seg.properties:
            assert type(seg.properties[key]) is type(seg2.properties[key])

    def test_empty_properties(self):
        seg = Segment(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block=TextBlock(text="hello"),
        )
        seg2 = Segment.model_validate(seg.model_dump(mode="json"))
        assert seg2.properties == {}

    def test_from_code_plain_values(self):
        seg = Segment(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block=TextBlock(text="hello"),
            properties={"name": "foo", "count": 7},
        )
        assert seg.properties == {"name": "foo", "count": 7}

    def test_context_preserved(self):
        seg = Segment(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            context=with_part({}, Author(name="user")),
            block=TextBlock(text="hello"),
        )
        seg2 = Segment.model_validate(seg.model_dump(mode="json"))
        assert seg2.context.get("author") == Author(name="user")

    def test_session_and_source_round_trip(self):
        seg = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            session_id="s1",
            source_id="alice",
            block=TextBlock(text="hello"),
        )
        seg2 = Segment.model_validate(seg.model_dump(mode="json"))
        assert (seg2.session_id, seg2.source_id) == ("s1", "alice")

    def test_source_defaults_to_null_and_session_is_required(self):
        fields = {
            "uuid": uuid4(),
            "event_uuid": uuid4(),
            "index": 0,
            "offset": 0,
            "timestamp": datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            "block": {"kind": "text", "text": "hello"},
        }
        seg = Segment.model_validate({**fields, "session_id": "s1"})
        assert seg.source_id is None
        with pytest.raises(ValidationError, match="session_id"):
            Segment.model_validate(fields)
        with pytest.raises(ValidationError, match="session_id"):
            Segment.model_validate({**fields, "session_id": ""})

    def test_a_naive_timestamp_is_rejected(self):
        with pytest.raises(ValidationError, match="timestamp"):
            Segment.model_validate(
                {
                    "uuid": uuid4(),
                    "event_uuid": uuid4(),
                    "index": 0,
                    "offset": 0,
                    "timestamp": datetime(2026, 1, 15, 10, 30, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                    "session_id": "s1",
                    "block": {"block_type": "text", "text": "hello"},
                }
            )


class TestBounds:
    @pytest.mark.parametrize("field", ["session_id", "source_id"])
    def test_overlong_id_is_rejected(self, field):
        overlong = "x" * (ID_MAX_BYTES + 1)
        with pytest.raises(ValidationError, match=field):
            Event.model_validate(
                {
                    "uuid": uuid4(),
                    "timestamp": datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
                    "blocks": [{"kind": "text", "text": "hello"}],
                    field: overlong,
                }
            )

    @pytest.mark.parametrize("field", ["index", "offset"])
    def test_negative_position_is_rejected(self, field):
        with pytest.raises(ValidationError, match=field):
            Segment.model_validate(
                {
                    "uuid": uuid4(),
                    "event_uuid": uuid4(),
                    "index": 0,
                    "offset": 0,
                    "timestamp": datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
                    "block": {"kind": "text", "text": "hello"},
                    field: -1,
                }
            )

    def test_query_hit_window_is_the_seed_among_its_neighbors(self):
        def segment(seconds: int) -> Segment:
            return Segment(
                uuid=uuid4(),
                event_uuid=uuid4(),
                index=0,
                offset=0,
                timestamp=datetime(2026, 1, 15, 10, 30, seconds, tzinfo=UTC),
                session_id="s1",
                block=TextBlock(text="hello"),
            )

        before, seed, after = segment(0), segment(1), segment(2)
        hit = QueryHit(
            score=1.0,
            seed=seed,
            neighborhood=Neighborhood(before=[before], after=[after]),
        )
        assert hit.window() == [before, seed, after]


class TestEventRoundTrip:
    def test_all_property_types(self):
        evt = Event(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            blocks=[TextBlock(text="hi")],
            properties=SAMPLE_PROPERTIES,
        )
        evt2 = Event.model_validate(evt.model_dump(mode="json"))
        assert evt.properties == evt2.properties
        for key in evt.properties:
            assert type(evt.properties[key]) is type(evt2.properties[key])


class TestDerivativeRoundTrip:
    def test_all_property_types(self):
        der = Derivative(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            segment_uuid=uuid4(),
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block_kind="text",
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
            "block": {"kind": "text", "text": "hi"},
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
            "block": {"kind": "text", "text": "hi"},
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
            "block": {"kind": "text", "text": "hi"},
            "properties": {
                "dt": {"t": "datetime", "v": "2026-01-15T00:00:00+00:00"},
            },
        }
        with pytest.raises(ValidationError):
            Segment.model_validate(data)


class TestContextParts:
    def test_context_models_do_not_declare_index_hints(self):
        assert not hasattr(Author, "indexed_properties")

    def test_with_part_replaces_the_part_of_that_kind(self):
        context = with_part({}, Author(name="user"))
        replaced = with_part(context, Author(name="other"))
        assert context.get("author") == Author(name="user")
        assert replaced.get("author") == Author(name="other")
        assert list(replaced) == ["author"]

    def test_author_renders_its_name_and_unknown_renders_nothing(self):
        unknown = UnknownPart(kind_name="plugin", data={"x": 1})
        assert Author(name="user").render(FormatOptions()) == "user"
        assert unknown.render(FormatOptions()) is None


class TestContextAndBlockSerialization:
    def test_context_round_trip(self):
        context = with_part({}, Author(name="user"))

        serialized = encode_context(context)
        deserialized = decode_context(serialized)

        assert serialized == {"author": {"name": "user"}}
        assert deserialized == context

    def test_empty_context_round_trip(self):
        assert encode_context({}) == {}
        assert decode_context({}) == {}

    def test_unregistered_kind_round_trips_unchanged(self):
        encoded = {"author": {"name": "user"}, "plugin": {"x": 1, "y": "z"}}

        decoded = decode_context(encoded)

        assert decoded["plugin"] == UnknownPart(
            kind_name="plugin", data={"x": 1, "y": "z"}
        )
        assert decoded.get("author") == Author(name="user")
        assert encode_context(decoded) == encoded

    def test_part_that_is_not_an_object_is_rejected(self):
        with pytest.raises(TypeError, match="object"):
            decode_context({"author": "user"})

    def test_model_dump_encodes_parts_by_kind(self):
        seg = Segment(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            context=with_part({}, Author(name="user")),
            block=TextBlock(text="hello"),
        )
        dumped = seg.model_dump(mode="json")
        assert dumped["context"] == {"author": {"name": "user"}}
        assert dumped["block"] == {"kind": "text", "text": "hello"}

    def test_block_round_trip(self):
        block = TextBlock(text="hello")

        serialized = encode_block(block)
        deserialized = decode_block(serialized)

        assert deserialized == block
