"""Tests for event memory data type serialization round-trips."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memmachine_server.episodic_memory.event_memory.data_types import (
    ID_MAX_BYTES,
    TOOL_NAME_MAX_BYTES,
    Author,
    Context,
    DateTimeFormat,
    Derivative,
    Event,
    InjectedBlock,
    Neighborhood,
    QueryHit,
    Segment,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    UnknownPart,
    decode_block,
    encode_block,
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
            context=Context(Author(name="user")),
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
    def test_round_trip(self):
        der = Derivative(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            segment_uuid=uuid4(),
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            block_kind="text",
            text="hello",
        )
        assert Derivative.model_validate(der.model_dump(mode="json")) == der


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
        context = Context(Author(name="user"))
        replaced = context.with_part(Author(name="other"))
        assert context.get("author") == Author(name="user")
        assert replaced.get("author") == Author(name="other")
        assert list(replaced) == [Author(name="other")]

    def test_two_parts_of_one_kind_are_rejected(self):
        with pytest.raises(ValueError, match="author"):
            Context(Author(name="user"), Author(name="other"))

    def test_author_renders_its_name_and_unknown_renders_nothing(self):
        unknown = UnknownPart(kind_name="plugin", data={"x": 1})
        assert Author(name="user").render(DateTimeFormat()) == "user"
        assert unknown.render(DateTimeFormat()) is None


class TestContextAndBlockSerialization:
    def test_context_round_trip(self):
        context = Context(Author(name="user"))

        serialized = context.encode()
        deserialized = Context.decode(serialized)

        assert serialized == {"author": {"name": "user"}}
        assert deserialized == context

    def test_empty_context_round_trip(self):
        assert Context().encode() == {}
        assert Context.decode({}) == Context()

    def test_unregistered_kind_round_trips_unchanged(self):
        encoded = {"author": {"name": "user"}, "plugin": {"x": 1, "y": "z"}}

        decoded = Context.decode(encoded)

        assert decoded.get("plugin") == UnknownPart(
            kind_name="plugin", data={"x": 1, "y": "z"}
        )
        assert decoded.get("author") == Author(name="user")
        assert decoded.encode() == encoded

    def test_part_that_is_not_an_object_is_rejected(self):
        with pytest.raises(TypeError, match="object"):
            Context.decode({"author": "user"})

    def test_model_dump_encodes_parts_by_kind(self):
        seg = Segment(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            context=Context(Author(name="user")),
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


class TestCaptureBlockKinds:
    """The kinds a coding agent's transcript produces, beside the message kind."""

    @pytest.mark.parametrize(
        "block",
        [
            TextBlock(text="run the tests"),
            ThinkingBlock(text="The suite is red; read the failure first."),
            ToolCallBlock(
                name="Bash",
                input={"command": "pytest -q", "timeout": 600, "env": {"CI": True}},
            ),
            ToolResultBlock(name="Bash", output="1 failed", error=True),
            ToolResultBlock(name="Bash", output="4 passed"),
            InjectedBlock(text="Run the gates before committing.", source="hook"),
        ],
    )
    def test_a_kind_round_trips_through_the_codec(self, block):
        assert decode_block(encode_block(block)) == block

    def test_the_encoded_form_carries_the_kind_and_its_fields(self):
        assert encode_block(ToolCallBlock(name="Bash", input={"command": "ls"})) == {
            "kind": "tool_call",
            "name": "Bash",
            "input": {"command": "ls"},
        }
        assert encode_block(
            ToolResultBlock(name="Bash", output="4 passed", error=False)
        ) == {
            "kind": "tool_result",
            "name": "Bash",
            "output": "4 passed",
            "error": False,
        }
        assert encode_block(InjectedBlock(text="Be brief.", source="reminder")) == {
            "kind": "injected",
            "text": "Be brief.",
            "source": "reminder",
        }
        assert encode_block(ThinkingBlock(text="Read the failure first.")) == {
            "kind": "thinking",
            "text": "Read the failure first.",
        }

    def test_a_call_renders_its_tool_and_its_input_on_one_line(self):
        rendered = ToolCallBlock(
            name="Bash", input={"command": "pytest -q\nruff check"}
        ).render(DateTimeFormat())
        assert rendered == 'tool_call Bash: {"command":"pytest -q\\nruff check"}'
        assert "\n" not in rendered

    def test_a_result_renders_its_tool_and_its_output(self):
        assert (
            ToolResultBlock(name="Bash", output="4 passed").render(DateTimeFormat())
            == "tool_result Bash: 4 passed"
        )

    def test_a_failed_result_renders_an_error_marker(self):
        assert ToolResultBlock(
            name="Bash", output="No such file or directory", error=True
        ).render(DateTimeFormat()) == (
            "tool_result Bash [error]: No such file or directory"
        )

    def test_reasoning_renders_behind_its_kind(self):
        assert (
            ThinkingBlock(text="Read the failure first.").render(DateTimeFormat())
            == "thinking: Read the failure first."
        )

    def test_injected_text_renders_its_source_and_its_text(self):
        assert (
            InjectedBlock(text="Summary of the session.", source="compaction").render(
                DateTimeFormat()
            )
            == "injected compaction: Summary of the session."
        )

    def test_a_result_is_a_success_unless_it_says_otherwise(self):
        assert ToolResultBlock(name="Bash", output="4 passed").error is False

    @pytest.mark.parametrize("kind", [ToolCallBlock, ToolResultBlock])
    def test_an_empty_tool_name_is_rejected(self, kind):
        with pytest.raises(ValidationError, match="name"):
            decode_block({**_minimal(kind), "name": ""})

    @pytest.mark.parametrize("kind", [ToolCallBlock, ToolResultBlock])
    def test_an_overlong_tool_name_is_rejected(self, kind):
        overlong = "t" * (TOOL_NAME_MAX_BYTES + 1)
        with pytest.raises(ValidationError, match="name"):
            decode_block({**_minimal(kind), "name": overlong})

    def test_an_unlisted_injection_source_is_rejected(self):
        with pytest.raises(ValidationError, match="source"):
            decode_block({"kind": "injected", "text": "hi", "source": "typed"})

    def test_an_unregistered_kind_is_rejected(self):
        with pytest.raises(ValidationError, match="kind"):
            decode_block({"kind": "hologram", "text": "hi"})


def _minimal(kind: type[ToolCallBlock] | type[ToolResultBlock]) -> dict:
    """The encoded form of a block of this kind, its name left to the caller."""
    if kind is ToolCallBlock:
        return {"kind": "tool_call", "name": "Bash", "input": {}}
    return {"kind": "tool_result", "name": "Bash", "output": "4 passed"}
