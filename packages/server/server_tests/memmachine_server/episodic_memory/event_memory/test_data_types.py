"""Tests for event memory data type serialization round-trips."""

from datetime import UTC, datetime
from typing import ClassVar, Literal
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from memmachine_server.episodic_memory.event_memory.data_types import (
    CitationContext,
    Content,
    Derivative,
    Event,
    MessageContext,
    Segment,
    Text,
    resolve_indexed_context_properties_schema,
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


class TestIndexedContextPropertiesValidation:
    """Verify resolve_indexed_context_properties_schema for Context subtypes.

    Tests construct ad-hoc subtypes in function-local scope and pass
    them directly to the validator. No global state, no registration,
    no cleanup — the classes exist only for the duration of the test.
    """

    def test_real_context_subtypes_pass(self):
        # Baseline: the module-level validation already ran at import,
        # but we re-run it explicitly here to pin the behavior.
        merged = resolve_indexed_context_properties_schema(
            [MessageContext, CitationContext]
        )
        assert merged == {
            "type": str,
            "source": str,
            "source_type": str,
            "location": str,
        }

    def test_missing_indexed_properties_is_optional(self):
        class _NoSchema(BaseModel):
            type: Literal["no_schema"] = "no_schema"
            source: str

        # No indexed_properties means no indexed fields, and the
        # merged schema is empty. The pydantic fields are still
        # stored in the segment store but do not appear in the
        # vector-store schema.
        merged = resolve_indexed_context_properties_schema([_NoSchema])
        assert merged == {}

    def test_partial_indexing_supported(self):
        class _Partial(BaseModel):
            type: Literal["partial"] = "partial"
            count: int
            rating: float

            # Only count is indexed; type and rating live only in the
            # segment store.
            indexed_properties: ClassVar[frozenset[str]] = frozenset({"count"})

        merged = resolve_indexed_context_properties_schema([_Partial])
        assert merged == {"count": int}

    def test_non_property_value_annotation_raises(self):
        class _BadAnnotation(BaseModel):
            type: Literal["bad_annotation"] = "bad_annotation"
            tags: list[str]  # not a PropertyValue

        with pytest.raises(TypeError, match="does not resolve to a PropertyValue"):
            resolve_indexed_context_properties_schema([_BadAnnotation])

    def test_non_frozenset_indexed_properties_raises(self):
        class _BadType(BaseModel):
            type: Literal["bad"] = "bad"
            source: str

            indexed_properties: ClassVar = ("type", "source")  # tuple, not frozenset

        with pytest.raises(TypeError, match="indexed_properties must be a frozenset"):
            resolve_indexed_context_properties_schema([_BadType])

    def test_indexed_properties_unknown_field_raises(self):
        class _Drift(BaseModel):
            type: Literal["drift"] = "drift"

            indexed_properties: ClassVar[frozenset[str]] = frozenset(
                {"type", "phantom"}  # phantom not declared as a pydantic field
            )

        with pytest.raises(TypeError, match="not present in Pydantic model_fields"):
            resolve_indexed_context_properties_schema([_Drift])

    def test_cross_subtype_conflict_raises(self):
        class _Conflict(BaseModel):
            type: Literal["conflict"] = "conflict"
            source: int  # MessageContext annotates source: str

            indexed_properties: ClassVar[frozenset[str]] = frozenset({"type", "source"})

        # Both subtypes index `source` but their annotations resolve
        # to different PropertyValue types (MessageContext: str,
        # _Conflict: int).
        with pytest.raises(
            TypeError,
            match="indexed with conflicting PropertyValue types",
        ):
            resolve_indexed_context_properties_schema([MessageContext, _Conflict])

    def test_cross_subtype_annotation_mismatch_raises(self):
        class _Indexes(BaseModel):
            type: Literal["indexes"] = "indexes"
            count: int

            indexed_properties: ClassVar[frozenset[str]] = frozenset({"type", "count"})

        class _ShadowsWithStr(BaseModel):
            type: Literal["shadows_with_str"] = "shadows_with_str"
            # Has the same field name as _Indexes but with a different
            # annotation. _ShadowsWithStr does NOT index `count`, but
            # its pydantic annotation still has to match the indexed
            # type across subtypes — otherwise a record from
            # _ShadowsWithStr couldn't fit in the indexed column.
            count: str

            indexed_properties: ClassVar[frozenset[str]] = frozenset({"type"})

        with pytest.raises(TypeError, match=r"count.*resolves to str"):
            resolve_indexed_context_properties_schema([_Indexes, _ShadowsWithStr])

    def test_cross_subtype_agreement_passes(self):
        class _Agree(BaseModel):
            type: Literal["agree"] = "agree"
            source: str  # matches MessageContext.source: str

            indexed_properties: ClassVar[frozenset[str]] = frozenset({"type", "source"})

        merged = resolve_indexed_context_properties_schema([MessageContext, _Agree])
        assert merged["source"] is str

    def test_subtype_with_no_overlap_passes(self):
        class _Standalone(BaseModel):
            type: Literal["standalone"] = "standalone"
            unique_field: int

            indexed_properties: ClassVar[frozenset[str]] = frozenset(
                {"type", "unique_field"}
            )

        merged = resolve_indexed_context_properties_schema(
            [MessageContext, _Standalone]
        )
        assert merged == {
            "type": str,
            "source": str,
            "unique_field": int,
        }
