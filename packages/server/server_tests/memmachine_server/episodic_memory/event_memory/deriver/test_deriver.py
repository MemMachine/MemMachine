"""Tests for the Deriver table: dispatch by kind, override order, envelopes."""

from datetime import UTC, datetime
from typing import override
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    Author,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import (
    BlockDeriver,
    Deriver,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _segment(text: str, *, context=None, properties=None) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        session_id="s1",
        source_id="chat",
        context=context if context is not None else {},
        block=TextBlock(text=text),
        properties=properties or {},
    )


class _Const(BlockDeriver[TextBlock]):
    kind = "text"

    @override
    async def derive(self, segment, block):
        return ["one"]


class _Two(BlockDeriver[TextBlock]):
    kind = "text"

    @override
    async def derive(self, segment, block):
        return [block.text + "!", block.text + "?"]


class _Recording(BlockDeriver[TextBlock]):
    kind = "text"

    def __init__(self) -> None:
        self.calls = []

    @override
    async def derive(self, segment, block):
        self.calls.append((segment, block))
        return []


async def test_no_handler_derives_nothing():
    assert await Deriver().derive(_segment("hi")) == []


async def test_handler_of_the_kind_derives_and_the_table_builds_the_envelope():
    segment = _segment(
        "hi", context={"author": Author(name="alice")}, properties={"k": "v"}
    )
    derivatives = await Deriver([_Two()]).derive(segment)
    assert [d.text for d in derivatives] == ["hi!", "hi?"]
    for derivative in derivatives:
        assert derivative.segment_uuid == segment.uuid
        assert derivative.timestamp == segment.timestamp
        assert derivative.session_id == "s1"
        assert derivative.source_id == "chat"
        assert derivative.context == segment.context
        assert derivative.block_kind == "text"
        assert derivative.properties == {"k": "v"}
    assert len({d.uuid for d in derivatives}) == 2


async def test_later_handler_replaces_earlier_for_its_kind():
    derivatives = await Deriver([_Two(), _Const()]).derive(_segment("hi"))
    assert [d.text for d in derivatives] == ["one"]


async def test_handler_receives_the_segment_and_its_block():
    recording = _Recording()
    segment = _segment("hi")
    await Deriver([recording]).derive(segment)
    assert recording.calls == [(segment, segment.block)]
