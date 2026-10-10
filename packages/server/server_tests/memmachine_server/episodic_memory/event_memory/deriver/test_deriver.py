"""Tests for the Deriver table: dispatch by kind, override order, envelopes."""

from datetime import UTC, datetime
from typing import override
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    Author,
    Block,
    Context,
    DateTimeFormat,
    InjectedBlock,
    Segment,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import (
    BlockDeriver,
    Deriver,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _segment(text: str = "hi", *, context=None, block: Block | None = None) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        session_id="s1",
        source_id="chat",
        context=context if context is not None else Context(),
        block=block if block is not None else TextBlock(text=text),
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
        super().__init__()
        self.calls = []

    @override
    async def derive(self, segment, block):
        self.calls.append((segment, block))
        return []


async def test_no_handler_derives_nothing():
    assert await Deriver().derive(_segment("hi")) == []


async def test_handler_of_the_kind_derives_and_the_table_builds_the_envelope():
    segment = _segment("hi", context=Context(Author(name="alice")))
    derivatives = await Deriver([_Two()]).derive(segment)
    # The table composes the text to embed: the handler's header, then
    # the derived text as one escaped token.
    assert [d.text for d in derivatives] == [
        '[Thursday, January 15, 2026] alice: "hi!"',
        '[Thursday, January 15, 2026] alice: "hi?"',
    ]
    for derivative in derivatives:
        assert derivative.segment_uuid == segment.uuid
        assert derivative.timestamp == segment.timestamp
        assert derivative.session_id == "s1"
        assert derivative.source_id == "chat"
        assert derivative.block_kind == "text"
    assert len({d.uuid for d in derivatives}) == 2


async def test_later_handler_replaces_earlier_for_its_kind():
    derivatives = await Deriver([_Two(), _Const()]).derive(_segment("hi"))
    assert [d.text for d in derivatives] == ['[Thursday, January 15, 2026] "one"']


async def test_the_handler_owns_the_format_of_what_it_embeds():
    bare = _Const(DateTimeFormat(date_style=None, time_style=None))
    [derivative] = await Deriver([bare]).derive(_segment("hi"))
    assert derivative.text == '"one"'


async def test_handler_receives_the_segment_and_its_block():
    recording = _Recording()
    segment = _segment("hi")
    await Deriver([recording]).derive(segment)
    assert recording.calls == [(segment, segment.block)]


async def test_the_capture_kinds_derive_nothing_under_the_text_deriver():
    deriver = Deriver([WholeTextDeriver()])
    for block in (
        ThinkingBlock(text="The suite is red."),
        ToolCallBlock(name="Bash", input={"command": "pytest -q"}),
        ToolResultBlock(name="Bash", output="4 passed"),
        InjectedBlock(text="Run the gates.", source="hook"),
    ):
        assert await deriver.derive(_segment(block=block)) == []
    assert await deriver.derive(_segment("run the tests")) != []
