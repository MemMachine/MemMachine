"""Read a Claude Code transcript into bodies for the v1 events API.

The transcript is the JSONL file the `Stop` hook names, one entry per
line, at `~/.claude/projects/<directory slug>/<session id>.jsonl`. Every
message, tool call, tool result and injected passage in it becomes one
event body, in transcript order, and the reader starts at a byte offset
so a session is read once.

Every entry the file holds since the mark is posted. Only `text` blocks
reach the search surface, so a tool call, its result, the assistant's
thinking and injected text stay on the timeline and are read by
expanding from a message. A message's text is written as it was; the
blocks the server holds as one segment each -- a tool result, thinking
and an injected passage -- are capped at `ONE_SEGMENT_MAX_BYTES` and say
where they were cut.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from memmachine_client.coding_agent_transcript import bounded_text

_SOURCE_ID = "claude-code"

# A session whose entries carry no session id of their own, and a
# subagent run whose entries carry no agent id, are named under this.
_SUBAGENT_NAMESPACE = uuid5(NAMESPACE_URL, "memmachine-client/claude-code/subagent")

# Text that entered the conversation without being typed, recognized by
# what opens it. Only the head is examined, so a message that quotes one
# of these markers further in stays a message.
_INJECTED_HEADS: tuple[tuple[str, str], ...] = (
    ("<system-reminder", "reminder"),
    ("<task-notification", "other"),
    ("<local-command", "command"),
    ("<command-name", "command"),
    ("<command-message", "command"),
    ("<command-args", "command"),
    ("caveat: the messages below", "command"),
    ("this session is being continued", "compaction"),
    ("the following skills were invoked", "skill"),
    ("userpromptsubmit hook additional context", "hook"),
)
_INJECTED_HEAD_CHARACTERS = 400

# What a result is named when the transcript does not say which tool
# returned it. A block names a tool in one word or another, so this
# says the transcript carried no name rather than leaving it empty.
_UNNAMED_TOOL = "unknown"


def read_entries(
    transcript_path: Path,
    *,
    session_id: str,
    properties: Mapping[str, str],
    start_offset: int,
    start_index: int,
) -> Iterator[tuple[list[dict[str, Any]], int, int]]:
    """Yield each entry's events, the offset past it and its index.

    An entry that carries nothing to remember yields an empty list, so a
    reader that ends on such entries still moves its mark. A line the
    writer has not finished is left for the next read, and the offset
    never lands inside one.

    Args:
        transcript_path: The JSONL file the `Stop` hook named.
        session_id: The session the hook reported, used for entries that
            name no session of their own and as the parent of subagents.
        properties: Properties every event carries, such as the agent and
            the project.
        start_offset: The byte offset to read from, 0 for a whole file.
        start_index: The index of the entry at that offset, which the
            reader counts on so that a mark names both.
    """
    reader = _EntryReader(
        session_id=session_id,
        properties=dict(properties),
        transcript_path=transcript_path,
        start_offset=start_offset,
    )
    with transcript_path.open("rb") as stream:
        stream.seek(start_offset)
        offset = start_offset
        index = start_index
        for raw_line in stream:
            if not raw_line.endswith(b"\n"):
                break
            offset += len(raw_line)
            index += 1
            entry = _parsed_entry(raw_line)
            yield ([] if entry is None else reader.events_of(entry)), offset, index


@dataclass
class _EntryReader:
    """Builds the event bodies of one transcript, entry by entry.

    A tool result names its tool by the call it answers, so the reader
    keeps the names of the calls it has read, and reads the entries
    before its start offset once if a result arrives without one.
    """

    session_id: str
    properties: dict[str, str]
    transcript_path: Path
    start_offset: int
    tool_names: dict[str, str] = field(default_factory=dict)
    launching_entry_uuid: str = ""
    read_earlier_calls: bool = False

    def events_of(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        """The events one entry produces, in the order they happened."""
        entry_uuid = _entry_uuid(entry)
        if entry_uuid is None or entry.get("type") not in {"user", "assistant"}:
            return []
        message = entry.get("message")
        if not entry.get("isSidechain"):
            self.launching_entry_uuid = str(entry.get("uuid", ""))
        parts: list[tuple[str, dict[str, Any]]]
        if entry.get("isCompactSummary"):
            summary = _entry_text(message)
            parts = (
                [
                    (
                        "user",
                        {
                            "kind": "injected",
                            "text": bounded_text(summary),
                            "source": "compaction",
                        },
                    )
                ]
                if summary.strip()
                else []
            )
        elif entry.get("type") == "user":
            parts = self._user_blocks(entry, message)
        else:
            parts = self._assistant_blocks(message)
        return self._bodies(entry, entry_uuid, parts)

    def _user_blocks(
        self, entry: dict[str, Any], message: object
    ) -> list[tuple[str, dict[str, Any]]]:
        """The author and block of each event a user entry produces.

        A user entry is either something the user typed, something the
        host put in the session around them, or the results of the tool
        calls the assistant made.
        """
        content = message.get("content") if isinstance(message, dict) else None
        results = [
            block
            for block in (content if isinstance(content, list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        if results:
            return [("user", self._result_block(block)) for block in results]
        text = _entry_text(message)
        if not text.strip():
            return []
        source = _injected_source(text)
        if source is None and entry.get("isMeta"):
            # The host says this is something it loaded in, without
            # saying what kind, and the text carries no marker either.
            source = "other"
        if source is None:
            return [("user", {"kind": "text", "text": text})]
        return [
            (
                "user",
                {"kind": "injected", "text": bounded_text(text), "source": source},
            )
        ]

    def _result_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """The `tool_result` block of one result in a user entry."""
        return {
            "kind": "tool_result",
            "name": self._tool_name(str(block.get("tool_use_id", ""))),
            "output": bounded_text(_result_output(block.get("content"))),
            "error": bool(block.get("is_error")),
        }

    def _tool_name(self, tool_use_id: str) -> str:
        """The name of the call a result answers.

        The call is read before its result, so it is already known unless
        the mark fell between the two; then the entries before the mark
        are read once for their calls. A call that is in no entry of the
        file leaves the result named `unknown`.
        """
        name = self.tool_names.get(tool_use_id)
        if name is not None:
            return name
        if self.read_earlier_calls or self.start_offset == 0:
            return _UNNAMED_TOOL
        self.read_earlier_calls = True
        self.tool_names.update(
            _tool_names_before(self.transcript_path, self.start_offset)
        )
        return self.tool_names.get(tool_use_id, _UNNAMED_TOOL)

    def _assistant_blocks(self, message: object) -> list[tuple[str, dict[str, Any]]]:
        """The author and block of each event an assistant entry produces.

        A turn's thinking is written down where the transcript carries
        it, in the order it happened, before the text it led to.
        """
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return [("assistant", {"kind": "text", "text": content})] if content else []
        blocks: list[tuple[str, dict[str, Any]]] = []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and str(block.get("text", "")).strip():
                blocks.append(("assistant", {"kind": "text", "text": block["text"]}))
            elif (
                block.get("type") == "thinking"
                and str(block.get("thinking", "")).strip()
            ):
                blocks.append(
                    (
                        "assistant",
                        {
                            "kind": "thinking",
                            "text": bounded_text(str(block["thinking"])),
                        },
                    )
                )
            elif block.get("type") == "tool_use":
                name = str(block.get("name", ""))
                self.tool_names[str(block.get("id", ""))] = name
                given = block.get("input")
                blocks.append(
                    (
                        "assistant",
                        {
                            "kind": "tool_call",
                            "name": name,
                            "input": given if isinstance(given, dict) else {},
                        },
                    )
                )
        return blocks

    def _bodies(
        self,
        entry: dict[str, Any],
        entry_uuid: UUID,
        parts: list[tuple[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """The event bodies of one entry's blocks.

        The first event of an entry is held under the entry's own uuid,
        and a second event of the same entry under a uuid derived from
        it, so every event of a re-read transcript keeps its identity.
        """
        session_id = self._event_session(entry)
        timestamp = _timestamp(entry.get("timestamp"))
        bodies: list[dict[str, Any]] = []
        for index, (author, block) in enumerate(parts):
            properties = dict(self.properties)
            if session_id != self.session_id:
                properties["parent_session"] = self.session_id
            if block["kind"] in {"tool_call", "tool_result"}:
                properties["tool_name"] = block["name"]
            body: dict[str, Any] = {
                "id": str(entry_uuid if index == 0 else uuid5(entry_uuid, str(index))),
                "session_id": session_id,
                "source_id": _SOURCE_ID,
                "context": {"author": {"name": author}},
                "blocks": [block],
                "properties": properties,
            }
            if timestamp is not None:
                body["timestamp"] = timestamp
            bodies.append(body)
        return bodies

    def _event_session(self, entry: dict[str, Any]) -> str:
        """The session an entry belongs to.

        A subagent runs in a session of its own, named by the agent id
        the entry carries; a subagent entry without one is named after
        the turn that launched it, so one subagent run is one session.
        """
        agent_id = entry.get("agentId")
        if isinstance(agent_id, str) and agent_id:
            return agent_id
        if entry.get("isSidechain"):
            launched_by = self.launching_entry_uuid or self.session_id
            return str(uuid5(_SUBAGENT_NAMESPACE, f"{self.session_id}:{launched_by}"))
        entry_session = entry.get("sessionId")
        if isinstance(entry_session, str) and entry_session:
            return entry_session
        return self.session_id


def _tool_names_before(transcript_path: Path, offset: int) -> dict[str, str]:
    """The name of every tool call in the entries before a byte offset."""
    names: dict[str, str] = {}
    with transcript_path.open("rb") as stream:
        for raw_line in stream:
            offset -= len(raw_line)
            if offset < 0:
                break
            entry = _parsed_entry(raw_line)
            message = entry.get("message") if entry is not None else None
            content = message.get("content") if isinstance(message, dict) else None
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    names[str(block.get("id", ""))] = str(block.get("name", ""))
    return names


def _parsed_entry(raw_line: bytes) -> dict[str, Any] | None:
    """One transcript line as an entry, None for a line that holds none."""
    text = raw_line.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        entry = json.loads(text)
    except json.JSONDecodeError:
        return None
    return entry if isinstance(entry, dict) else None


def _entry_uuid(entry: dict[str, Any]) -> UUID | None:
    """The uuid an entry is held under, None for an entry carrying none.

    Every message entry Claude Code writes carries one; an entry without
    one has no identity to deduplicate against and is left behind.
    """
    raw_uuid = entry.get("uuid")
    if not isinstance(raw_uuid, str):
        return None
    try:
        return UUID(raw_uuid)
    except ValueError:
        return None


def _entry_text(message: object) -> str:
    """The text of a message, whether it holds a string or text blocks."""
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    parts = [
        str(block.get("text", ""))
        for block in (content if isinstance(content, list) else [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part)


def _result_output(content: object) -> str:
    """The text of a tool result.

    A block of another kind is written as its kind in brackets, which
    keeps image bytes out of the memory while saying what was returned.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        else:
            parts.append(f"[{block.get('type', 'unknown')}]")
    return "\n".join(parts)


def _injected_source(text: str) -> str | None:
    """Where a passage came from, None for something the user typed."""
    head = text[:_INJECTED_HEAD_CHARACTERS].lstrip().lower()
    for marker, source in _INJECTED_HEADS:
        if head.startswith(marker):
            return source
    return None


def _timestamp(raw_timestamp: object) -> str | None:
    """An entry's time with its offset, None when it carries none.

    A time without an offset says nothing about when it happened, so it
    is left out and the server stamps the event with its own.
    """
    if not isinstance(raw_timestamp, str) or not raw_timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.isoformat() if parsed.tzinfo is not None else None
