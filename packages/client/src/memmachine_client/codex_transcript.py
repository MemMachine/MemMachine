"""Read a Codex rollout file into bodies for the v1 events API.

Codex records a session as the JSONL rollout file the `Stop` hook names,
under `$CODEX_HOME/sessions/<year>/<month>/<day>/rollout-<time>-<session
id>.jsonl`. Every line is one record, `{"timestamp", "type", "payload"}`,
and the records of type `response_item` are the conversation itself:
messages, function calls and their outputs. The `event_msg` records
restate those for the interface and are left behind, so nothing is
remembered twice.

A rollout record carries no id of its own, so an event is held under a
uuid derived from the session and the record's index in the file, which
makes a second read of the same record the same event.

A message's text is written as it was; the blocks the server holds as
one segment each -- a tool result, thinking and an injected passage --
are capped at `ONE_SEGMENT_MAX_BYTES` and say where they were cut.

A `reasoning` record is written down as thinking wherever it carries
text, which is what its `summary` and `content` hold; its
`encrypted_content` is opaque rather than text and is never posted. The
rollouts written on this machine carry an empty `summary`, a null
`content` and an encrypted body, so such a record produces no event.
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

_SOURCE_ID = "codex"

# Text Codex puts in the conversation rather than the user typing it,
# recognized by the tag that opens it. Every other tag on the developer
# channel is injected too, under `other`.
_INJECTED_TAG_SOURCES: dict[str, str] = {
    "apps_instructions": "other",
    "collaboration_mode": "other",
    "context_window_guidance": "other",
    "current_time_reminder": "reminder",
    "environment_context": "other",
    "environments_instructions": "other",
    "multi_agent_mode": "other",
    "permissions": "other",
    "personality_spec": "other",
    "plugins_instructions": "other",
    "skills_instructions": "skill",
    "startup_context": "other",
    "subagent_notification": "other",
    "turn_aborted": "other",
    "user_instructions": "other",
    "user_shell_command": "command",
}

# What an output is named when no call in the file carries its call
# id. A block names a tool in one word or another, so this says the
# rollout carried no name rather than leaving it empty.
_UNNAMED_TOOL = "unknown"


def read_entries(
    transcript_path: Path,
    *,
    session_id: str,
    properties: Mapping[str, str],
    start_offset: int,
    start_index: int,
) -> Iterator[tuple[list[dict[str, Any]], int, int]]:
    """Yield each record's events, the offset past it and its index.

    A record that carries nothing to remember yields an empty list, so a
    reader that ends on such records still moves its mark. A line the
    writer has not finished is left for the next read.

    Args:
        transcript_path: The rollout file the `Stop` hook named.
        session_id: The session the hook reported, which every event of
            this rollout belongs to.
        properties: Properties every event carries, such as the agent and
            the project.
        start_offset: The byte offset to read from, 0 for a whole file.
        start_index: The index of the record at that offset, which is
            what the events read from there are named after.
    """
    reader = _RecordReader(session_id=session_id, properties=dict(properties))
    with transcript_path.open("rb") as stream:
        stream.seek(start_offset)
        offset = start_offset
        index = start_index
        for raw_line in stream:
            if not raw_line.endswith(b"\n"):
                break
            offset += len(raw_line)
            record = _parsed_record(raw_line)
            events = [] if record is None else reader.events_of(record, index)
            index += 1
            yield events, offset, index


@dataclass
class _RecordReader:
    """Builds the event bodies of one rollout, record by record.

    A tool output names its tool by the call it answers, so the reader
    keeps the name of every call it has read under the call id the
    output carries.
    """

    session_id: str
    properties: dict[str, str]
    tool_names: dict[str, str] = field(default_factory=dict)

    def events_of(self, record: dict[str, Any], index: int) -> list[dict[str, Any]]:
        """The events one record produces, at most one.

        The session's own metadata and the turn's settings carry nothing
        that happened, and the `event_msg` records restate the
        conversation for the interface, so neither is written down.
        """
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return []
        if record.get("type") == "compacted":
            summary = str(payload.get("message", ""))
            part = (
                (
                    "user",
                    {
                        "kind": "injected",
                        "text": bounded_text(summary),
                        "source": "compaction",
                    },
                )
                if summary.strip()
                else None
            )
        elif record.get("type") == "response_item":
            part = self._response_part(payload)
        else:
            part = None
        if part is None:
            return []
        return [self._body(record, index, *part)]

    def _response_part(
        self, payload: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        """The author and block of the event a conversation record produces."""
        payload_type = payload.get("type")
        if payload_type == "message":
            return _message_part(payload)
        if payload_type in {"function_call", "custom_tool_call"}:
            name = str(payload.get("name", ""))
            self.tool_names[str(payload.get("call_id", ""))] = name
            given = payload.get(
                "arguments" if payload_type == "function_call" else "input"
            )
            return (
                "assistant",
                {"kind": "tool_call", "name": name, "input": _call_input(given)},
            )
        if payload_type == "reasoning":
            return _reasoning_part(payload)
        if payload_type in {"function_call_output", "custom_tool_call_output"}:
            output, failed = _output_text(payload.get("output"))
            return (
                "user",
                {
                    "kind": "tool_result",
                    "name": self.tool_names.get(
                        str(payload.get("call_id", "")), _UNNAMED_TOOL
                    ),
                    "output": bounded_text(output),
                    "error": failed,
                },
            )
        return None

    def _body(
        self,
        record: dict[str, Any],
        index: int,
        author: str,
        block: dict[str, Any],
    ) -> dict[str, Any]:
        """The event body of one record's block."""
        properties = dict(self.properties)
        if block["kind"] in {"tool_call", "tool_result"}:
            properties["tool_name"] = block["name"]
        body: dict[str, Any] = {
            "id": str(uuid5(_session_namespace(self.session_id), str(index))),
            "session_id": self.session_id,
            "source_id": _SOURCE_ID,
            "context": {"author": {"name": author}},
            "blocks": [block],
            "properties": properties,
        }
        timestamp = _timestamp(record.get("timestamp"))
        if timestamp is not None:
            body["timestamp"] = timestamp
        return body


def _reasoning_part(payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The author and block of a reasoning record, None where it holds no text.

    A record carries its reasoning in `summary`, in `content`, or in
    neither: `encrypted_content` is opaque rather than text, so a record
    holding only that is nothing to write down.
    """
    parts = (
        _message_text(payload.get("summary")),
        _message_text(payload.get("content")),
    )
    text = "\n".join(part for part in parts if part)
    if not text.strip():
        return None
    return ("assistant", {"kind": "thinking", "text": bounded_text(text)})


def _message_part(payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The author and block of a message record.

    The assistant's messages and what the user typed are text; the
    developer channel and the tagged passages on the user channel are
    what Codex loaded into the session, and carry where they came from.
    """
    text = _message_text(payload.get("content"))
    if not text.strip():
        return None
    role = payload.get("role")
    if role == "assistant":
        return ("assistant", {"kind": "text", "text": text})
    tag = _opening_tag(text)
    source = _INJECTED_TAG_SOURCES.get(tag)
    if role == "developer" and source is None:
        source = "other"
    if source is None:
        return ("user", {"kind": "text", "text": text})
    return ("user", {"kind": "injected", "text": bounded_text(text), "source": source})


def _message_text(content: object) -> str:
    """The text of a message's content blocks, whatever their kind."""
    if isinstance(content, str):
        return content
    parts = [
        str(block.get("text", ""))
        for block in (content if isinstance(content, list) else [])
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(part for part in parts if part)


def _opening_tag(text: str) -> str:
    """The name of the tag a passage opens with, empty for other text."""
    head = text.lstrip()
    if not head.startswith("<"):
        return ""
    end = head.find(">")
    if end < 0:
        return ""
    words = head[1:end].split()
    return words[0].lower() if words else ""


def _call_input(given: object) -> dict[str, Any]:
    """A tool call's arguments as an object.

    A function call carries them as a JSON object in a string; a custom
    tool call carries a body of its own, which is kept under `input`.
    """
    if isinstance(given, dict):
        return given
    if isinstance(given, str):
        try:
            parsed = json.loads(given)
        except json.JSONDecodeError:
            return {"input": given}
        return parsed if isinstance(parsed, dict) else {"input": given}
    return {}


def _output_text(output: object) -> tuple[str, bool]:
    """A tool output's text, and whether the tool reported a failure.

    Codex wraps some outputs in `{"output", "metadata"}`, where the
    metadata's exit code is the tool's own verdict; an output that
    carries no verdict is not a failure.
    """
    if isinstance(output, list):
        return _message_text(output), False
    if not isinstance(output, str):
        return json.dumps(output, ensure_ascii=False), False
    try:
        wrapper = json.loads(output)
    except json.JSONDecodeError:
        return output, False
    if not isinstance(wrapper, dict) or "output" not in wrapper:
        return output, False
    metadata = wrapper.get("metadata")
    exit_code = metadata.get("exit_code") if isinstance(metadata, dict) else None
    return str(wrapper["output"]), isinstance(exit_code, int) and exit_code != 0


def _parsed_record(raw_line: bytes) -> dict[str, Any] | None:
    """One rollout line as a record, None for a line that holds none."""
    text = raw_line.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def _session_namespace(session_id: str) -> UUID:
    """The namespace a session's records are named under.

    A session id that is a uuid is the namespace itself; any other id is
    first named under the URL namespace, so every session has one.
    """
    try:
        return UUID(session_id)
    except ValueError:
        return uuid5(NAMESPACE_URL, session_id)


def _timestamp(raw_timestamp: object) -> str | None:
    """A record's time with its offset, None when it carries none.

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
