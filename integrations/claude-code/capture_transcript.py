#!/usr/bin/env python3
"""Capture a Claude Code session's transcript into MemMachine's timeline.

Registered as a `Stop` hook, this runs once at the end of every turn and posts
whatever the transcript has gained since it last ran. It imports nothing but
the standard library: a hook runs as a subprocess on every turn, so its import
cost is paid on every turn, and the work here is reading a file and making one
HTTP request.

What gets stored
----------------

Everything on the transcript, in order: user and assistant messages, the tool
calls in between, and what those calls returned. Reconstructing "what
happened" means replaying a contiguous stretch of that, so it all has to be on
one timeline.

Only messages are worth *searching*, though, so each entry carries a `source`
in its metadata and the recall side filters on it. Tool calls, their output,
and text that was injected into the turn rather than typed in it are reached
by expanding around a message instead. Embedding them would let a pasted file
outrank the sentence that asked about it, and would spend the index on blobs
nobody searches by content.

`injected` is the subtle one. A user turn carries both what the person wrote
and whatever the harness loaded in around them -- hook output, skill bodies,
system reminders, and the session's own compaction summary. Role cannot tell
them apart, so they are classified on their shape here, at capture time. The
compaction summary is the case that matters most: compaction does not start a
new session, so the summary sits among the very turns it paraphrases, and
indexing it would let a description outrank its own source.

State
-----

One file per session under `--state-dir`, holding the number of transcript
lines already sent. That is the whole of it. The transcript is append-only and
compaction continues the same file rather than forking it, so a line count
stays valid for the life of the session; a crash between posting and writing
the mark re-sends a few entries, which is why entries carry deterministic ids.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Sources that recall may rank over. Everything else is reachable by expanding
# around a message, and stays out of the index.
SEARCHABLE_SOURCES = ("user_message", "assistant_message")

# Text that arrived with a user turn without being typed by the user. Matched
# on shape because the transcript records it under the same role.
_INJECTED_PATTERNS = (
    re.compile(r"^\s*<(system-reminder|command-name|command-message|local-command)"),
    re.compile(r"^\s*<user-prompt-submit-hook>"),
    re.compile(r"^\s*Caveat: The messages below were generated"),
    re.compile(r"^\s*This session is being continued from a previous"),
)

_DEFAULT_TIMEOUT_SECONDS = 30


def classify_user_text(text: str) -> str:
    """Whether a user-role entry was typed by the user or loaded around them."""
    return (
        "injected"
        if any(pattern.search(text) for pattern in _INJECTED_PATTERNS)
        else "user_message"
    )


def _text_of(content: Any) -> str:
    """Flatten a message's content blocks into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part for part in parts if part)


def _blocks_to_entries(
    content: list[Any], timestamp: str | None
) -> Iterator[dict[str, Any]]:
    """The entries a list of content blocks contributes, one per block."""
    for block in content:
        if not isinstance(block, dict):
            continue
        match block.get("type"):
            case "text" if block.get("text", "").strip():
                yield {
                    "source": "assistant_message",
                    "producer": "assistant",
                    "text": block["text"],
                    "timestamp": timestamp,
                }
            case "thinking" if block.get("thinking", "").strip():
                yield {
                    "source": "reasoning",
                    "producer": "assistant",
                    "text": block["thinking"],
                    "timestamp": timestamp,
                }
            case "tool_use":
                yield {
                    "source": "tool_call",
                    "producer": "assistant",
                    "text": json.dumps(block.get("input", {}), ensure_ascii=False),
                    "timestamp": timestamp,
                    "tool_name": str(block.get("name", "")),
                }
            case "tool_result":
                yield {
                    "source": "tool_result",
                    "producer": "tool",
                    "text": _text_of(block.get("content")),
                    "timestamp": timestamp,
                }


def _entries_from_record(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Split one transcript record into the timeline entries it contributes."""
    message = record.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    timestamp = record.get("timestamp")

    if record.get("isCompactSummary"):
        # Kept on the timeline -- it is the one record of where the session
        # lost its context -- but never indexed.
        yield {
            "source": "injected",
            "producer": "system",
            "text": _text_of(content),
            "timestamp": timestamp,
        }
        return

    # Tool results come back under the *user* role, because the harness is
    # what replies to a tool call. Dispatching on role alone would file them
    # as things the user said.
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        yield from _blocks_to_entries(content, timestamp)
        return

    match message.get("role"):
        case "user":
            text = _text_of(content)
            if text.strip():
                yield {
                    "source": classify_user_text(text),
                    "producer": "user",
                    "text": text,
                    "timestamp": timestamp,
                }
        case "assistant" if isinstance(content, list):
            yield from _blocks_to_entries(content, timestamp)


def entries_since(transcript_path: Path, offset: int) -> tuple[list[dict], int]:
    """Read the transcript past `offset` lines, returning entries and a new mark."""
    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    for line in lines[offset:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            entries.extend(_entries_from_record(record))
    return entries, len(lines)


def _state_path(state_dir: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id) or "unknown"
    return state_dir / f"{safe}.json"


def read_mark(state_dir: Path, session_id: str) -> int:
    """How many transcript lines have already been captured for this session."""
    path = _state_path(state_dir, session_id)
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["lines"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def write_mark(state_dir: Path, session_id: str, lines: int) -> None:
    """Record the capture mark, after the entries behind it are stored."""
    state_dir.mkdir(parents=True, exist_ok=True)
    _state_path(state_dir, session_id).write_text(
        json.dumps({"lines": lines}), encoding="utf-8"
    )


def _message(entry: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Shape one timeline entry as a MemMachine message."""
    return {
        "content": entry["text"],
        "producer": entry["producer"],
        "produced_for": "assistant",
        "role": entry["producer"],
        "timestamp": entry["timestamp"],
        "metadata": {
            key: value
            for key, value in {
                "source": entry["source"],
                "session_id": session_id,
                "tool_name": entry.get("tool_name"),
            }.items()
            if value
        },
    }


def post(url: str, payload: dict[str, Any], *, timeout: float) -> None:
    """POST JSON, raising on a non-2xx response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def main() -> int:
    """Capture the current turn, or explain why it could not."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server", default=os.getenv("MEMMACHINE_URL", "http://127.0.0.1:8080")
    )
    parser.add_argument("--org-id", default=os.getenv("MM_ORG_ID", "claude-code"))
    parser.add_argument("--project-id", default=os.getenv("MM_PROJ_ID", ""))
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(
            os.getenv("MEMMACHINE_STATE_DIR", "~/.memmachine/claude-code")
        ).expanduser(),
    )
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    transcript_path = event.get("transcript_path")
    session_id = event.get("session_id") or ""
    if not transcript_path or not Path(transcript_path).exists():
        return 0

    # One project per working directory, so each repository gets its own
    # timeline and a search is scoped to the work it belongs to.
    project_id = args.project_id or re.sub(
        r"[^A-Za-z0-9_.-]+", "_", event.get("cwd") or "default"
    )

    mark = read_mark(args.state_dir, session_id)
    entries, new_mark = entries_since(Path(transcript_path), mark)
    if not entries:
        return 0

    payload = {
        "org_id": args.org_id,
        "project_id": project_id,
        "types": ["episodic"],
        "messages": [_message(entry, session_id) for entry in entries],
    }
    try:
        post(
            f"{args.server.rstrip('/')}/api/v2/memories", payload, timeout=args.timeout
        )
    except (urllib.error.URLError, OSError) as error:
        # A hook that fails must not fail the turn: the next capture re-reads
        # from the same mark and sends this turn again.
        print(f"memmachine capture skipped: {error}", file=sys.stderr)
        return 0

    write_mark(args.state_dir, session_id, new_mark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
