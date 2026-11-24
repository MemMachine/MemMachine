"""What both agents' transcript readers hold to.

A tool result and an injected passage are each one segment on the
server, however long they are, and an expansion step is counted in
segments: one long result would spend a whole step on itself. So a
reader caps what those blocks carry, and says where it cut.
"""

from __future__ import annotations

ONE_SEGMENT_MAX_BYTES = 8192
"""Bound on the text of a block the server holds as one segment."""


def bounded_text(text: str) -> str:
    """Text capped to what one segment carries, saying what it cut.

    Text within the cap is returned as it came. Longer text is cut at
    the character boundary at or before the cap, and one line naming the
    bytes kept and the bytes there were follows it, so a reader of the
    memory knows the passage goes on. The marker is the reader's, not
    the agent's, so it is written after the cap rather than inside it.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= ONE_SEGMENT_MAX_BYTES:
        return text
    kept = encoded[:ONE_SEGMENT_MAX_BYTES].decode("utf-8", errors="ignore")
    return f"{kept}\n[truncated: {len(kept.encode('utf-8'))} of {len(encoded)} bytes]"
