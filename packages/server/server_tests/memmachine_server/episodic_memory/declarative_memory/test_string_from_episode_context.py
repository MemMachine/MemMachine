"""Unit tests for ``DeclarativeMemory.string_from_episode_context``.

These tests target the static formatter directly and do not need any of
the heavyweight integration fixtures (Neo4j, embedders, rerankers) used
by ``test_declarative_memory.py``.
"""

import json
from datetime import UTC, datetime

from memmachine_server.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    Episode,
)


def _make_episode(content):
    return Episode(
        uid="ep_1",
        timestamp=datetime(2026, 1, 14, 13, 30, tzinfo=UTC),
        source="user_1",
        content_type=ContentType.MESSAGE,
        content=content,
    )


def test_ascii_content_baseline():
    """Sanity check that ASCII content still formats as before — guards
    against regressions in the timestamp/source/JSON layout."""
    result = DeclarativeMemory.string_from_episode_context(
        [_make_episode("Hello world")]
    )
    assert result == '[Wednesday, January 14, 2026 at 01:30 PM] user_1: "Hello world"\n'


def test_non_ascii_content_preserved_literally():
    """Non-ASCII characters must reach the reranker / LLM as-is, not as
    ``\\uXXXX`` escapes — escapes inflate token counts and obscure
    semantic content for the reranker."""
    result = DeclarativeMemory.string_from_episode_context(
        [_make_episode("寿司 café 🍕 Привет naïve")]
    )

    assert "寿司" in result
    assert "café" in result
    assert "🍕" in result
    assert "Привет" in result
    assert "naïve" in result
    assert "\\u" not in result


def test_non_ascii_content_lossless_roundtrip():
    """The JSON-encoded content portion must round-trip back to the
    original string — the reranker scoring relies on the literal text
    matching the query distribution."""
    original = '日本語 — "quoted" + emoji 🎉'
    result = DeclarativeMemory.string_from_episode_context([_make_episode(original)])
    json_part = result.split(": ", 1)[1].rstrip("\n")
    assert json.loads(json_part) == original


def test_output_is_utf8_encodable():
    """The context string is fed to ``Reranker.score`` and (via siblings)
    into LLM prompts; it must encode to UTF-8 cleanly."""
    result = DeclarativeMemory.string_from_episode_context(
        [_make_episode("Mixed: ASCII + 中文 + 🚀 + 👨‍👩‍👧‍👦")]
    )
    assert result.encode("utf-8").decode("utf-8") == result


def test_multiple_episodes_each_preserve_unicode():
    eps = [_make_episode("café"), _make_episode("寿司"), _make_episode("🚀")]
    result = DeclarativeMemory.string_from_episode_context(eps)
    lines = result.strip().split("\n")
    assert len(lines) == 3
    assert "café" in lines[0]
    assert "寿司" in lines[1]
    assert "🚀" in lines[2]
