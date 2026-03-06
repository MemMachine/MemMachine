"""Tests for the PostgreSQL text sanitizer utility."""

import logging

from memmachine_server.semantic_memory.storage.text_sanitizer import sanitize_pg_text


class TestSanitizePgText:
    """Tests for sanitize_pg_text."""

    def test_clean_string_passes_through(self):
        assert sanitize_pg_text("hello world") == "hello world"

    def test_empty_string_passes_through(self):
        assert sanitize_pg_text("") == ""

    def test_unicode_string_passes_through(self):
        value = "Día de Los Muertos"
        assert sanitize_pg_text(value) == value

    def test_strips_single_null_byte(self):
        assert sanitize_pg_text("D\x00a de Los Muertos") == "Da de Los Muertos"

    def test_strips_multiple_null_bytes(self):
        assert sanitize_pg_text("a\x00b\x00c") == "abc"

    def test_strips_leading_null_byte(self):
        assert sanitize_pg_text("\x00hello") == "hello"

    def test_strips_trailing_null_byte(self):
        assert sanitize_pg_text("hello\x00") == "hello"

    def test_string_of_only_null_bytes(self):
        assert sanitize_pg_text("\x00\x00\x00") == ""

    def test_logs_warning_when_stripping(self, caplog):
        with caplog.at_level(logging.WARNING):
            sanitize_pg_text("D\x00a de Los Muertos", context="feature.value")

        assert len(caplog.records) == 1
        assert "null byte" in caplog.records[0].message.lower()
        assert "feature.value" in caplog.records[0].message

    def test_no_log_for_clean_string(self, caplog):
        with caplog.at_level(logging.WARNING):
            sanitize_pg_text("clean string")

        assert len(caplog.records) == 0
