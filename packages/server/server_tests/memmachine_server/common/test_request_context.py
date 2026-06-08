"""Tests for the request-locale contextvar."""

from babel import Locale

from memmachine_server.common.request_context import (
    DEFAULT_LOCALE,
    get_request_locale,
    reset_request_locale,
    set_request_locale,
)


def test_default_is_en_us():
    assert Locale("en", "US") == DEFAULT_LOCALE
    assert get_request_locale() == DEFAULT_LOCALE


def test_set_then_reset_restores_previous():
    token = set_request_locale(Locale("fr", "FR"))
    try:
        assert get_request_locale() == Locale("fr", "FR")
    finally:
        reset_request_locale(token)
    assert get_request_locale() == DEFAULT_LOCALE
