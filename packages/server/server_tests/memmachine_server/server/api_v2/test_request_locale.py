"""Tests for the locale query parameter and the request-locale dependency."""

import pytest
from babel import Locale

from memmachine_server.common.request_context import (
    DEFAULT_LOCALE,
    get_request_locale,
)
from memmachine_server.server.api_v2.service import (
    _parse_locale,
    provide_request_locale,
)


def test_parse_locale_falls_back_to_default():
    assert _parse_locale(None) == DEFAULT_LOCALE
    assert _parse_locale("") == DEFAULT_LOCALE
    assert _parse_locale("zz-ZZ") == DEFAULT_LOCALE  # unrecognized


def test_parse_locale_resolves_bcp47():
    assert _parse_locale("en-GB") == Locale("en", "GB")
    assert _parse_locale("fr-FR") == Locale("fr", "FR")
    assert _parse_locale("ja") == Locale("ja")  # language-only is fine


@pytest.mark.asyncio
async def test_provide_request_locale_sets_and_resets():
    gen = provide_request_locale("de-DE")
    await anext(gen)
    assert get_request_locale() == Locale("de", "DE")
    # Exhausting the generator runs the teardown that resets the contextvar.
    with pytest.raises(StopAsyncIteration):
        await anext(gen)
    assert get_request_locale() == DEFAULT_LOCALE
