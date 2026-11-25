"""Tests for the MEMMACHINE_CONFIG_API environment flag parsing.

Operators write ``0``, ``false``, ``no`` or ``off`` (in any case) when they
mean off. The flag used to be read with plain truthiness, so every non-empty
spelling mounted the configuration API, which can change the running server's
configuration. These tests pin the accepted spellings on both sides.
"""

import pytest

from memmachine_server.server.app import _env_flag

ENV_VAR = "MEMMACHINE_CONFIG_API"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("no", False),
        ("No", False),
        ("off", False),
        ("OFF", False),
        (" 0 ", False),
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
    ],
)
def test_env_flag(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(ENV_VAR, value)
    assert _env_flag(ENV_VAR) is expected
