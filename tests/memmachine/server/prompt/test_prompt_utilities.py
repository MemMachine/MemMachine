"""Tests for prompt utility functions in memmachine.server.prompt."""

import pytest

from memmachine.server.prompt.crm_prompt import current_date_dow, enum_list


def test_current_date_dow_default():
    timestamp = current_date_dow()
    assert len(timestamp) > 6


def test_invalid_timezone_raises_no_exception():
    """If timezone does not exist, no exception is raised."""
    try:
        timestamp = current_date_dow("Invalid/Timezone")
        assert len(timestamp) > 6
    except Exception as e:
        pytest.fail(f"current_date_dow raised an exception: {e}")


def test_enum_list_empty():
    assert enum_list([]) == ""


def test_enum_list_single():
    assert enum_list(["A"]) == '"A"'


def test_enum_list_multiple():
    assert enum_list(["A", "B", "C"]) == '"A", "B", "C"'
