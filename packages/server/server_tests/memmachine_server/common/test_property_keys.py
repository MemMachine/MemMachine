"""Tests for the reserved property key namespace."""

import pytest

from memmachine_server.common.property_keys import (
    RESERVED_PROPERTY_KEY_PREFIX,
    is_reserved_property_key,
    reserved_property_key,
    validate_caller_property_key,
)


def test_reserved_key_is_prefix_system_field():
    assert reserved_property_key("event", "timestamp") == "memmachine_event_timestamp"
    assert is_reserved_property_key("memmachine_event_timestamp")
    assert not is_reserved_property_key("timestamp")


def test_reserved_key_overrunning_the_budget_fails_at_build_time():
    with pytest.raises(ValueError, match="naming contract"):
        reserved_property_key("event", "a_field_name_far_too_long_for_it")


def test_caller_key_in_the_reserved_namespace_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        validate_caller_property_key(f"{RESERVED_PROPERTY_KEY_PREFIX}mine")


@pytest.mark.parametrize("key", ["Color", "m.color", "a" * 33, ""])
def test_caller_key_outside_the_naming_contract_is_rejected(key):
    with pytest.raises(ValueError, match=r"\[a-z0-9_\]"):
        validate_caller_property_key(key)


@pytest.mark.parametrize("key", ["color", "_episode_uid", "a" * 32])
def test_legal_caller_key_passes(key):
    validate_caller_property_key(key)
