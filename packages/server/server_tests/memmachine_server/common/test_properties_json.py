"""Tests for properties_json: encode/decode round-trips."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest

from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)


class TestEncodeDecodeRoundTrip:
    def test_bool(self):
        original = {"flag": True}
        assert decode_properties(encode_properties(original)) == original

    def test_int(self):
        original = {"count": 42}
        assert decode_properties(encode_properties(original)) == original

    def test_float(self):
        original = {"score": 3.14}
        assert decode_properties(encode_properties(original)) == original

    def test_str(self):
        original = {"name": "hello"}
        assert decode_properties(encode_properties(original)) == original

    def test_datetime_utc(self):
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        result = decode_properties(encode_properties({"ts": dt}))
        assert result["ts"] == dt

    def test_datetime_positive_offset(self):
        tz = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 6, 15, 17, 30, 0, tzinfo=tz)
        result = decode_properties(encode_properties({"ts": dt}))
        assert result["ts"] == dt
        ts = result["ts"]
        assert isinstance(ts, datetime)
        assert ts.utcoffset() == timedelta(hours=5, minutes=30)

    def test_datetime_negative_offset(self):
        tz = timezone(timedelta(hours=-8))
        dt = datetime(2024, 6, 15, 4, 0, 0, tzinfo=tz)
        result = decode_properties(encode_properties({"ts": dt}))
        assert result["ts"] == dt
        ts = result["ts"]
        assert isinstance(ts, datetime)
        assert ts.utcoffset() == timedelta(hours=-8)

    def test_multiple_fields(self):
        original = {
            "flag": False,
            "count": 7,
            "ratio": 0.5,
            "label": "test",
            "created": datetime(2024, 1, 1, tzinfo=UTC),
        }
        assert decode_properties(encode_properties(original)) == original

    def test_none_returns_empty(self):
        assert encode_properties(None) == {}
        assert decode_properties(None) == {}

    def test_empty_dict_returns_empty(self):
        assert encode_properties({}) == {}
        assert decode_properties({}) == {}


class TestEncodeFormat:
    def test_int_format(self):
        encoded = encode_properties({"x": 10})
        assert encoded == {"x": {"v": 10, "t": "int"}}

    def test_datetime_stores_utc_and_offset(self):
        tz = timezone(timedelta(hours=9))
        dt = datetime(2024, 6, 15, 21, 0, 0, tzinfo=tz)
        encoded = encode_properties({"ts": dt})
        entry = encoded["ts"]
        assert entry["t"] == "datetime"
        assert entry["tz"] == 9 * 3600
        assert entry["v"] == dt.astimezone(UTC).isoformat()


class TestDecodeErrors:
    def test_unknown_type_name(self):
        with pytest.raises(ValueError, match="unknown type name"):
            decode_properties({"x": {"t": "unknown", "v": 1}})

    def test_non_dict_entry(self):
        with pytest.raises(TypeError, match="must be Mapping"):
            decode_properties({"x": 42})

    def test_non_string_key(self):
        with pytest.raises(TypeError, match="Property key must be str"):
            decode_properties({42: {"t": "int", "v": 1}})

    def test_extra_keys_scalar(self):
        with pytest.raises(ValueError, match="must have exactly keys"):
            decode_properties({"x": {"t": "int", "v": 1, "extra": "junk"}})

    def test_extra_keys_datetime(self):
        with pytest.raises(ValueError, match="must have exactly keys"):
            decode_properties(
                {
                    "x": {
                        "t": "datetime",
                        "v": "2024-01-01T00:00:00+00:00",
                        "tz": 0,
                        "extra": 1,
                    }
                }
            )

    def test_missing_tz_datetime(self):
        with pytest.raises(ValueError, match="datetime entry must have exactly keys"):
            decode_properties(
                {"x": {"t": "datetime", "v": "2024-01-01T00:00:00+00:00"}}
            )

    def test_wrong_tz_type(self):
        with pytest.raises(TypeError, match="must be int"):
            decode_properties(
                {"x": {"t": "datetime", "v": "2024-01-01T00:00:00+00:00", "tz": "zero"}}
            )

    def test_wrong_value_type(self):
        with pytest.raises(TypeError, match="must be int"):
            decode_properties({"x": {"t": "int", "v": "not_an_int"}})

    def test_missing_value(self):
        with pytest.raises(ValueError, match="entry missing"):
            decode_properties({"x": {"t": "int"}})

    def test_missing_type(self):
        with pytest.raises(TypeError, match="must be str"):
            decode_properties({"x": {"v": 1}})


class TestEncodeErrors:
    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported property value type"):
            encode_properties(cast(Any, {"x": [1, 2, 3]}))
