"""The binding of records and filters to a store's declared schema."""

import math
import re
from uuid import uuid4

import pytest

from memmachine_server.common.filter import Equals, Property
from memmachine_server.common.vector_store.data_types import (
    PropertyTypeMismatchError,
    Record,
)
from memmachine_server.common.vector_store.utils import bind_filter, bind_record

_INDEXED_PROPERTIES = {"age": int, "score": float}

_REFUSED = [
    ("score", math.nan, "Property 'score' got float nan, which is not finite."),
    ("score", -math.inf, "Property 'score' got float -inf, which is not finite."),
    (
        "age",
        2**63,
        (
            "Property 'age' got int 9223372036854775808, "
            "which is outside the 64-bit signed range."
        ),
    ),
    (
        "score",
        -(2**63) - 1,
        (
            "Property 'score' got int -9223372036854775809, "
            "which is outside the 64-bit signed range."
        ),
    ),
    ("age", 1.5, "Property 'age' is declared as int, got float 1.5."),
    ("score", True, "Property 'score' is declared as float, got bool True."),
]


@pytest.mark.parametrize(("key", "value", "message"), _REFUSED)
def test_a_refused_record_value_says_what_is_wrong(key, value, message):
    record = Record(uuid=uuid4(), vector=[1.0], properties={key: value})
    with pytest.raises(PropertyTypeMismatchError, match=f"^{re.escape(message)}$"):
        bind_record(record, _INDEXED_PROPERTIES)


@pytest.mark.parametrize(("key", "value", "message"), _REFUSED)
def test_a_refused_filter_value_says_what_is_wrong(key, value, message):
    with pytest.raises(PropertyTypeMismatchError, match=f"^{re.escape(message)}$"):
        bind_filter(Equals(Property(key), value), _INDEXED_PROPERTIES)
