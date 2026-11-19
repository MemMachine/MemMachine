"""The vector store's data types: the rule a vector store name follows, and a record's vector."""

import math
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memmachine_server.common.vector_store.data_types import (
    Record,
    validate_vector_store_name,
)


@pytest.mark.parametrize("name", ["test_vector_store", "0" * 32])
def test_a_vector_store_name_of_lowercase_letters_digits_and_underscores(name):
    validate_vector_store_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Upper",
        "with-hyphen",
        "trailing_newline\n",
        "0" * 33,
    ],
)
def test_any_other_vector_store_name_is_refused(name):
    with pytest.raises(ValueError, match="Vector store name"):
        validate_vector_store_name(name)


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf])
def test_a_record_refuses_a_vector_coordinate_that_is_not_finite(coordinate):
    with pytest.raises(ValidationError, match="finite number"):
        Record(uuid=uuid4(), vector=[1.0, coordinate, 0.0])
