"""Tests for the vector store's data types."""

import math
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memmachine_server.common.vector_store.data_types import Record


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf])
def test_a_record_refuses_a_vector_coordinate_that_is_not_finite(coordinate):
    with pytest.raises(ValidationError, match="finite number"):
        Record(uuid=uuid4(), vector=[1.0, coordinate, 0.0])
