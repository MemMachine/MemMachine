"""The vector store's data types: the rule a vector store name follows."""

import pytest

from memmachine_server.common.vector_store.data_types import (
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
