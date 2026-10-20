"""Tests for the vector store utilities."""

import pytest

from memmachine_server.common.vector_store.utils import pack_requests


@pytest.mark.parametrize(
    ("sizes", "envelope_bytes", "max_request_bytes", "expected"),
    [
        ([3, 4, 5], 2, 14, [["a", "b", "c"]]),
        ([3, 4, 5], 2, 13, [["a", "b"], ["c"]]),
        ([5, 5, 5, 5], 0, 10, [["a", "b"], ["c", "d"]]),
        ([5, 5, 5, 5], 0, 9, [["a"], ["b"], ["c"], ["d"]]),
        ([4, 4, 4], 1, 5, [["a"], ["b"], ["c"]]),
        ([6, 3, 1, 5], 0, 10, [["a", "b", "c"], ["d"]]),
        ([1, 9, 1], 0, 9, [["a"], ["b"], ["c"]]),
    ],
    ids=[
        "an_exact_fit",
        "a_byte_short_of_a_fit",
        "a_boundary_between_requests",
        "a_byte_short_of_two_per_request",
        "one_per_request",
        "a_run_filled_to_its_last_byte",
        "a_record_filling_a_request_alone",
    ],
)
def test_records_are_split_in_order_into_the_fewest_runs_that_fit(
    sizes: list[int],
    envelope_bytes: int,
    max_request_bytes: int,
    expected: list[list[str]],
):
    records = ["a", "b", "c", "d"][: len(sizes)]

    assert (
        pack_requests(
            records,
            sizes,
            envelope_bytes=envelope_bytes,
            max_request_bytes=max_request_bytes,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("sizes", "envelope_bytes", "max_request_bytes"),
    [([1, 1, 9, 1], 2, 10), ([1], 10, 10)],
    ids=["a_record_over_a_request_alone", "an_envelope_filling_the_request"],
)
def test_a_record_that_does_not_fit_a_request_alone_is_refused(
    sizes: list[int], envelope_bytes: int, max_request_bytes: int
):
    with pytest.raises(ValueError, match="max_request_bytes"):
        pack_requests(
            list(range(len(sizes))),
            sizes,
            envelope_bytes=envelope_bytes,
            max_request_bytes=max_request_bytes,
        )
