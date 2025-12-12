import pytest

from memmachine.common.utils import chunk_text, chunk_text_balanced, unflatten_like


def test_chunk_text():
    text = ""
    assert chunk_text(text, max_length=5) == []

    text = "This is a sample text for chunking."
    assert chunk_text(text, max_length=10) == [
        "This is a ",
        "sample tex",
        "t for chun",
        "king.",
    ]

    text = "AAAAABBBBBCCCCC"
    assert chunk_text(text, max_length=5) == [
        "AAAAA",
        "BBBBB",
        "CCCCC",
    ]

    with pytest.raises(ValueError, match=r"max_length must be greater than 0"):
        chunk_text(text, max_length=0)

    with pytest.raises(ValueError, match=r"max_length must be greater than 0"):
        chunk_text(text, max_length=-1)


def test_chunk_text_balanced():
    text = ""
    assert chunk_text_balanced(text, max_length=5) == []

    text = "This is a sample text for balanced chunking."
    chunks = chunk_text_balanced(text, max_length=10)
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert (
        max(len(chunk) for chunk in chunks) - min(len(chunk) for chunk in chunks) <= 1
    )

    text = "AAAAABBBBBCCCCC"
    chunks = chunk_text_balanced(text, max_length=5)
    assert chunks == ["AAAAA", "BBBBB", "CCCCC"]
    chunks = chunk_text_balanced(text, max_length=3)
    assert chunks == ["AAA", "AAB", "BBB", "BCC", "CCC"]

    with pytest.raises(ValueError, match=r"max_length must be greater than 0"):
        chunk_text(text, max_length=0)

    with pytest.raises(ValueError, match=r"max_length must be greater than 0"):
        chunk_text(text, max_length=-1)


def test_unflatten_like():
    flat_list = [1, 2, 3, 4, 5, 6]
    template = [[0, 0], [0, 0, 0], [0]]
    result = unflatten_like(flat_list, template)
    assert result == [[1, 2], [3, 4, 5], [6]]

    flat_list = []
    template = [[], [], []]
    result = unflatten_like(flat_list, template)
    assert result == [[], [], []]

    flat_list = [1, 2]
    template = [[], [0, 0]]
    result = unflatten_like(flat_list, template)
    assert result == [[], [1, 2]]

    flat_list = [1, 2, 3]
    template = [0, [0, 0]]
    with pytest.raises(
        TypeError, match=r"All elements in template_list must be lists."
    ):
        result = unflatten_like(flat_list, template)

    flat_list = [1, 2, 3]
    template = [[], [0, 0]]
    with pytest.raises(
        ValueError, match=r"flat_list cannot be unflattened to match template_list."
    ):
        result = unflatten_like(flat_list, template)

    flat_list = [1, 2, 3]
    template = [[], [0, 0], []]
    with pytest.raises(
        ValueError, match=r"flat_list cannot be unflattened to match template_list."
    ):
        result = unflatten_like(flat_list, template)
