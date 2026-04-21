"""Unit tests for evaluate_llm_judge retry logic."""

import json
from unittest.mock import MagicMock

import pytest

from evaluation.retrieval_agent.llm_judge import _MAX_JUDGE_ATTEMPTS, evaluate_llm_judge


def _call_fn_returning(*responses: str):
    """Return a stub call_fn that yields each response in sequence."""
    mock = MagicMock(side_effect=list(responses))
    return mock


def test_correct_label_returns_1():
    call_fn = _call_fn_returning(json.dumps({"label": "CORRECT"}))
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == 1


def test_wrong_label_returns_0():
    call_fn = _call_fn_returning(json.dumps({"label": "WRONG"}))
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == 0


def test_unknown_label_returns_0():
    call_fn = _call_fn_returning(json.dumps({"label": "MAYBE"}))
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == 0


def test_call_fn_called_once_on_success():
    call_fn = _call_fn_returning(json.dumps({"label": "CORRECT"}))
    evaluate_llm_judge("q", "gold", "gen", call_fn)
    assert call_fn.call_count == 1


def test_missing_label_retries_then_succeeds():
    call_fn = _call_fn_returning(
        json.dumps({}),
        json.dumps({"label": "CORRECT"}),
    )
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == 1
    assert call_fn.call_count == 2


def test_missing_label_both_attempts_returns_0():
    call_fn = _call_fn_returning(
        json.dumps({}),
        json.dumps({}),
    )
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == 0
    assert call_fn.call_count == _MAX_JUDGE_ATTEMPTS


def test_non_dict_response_retries_then_succeeds():
    call_fn = _call_fn_returning(
        "just plain text",
        json.dumps({"label": "WRONG"}),
    )
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == 0
    assert call_fn.call_count == 2


def test_call_fn_called_twice_on_retry():
    call_fn = _call_fn_returning(
        json.dumps({"no_label": "oops"}),
        json.dumps({"label": "WRONG"}),
    )
    evaluate_llm_judge("q", "gold", "gen", call_fn)
    assert call_fn.call_count == 2


def test_non_dict_both_attempts_returns_0():
    call_fn = _call_fn_returning("text", "also text")
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == 0
    assert call_fn.call_count == _MAX_JUDGE_ATTEMPTS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (json.dumps({"label": "CORRECT"}), 1),
        (json.dumps({"label": "WRONG"}), 0),
        (json.dumps({"label": "anything else"}), 0),
    ],
)
def test_label_values(raw, expected):
    call_fn = _call_fn_returning(raw)
    assert evaluate_llm_judge("q", "gold", "gen", call_fn) == expected
