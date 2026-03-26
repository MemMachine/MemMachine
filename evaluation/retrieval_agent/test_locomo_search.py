import pytest

from evaluation.retrieval_agent.locomo_search import DEFAULT_CONCURRENCY, build_parser


def test_locomo_search_concurrency_defaults_to_one():
    args = build_parser().parse_args(
        [
            "--data-path",
            "data.json",
            "--eval-result-path",
            "results.json",
            "--test-target",
            "retrieval_agent",
            "--config-path",
            "configuration.yml",
        ]
    )

    assert args.concurrency == DEFAULT_CONCURRENCY == 1


def test_locomo_search_accepts_explicit_concurrency():
    args = build_parser().parse_args(
        [
            "--data-path",
            "data.json",
            "--eval-result-path",
            "results.json",
            "--test-target",
            "retrieval_agent",
            "--config-path",
            "configuration.yml",
            "--concurrency",
            "3",
        ]
    )

    assert args.concurrency == 3


def test_locomo_search_rejects_non_positive_concurrency():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--data-path",
                "data.json",
                "--eval-result-path",
                "results.json",
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
                "--concurrency",
                "0",
            ]
        )
