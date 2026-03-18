import pytest

from evaluation.retrieval_agent.evaluate import build_parser as build_evaluate_parser
from evaluation.retrieval_agent.hotpotQA_test import (
    DEFAULT_CONCURRENCY as HOTPOTQA_DEFAULT_CONCURRENCY,
)
from evaluation.retrieval_agent.hotpotQA_test import (
    build_parser as build_hotpotqa_parser,
)
from evaluation.retrieval_agent.locomo_ingest import (
    DEFAULT_CONCURRENCY as LOCOMO_INGEST_DEFAULT_CONCURRENCY,
)
from evaluation.retrieval_agent.locomo_ingest import (
    build_parser as build_locomo_ingest_parser,
)
from evaluation.retrieval_agent.longmemeval_test import (
    DEFAULT_CONCURRENCY as LONGMEMEVAL_DEFAULT_CONCURRENCY,
)
from evaluation.retrieval_agent.longmemeval_test import (
    build_parser as build_longmemeval_parser,
)
from evaluation.retrieval_agent.wikimultihop_search import (
    build_parser as build_wikimultihop_parser,
)


def test_locomo_ingest_concurrency_defaults_to_ten():
    args = build_locomo_ingest_parser().parse_args(
        [
            "--data-path",
            "data.json",
            "--config-path",
            "configuration.yml",
        ]
    )

    assert args.concurrency == LOCOMO_INGEST_DEFAULT_CONCURRENCY == 10


@pytest.mark.parametrize(
    ("parser_factory", "args", "expected_concurrency"),
    [
        (
            build_locomo_ingest_parser,
            ["--data-path", "data.json", "--config-path", "configuration.yml"],
            LOCOMO_INGEST_DEFAULT_CONCURRENCY,
        ),
        (
            build_wikimultihop_parser,
            [
                "--data-path",
                "data.json",
                "--eval-result-path",
                "results.json",
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
            ],
            10,
        ),
        (
            build_hotpotqa_parser,
            [
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
            ],
            HOTPOTQA_DEFAULT_CONCURRENCY,
        ),
        (
            build_longmemeval_parser,
            [
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
            ],
            LONGMEMEVAL_DEFAULT_CONCURRENCY,
        ),
    ],
)
def test_benchmark_parsers_set_expected_default_concurrency(
    parser_factory, args, expected_concurrency
):
    parsed_args = parser_factory().parse_args(args)

    assert parsed_args.concurrency == expected_concurrency


@pytest.mark.parametrize(
    ("parser_factory", "args"),
    [
        (
            build_locomo_ingest_parser,
            [
                "--data-path",
                "data.json",
                "--config-path",
                "configuration.yml",
                "--concurrency",
                "4",
            ],
        ),
        (
            build_wikimultihop_parser,
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
                "4",
            ],
        ),
        (
            build_hotpotqa_parser,
            [
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
                "--concurrency",
                "4",
            ],
        ),
        (
            build_longmemeval_parser,
            [
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
                "--concurrency",
                "4",
            ],
        ),
    ],
)
def test_benchmark_parsers_accept_explicit_concurrency(parser_factory, args):
    parsed_args = parser_factory().parse_args(args)

    assert parsed_args.concurrency == 4


@pytest.mark.parametrize(
    ("parser_factory", "args"),
    [
        (
            build_locomo_ingest_parser,
            [
                "--data-path",
                "data.json",
                "--config-path",
                "configuration.yml",
                "--concurrency",
                "0",
            ],
        ),
        (
            build_wikimultihop_parser,
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
            ],
        ),
        (
            build_hotpotqa_parser,
            [
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
                "--concurrency",
                "0",
            ],
        ),
        (
            build_longmemeval_parser,
            [
                "--test-target",
                "retrieval_agent",
                "--config-path",
                "configuration.yml",
                "--concurrency",
                "0",
            ],
        ),
    ],
)
def test_benchmark_parsers_reject_non_positive_concurrency(parser_factory, args):
    with pytest.raises(SystemExit):
        parser_factory().parse_args(args)


def test_evaluate_parser_accepts_explicit_max_workers():
    args = build_evaluate_parser().parse_args(
        [
            "--config-path",
            "configuration.yml",
            "--max_workers",
            "6",
        ]
    )

    assert args.max_workers == 6


def test_evaluate_parser_rejects_non_positive_max_workers():
    with pytest.raises(SystemExit):
        build_evaluate_parser().parse_args(
            [
                "--config-path",
                "configuration.yml",
                "--max_workers",
                "0",
            ]
        )
