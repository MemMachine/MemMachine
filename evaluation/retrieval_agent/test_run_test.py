import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TEST = REPO_ROOT / "evaluation" / "retrieval_agent" / "run_test.sh"


def test_locomo_help_mentions_search_concurrency():
    result = subprocess.run(
        ["bash", str(RUN_TEST), "locomo", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--ingest-concurrency" in result.stdout
    assert "--search-concurrency" in result.stdout
    assert "--judge-concurrency" in result.stdout
    assert "default: 1" in result.stdout


def test_wikimultihop_help_mentions_search_and_judge_concurrency():
    result = subprocess.run(
        ["bash", str(RUN_TEST), "wikimultihop", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--search-concurrency" in result.stdout
    assert "--judge-concurrency" in result.stdout


def test_locomo_rejects_search_concurrency_for_ingest():
    result = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "locomo",
            "exp1",
            "ingest",
            "retrieval_agent",
            "--search-concurrency",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--search-concurrency can only be used with search runs" in result.stdout


def test_wikimultihop_rejects_ingest_concurrency():
    result = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "wikimultihop",
            "exp1",
            "search",
            "retrieval_agent",
            "10",
            "--ingest-concurrency",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--ingest-concurrency is only supported for locomo ingest" in result.stdout
