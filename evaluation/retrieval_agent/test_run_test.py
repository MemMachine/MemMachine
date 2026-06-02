import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TEST = REPO_ROOT / "evaluation" / "retrieval_agent" / "run_test.sh"


def _write_file(path: Path, contents: str, *, executable: bool = False) -> None:
    path.write_text(textwrap.dedent(contents).lstrip(), encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


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


def test_help_mentions_delete_run_type():
    for benchmark in ("locomo", "wikimultihop", "hotpotqa", "longmemeval"):
        result = subprocess.run(
            ["bash", str(RUN_TEST), benchmark, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, benchmark
        assert "delete" in result.stdout, f"{benchmark} help missing 'delete'"


def test_locomo_delete_rejects_ingest_concurrency():
    result = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "locomo",
            "exp1",
            "delete",
            "retrieval_agent",
            "--ingest-concurrency",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--ingest-concurrency can only be used with locomo ingest" in result.stdout


def test_wikimultihop_delete_rejects_search_concurrency():
    result = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "wikimultihop",
            "exp1",
            "delete",
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


def test_wikimultihop_delete_rejects_extra_positional_args():
    # With delete, wikimultihop should accept exactly 4 positional args.
    # Passing a 5th (LENGTH) must be rejected by validate_args.
    result = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "wikimultihop",
            "exp1",
            "delete",
            "retrieval_agent",
            "10",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "WikiMultihop Usage" in result.stdout


def test_longmemeval_delete_invokes_delete_script(tmp_path):
    repo_root = tmp_path / "repo"
    script_dir = repo_root / "evaluation" / "retrieval_agent"
    bin_dir = tmp_path / "bin"
    script_dir.mkdir(parents=True)
    bin_dir.mkdir()

    run_test_copy = script_dir / "run_test.sh"
    shutil.copy(RUN_TEST, run_test_copy)
    run_test_copy.chmod(run_test_copy.stat().st_mode | stat.S_IXUSR)

    (script_dir / "configuration.yml").write_text(
        "logging:\n  level: INFO\n", encoding="utf-8"
    )

    invocations_log = tmp_path / "invocations.log"
    _write_file(
        script_dir / "longmemeval_test.py",
        f"""
        import sys
        from pathlib import Path

        Path(r"{invocations_log}").write_text(" ".join(sys.argv[1:]), encoding="utf-8")
        """,
    )

    _write_file(
        bin_dir / "uv",
        f"""
        #!/usr/bin/env bash
        if [ "$1" = "run" ] && [ "$2" = "python" ]; then
            shift 2
            exec "{sys.executable}" "$@"
        fi
        exit 1
        """,
        executable=True,
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(run_test_copy),
            "longmemeval",
            "exp1",
            "delete",
            "retrieval_agent",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    invoked_args = invocations_log.read_text(encoding="utf-8")
    assert "--run-type delete" in invoked_args
    assert "--session-id longmemeval_exp1" in invoked_args


def test_longmemeval_search_uses_uv_for_preflight_and_postprocessing(tmp_path):
    repo_root = tmp_path / "repo"
    script_dir = repo_root / "evaluation" / "retrieval_agent"
    bin_dir = tmp_path / "bin"
    script_dir.mkdir(parents=True)
    bin_dir.mkdir()

    run_test_copy = script_dir / "run_test.sh"
    shutil.copy(RUN_TEST, run_test_copy)
    run_test_copy.chmod(run_test_copy.stat().st_mode | stat.S_IXUSR)

    (script_dir / "configuration.yml").write_text(
        "logging:\n  level: INFO\n", encoding="utf-8"
    )

    _write_file(
        script_dir / "preflight.py",
        """
        raise SystemExit(0)
        """,
    )
    _write_file(
        script_dir / "longmemeval_test.py",
        """
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--eval-result-path", required=True)
        args, _ = parser.parse_known_args()
        Path(args.eval_result_path).write_text("{}", encoding="utf-8")
        """,
    )
    _write_file(
        script_dir / "evaluate.py",
        """
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--target-path", required=True)
        args, _ = parser.parse_known_args()
        Path(args.target_path).write_text("{}", encoding="utf-8")
        """,
    )
    _write_file(
        script_dir / "generate_scores.py",
        """
        print("score: ok")
        """,
    )

    uv_log = tmp_path / "uv.log"
    python_log = tmp_path / "python.log"
    _write_file(
        bin_dir / "uv",
        f"""
        #!/usr/bin/env bash
        echo "$@" >> "{uv_log}"
        if [ "$1" = "run" ] && [ "$2" = "python" ]; then
            shift 2
            exec "{sys.executable}" "$@"
        fi
        exit 1
        """,
        executable=True,
    )
    _write_file(
        bin_dir / "python",
        f"""
        #!/usr/bin/env bash
        echo "$@" >> "{python_log}"
        exit 127
        """,
        executable=True,
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(run_test_copy),
            "longmemeval",
            "exp1",
            "search",
            "longmemeval_s_cleaned",
            "retrieval_agent",
            "1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "score: ok" in result.stdout

    uv_invocations = uv_log.read_text(encoding="utf-8").splitlines()
    assert any("preflight.py" in line for line in uv_invocations)
    assert any("longmemeval_test.py" in line for line in uv_invocations)
    assert any("evaluate.py" in line for line in uv_invocations)
    assert any("generate_scores.py" in line for line in uv_invocations)
    assert not python_log.exists()
