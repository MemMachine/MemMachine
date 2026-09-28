"""Tests for the prometheus multiprocess directory prepared at startup.

The directory has to exist before any metric is registered, and every worker
has to agree on the same one, so the work happens in ``main()`` rather than in
the ``/metrics`` handler. These tests pin the parts that are easy to get wrong
and impossible to notice at runtime: a scrape from a misconfigured server
returns plausible-looking numbers rather than an error.
"""

import os
from pathlib import Path

import pytest

from memmachine_server.server.app import _prepare_multiproc_dir, _worker_count

ENV_VAR = "PROMETHEUS_MULTIPROC_DIR"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("MEMMACHINE_WORKERS", raising=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 1), ("1", 1), ("4", 4), ("not-a-number", 1), ("", 1)],
)
def test_worker_count(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("MEMMACHINE_WORKERS", value)
    assert _worker_count() == expected


def test_single_worker_leaves_the_variable_unset(monkeypatch):
    """One worker needs no aggregation, and multiprocess mode costs a scrape."""
    monkeypatch.setenv("MEMMACHINE_WORKERS", "1")
    _prepare_multiproc_dir()
    assert ENV_VAR not in os.environ


def test_multiple_workers_choose_a_directory(monkeypatch, tmp_path):
    """Without this, each worker answers scrapes from its own registry."""
    monkeypatch.setenv("MEMMACHINE_WORKERS", "4")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    _prepare_multiproc_dir()

    chosen = os.environ.get(ENV_VAR)
    assert chosen is not None
    assert Path(chosen).is_dir()
    assert Path(chosen).parent == tmp_path


def test_an_explicit_directory_wins(monkeypatch, tmp_path):
    """Two servers on one host must be able to keep their metrics apart."""
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("MEMMACHINE_WORKERS", "4")
    monkeypatch.setenv(ENV_VAR, str(explicit))

    _prepare_multiproc_dir()

    assert os.environ[ENV_VAR] == str(explicit)
    assert explicit.is_dir()


def test_an_explicit_directory_is_honoured_for_one_worker(monkeypatch, tmp_path):
    """The worker count gates the default, not the operator's own setting."""
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("MEMMACHINE_WORKERS", "1")
    monkeypatch.setenv(ENV_VAR, str(explicit))

    _prepare_multiproc_dir()

    assert explicit.is_dir()


def test_nested_directories_are_created(monkeypatch, tmp_path):
    target = tmp_path / "a" / "b" / "c"
    monkeypatch.setenv(ENV_VAR, str(target))

    _prepare_multiproc_dir()

    assert target.is_dir()


def test_stale_files_are_cleared(monkeypatch, tmp_path):
    """A previous run's dead workers would otherwise be summed into every scrape."""
    target = tmp_path / "metrics"
    target.mkdir()
    (target / "counter_123.db").write_bytes(b"stale")
    keep = target / "notes.txt"
    keep.write_text("not ours")
    monkeypatch.setenv(ENV_VAR, str(target))

    _prepare_multiproc_dir()

    assert list(target.glob("*.db")) == []
    assert keep.exists()


def test_an_unwritable_explicit_directory_warns_but_does_not_raise(
    monkeypatch, tmp_path, caplog
):
    """Failing to boot over metrics would be the wrong trade."""
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, so mkdir cannot succeed here")
    monkeypatch.setenv(ENV_VAR, str(blocked / "metrics"))

    _prepare_multiproc_dir()

    assert "will not be aggregated" in caplog.text


def test_a_failed_default_is_withdrawn(monkeypatch, tmp_path):
    """prometheus_client raises in every worker if it cannot open the directory.

    Leaving a variable set that only this code chose would turn an unwritable
    temp directory into a server that refuses to start.
    """
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, so mkdir cannot succeed under it")
    monkeypatch.setenv("MEMMACHINE_WORKERS", "4")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(blocked))

    _prepare_multiproc_dir()

    assert ENV_VAR not in os.environ
