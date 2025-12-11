import contextlib
import logging
import os

import pytest

from memmachine.common.configuration.log_conf import LogConf, LogLevel, to_log_level


# --- Helper to isolate environment ---
@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ["LOG_LEVEL", "LOG_FORMAT", "LOG_PATH"]:
        monkeypatch.delenv(var, raising=False)


# --- Tests for to_log_level ---
def test_to_log_level_valid():
    assert to_log_level("info") == LogLevel.INFO
    assert to_log_level("DEBUG") == LogLevel.DEBUG


def test_to_log_level_invalid():
    with pytest.raises(ValueError, match="Invalid log level: badlevel"):
        to_log_level("badlevel")


# --- Tests for LogConf.level validation ---
def test_logconf_level_validation_from_enum():
    c = LogConf(level=LogLevel.WARNING)
    assert c.level == LogLevel.WARNING


def test_logconf_level_validation_from_str():
    c = LogConf(level=LogLevel.ERROR)
    assert c.level == LogLevel.ERROR


def test_logconf_level_validation_invalid():
    with pytest.raises(ValueError, match="not a valid LogLevel"):
        LogConf(level=LogLevel("BAD"))


# --- Tests for format validation ---
def test_logconf_format_validation_valid():
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    c = LogConf(format=fmt)
    assert c.format == fmt


@pytest.mark.parametrize(
    "badfmt",
    [
        "%(asctime)s [%(levelname)s]",
        "%(message)s only message",
    ],
)
def test_logconf_format_validation_invalid(badfmt):
    with pytest.raises(ValueError, match="log format must include"):
        LogConf(format=badfmt)


def test_logconf_path_none_or_empty():
    assert LogConf().path == "MemMachine.log"
    assert LogConf(path="").path is None


def test_logconf_path_valid(tmp_path):
    f = tmp_path / "test.log"
    c = LogConf(path=str(f))
    assert c.path == str(f)


def test_logconf_path_nonexistent_dir(tmp_path):
    bad_dir = tmp_path / "noexist"
    bad_path = bad_dir / "file.log"
    with pytest.raises(ValueError, match="does not exist"):
        LogConf(path=str(bad_path))


def test_logconf_path_non_writable(monkeypatch, tmp_path):
    file_path = tmp_path / "f.log"

    def fake_access(_, __):
        return False

    monkeypatch.setattr(os, "access", fake_access)
    with pytest.raises(ValueError, match="not writable"):
        LogConf(path=str(file_path))


# --- Tests for apply() ---
def test_apply_basic_config(monkeypatch):
    """Covers basic apply without env vars."""
    c = LogConf()
    c.apply()
    logger = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    old_level = root.level
    old_handlers = list(root.handlers)
    yield
    # restore
    for h in list(root.handlers):
        root.removeHandler(h)
        with contextlib.suppress(Exception):
            h.close()
    for h in old_handlers:
        root.addHandler(h)
    root.setLevel(old_level)


def test_apply_with_env_overrides(monkeypatch, tmp_path, restore_logging):
    """Covers env variable overrides and file logging."""
    log_file = tmp_path / "env-test.log"

    monkeypatch.setenv("LOG_LEVEL", "info")
    monkeypatch.setenv("LOG_FORMAT", "%(asctime)s %(message)s %(levelname)s")
    monkeypatch.setenv("LOG_PATH", str(log_file))

    c = LogConf()
    c.apply()

    logger = logging.getLogger("new_logger")
    logger.debug("test debug")
    logger.info("test info")
    logger.error("test error")

    assert log_file.exists()
    content = log_file.read_text()
    assert "test debug" not in content
    assert "test info" in content
    assert "test error" in content


def test_apply_invalid_env_level(monkeypatch):
    """Covers invalid env level handling."""
    monkeypatch.setenv("LOG_LEVEL", "bad")

    c = LogConf()
    with pytest.raises(ValueError, match="Invalid log level: bad"):
        c.apply()
