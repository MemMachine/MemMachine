import pytest

from memmachine_server.common.concurrency_scope import ConcurrencyScope
from memmachine_server.common.configuration import Configuration, ServerConf


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Automatically clear HOST and PORT before each test."""
    for var in ["HOST", "PORT", "MEMMACHINE_CONCURRENCY_SCOPE", "MEMMACHINE_WORKERS"]:
        monkeypatch.delenv(var, raising=False)


def test_defaults_without_env():
    conf = ServerConf()
    assert conf.host == "localhost"
    assert conf.port == 8080
    assert conf.concurrency_scope == ConcurrencyScope.PROCESS
    assert conf.effective_concurrency_scope == ConcurrencyScope.PROCESS


def test_host_overridden_by_env(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    conf = ServerConf()
    assert conf.host == "0.0.0.0"
    assert conf.port == 8080  # default still applies


def test_port_overridden_by_env(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    conf = ServerConf()
    assert conf.port == 9000
    assert conf.host == "localhost"  # default still applies


def test_both_host_and_port_overridden(monkeypatch):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "5001")
    conf = ServerConf()
    assert conf.host == "127.0.0.1"
    assert conf.port == 5001


def test_invalid_server_port():
    with pytest.raises(ValueError, match="port") as excinfo:
        ServerConf(port=70000)  # Invalid port, should be between 1 and 65535

    assert "port" in str(excinfo.value)


def test_invalid_port_raises_error(monkeypatch):
    monkeypatch.setenv("PORT", "-1")

    with pytest.raises(ValueError, match="port") as excinfo:
        ServerConf()

    assert "port" in str(excinfo.value)


def test_scope_environment_overrides_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMMACHINE_CONCURRENCY_SCOPE", " CLUSTER ")
    conf = ServerConf(concurrency_scope=ConcurrencyScope.HOST)
    assert conf.concurrency_scope == ConcurrencyScope.CLUSTER
    assert conf.to_yaml_dict()["concurrency_scope"] == "cluster"


def test_omitted_server_section_reads_scope_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MEMMACHINE_CONCURRENCY_SCOPE", "cluster")
    assert (
        Configuration.model_construct().server.concurrency_scope
        == ConcurrencyScope.CLUSTER
    )


def test_invalid_scope_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMMACHINE_CONCURRENCY_SCOPE", "unknown")
    with pytest.raises(ValueError, match="concurrency_scope"):
        ServerConf()


@pytest.mark.parametrize("workers", ["2", "8"])
def test_multiple_workers_require_host_scope(
    monkeypatch: pytest.MonkeyPatch, workers: str
):
    monkeypatch.setenv("MEMMACHINE_WORKERS", workers)
    assert ServerConf().effective_concurrency_scope == ConcurrencyScope.HOST
    conf = ServerConf(concurrency_scope=ConcurrencyScope.CLUSTER)
    assert conf.effective_concurrency_scope == ConcurrencyScope.CLUSTER


@pytest.mark.parametrize("workers", ["1", "invalid"])
def test_single_worker_and_invalid_worker_fallback(
    monkeypatch: pytest.MonkeyPatch, workers: str
):
    monkeypatch.setenv("MEMMACHINE_WORKERS", workers)
    assert ServerConf().effective_concurrency_scope == ConcurrencyScope.PROCESS
