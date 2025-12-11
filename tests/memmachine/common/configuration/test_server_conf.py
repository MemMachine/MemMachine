import pytest

from memmachine.common.configuration import ServerConf


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Automatically clear HOST and PORT before each test."""
    for var in ["HOST", "PORT"]:
        monkeypatch.delenv(var, raising=False)


def test_defaults_without_env():
    conf = ServerConf()
    assert conf.host == "localhost"
    assert conf.port == 8080


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
