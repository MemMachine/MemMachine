from unittest.mock import MagicMock

import pytest

from memmachine_server.common.errors import ConfigurationError
from memmachine_server.server import app as app_module


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMMACHINE_WORKERS", raising=False)
    monkeypatch.delenv("MEMMACHINE_CONFIG_API", raising=False)


def test_config_api_stays_off_by_default() -> None:
    app = app_module.MemMachineAPI()
    assert "/api/v2/config" not in app.openapi()["paths"]


@pytest.mark.parametrize("workers", ["2", "8"])
def test_explicit_config_api_rejects_multiple_workers(
    monkeypatch: pytest.MonkeyPatch, workers: str
) -> None:
    monkeypatch.setenv("MEMMACHINE_WORKERS", workers)
    with pytest.raises(ConfigurationError, match="single server process"):
        app_module.MemMachineAPI(with_config_api=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_single_process_config_api_selection(enabled: bool) -> None:
    app = app_module.MemMachineAPI(with_config_api=enabled)
    assert ("/api/v2/config" in app.openapi()["paths"]) is enabled


def test_multiple_workers_remain_available_without_config_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMMACHINE_WORKERS", "2")
    app = app_module.MemMachineAPI()
    assert "/api/v2/config" not in app.openapi()["paths"]


def test_start_http_checks_late_config_api_enablement_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMMACHINE_CONFIG_API", "1")
    monkeypatch.setenv("MEMMACHINE_WORKERS", "2")
    loader = MagicMock()
    runner = MagicMock()
    monkeypatch.setattr(app_module, "load_configuration", loader)
    monkeypatch.setattr(app_module.uvicorn, "run", runner)
    with pytest.raises(ConfigurationError, match="single server process"):
        app_module.start_http()
    loader.assert_not_called()
    runner.assert_not_called()


def test_failed_start_can_retry_after_disabling_config_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMMACHINE_CONFIG_API", "1")
    monkeypatch.setenv("MEMMACHINE_WORKERS", "2")
    with pytest.raises(ConfigurationError):
        app_module.start_http()
    monkeypatch.delenv("MEMMACHINE_CONFIG_API")
    monkeypatch.setattr(app_module, "load_configuration", MagicMock())
    runner = MagicMock()
    monkeypatch.setattr(app_module.uvicorn, "run", runner)
    app_module.start_http()
    assert runner.call_args.kwargs["workers"] == 2
