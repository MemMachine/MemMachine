"""Tests for server middleware components."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memmachine.server.middleware import AccessLogMiddleware, RequestMetricsMiddleware


class FakeCounter:
    def __init__(self):
        self.calls = []

    def increment(self, value=1, labels=None):
        self.calls.append((value, labels))


class FakeHistogram:
    def __init__(self):
        self.calls = []

    def observe(self, value, labels=None):
        self.calls.append((value, labels))


class FakeMetricsFactory:
    def __init__(self):
        self.counter = FakeCounter()
        self.histogram = FakeHistogram()

    def get_counter(self, *args, **kwargs):
        return self.counter

    def get_histogram(self, *args, **kwargs):
        return self.histogram


@pytest.fixture
def app_with_access_log():
    app = FastAPI()
    app.add_middleware(AccessLogMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


def test_access_log_middleware_logs(caplog, app_with_access_log):
    caplog.set_level(logging.INFO)

    client = TestClient(app_with_access_log)
    response = client.get("/ping")

    assert response.status_code == 200

    # One access log entry
    records = [r for r in caplog.records if "method=GET" in r.message]
    assert len(records) == 1

    record = records[0]
    assert 'path="/ping"' in record.message
    assert "status=200" in record.message


@pytest.fixture
def fake_metrics_factory():
    return FakeMetricsFactory()


@pytest.fixture
def app_with_metrics(monkeypatch, fake_metrics_factory):
    app = FastAPI()

    # Patch MetricsFactoryIdMixin.get_metrics_factory
    monkeypatch.setattr(
        RequestMetricsMiddleware,
        "get_metrics_factory",
        lambda self: fake_metrics_factory,
    )

    app.add_middleware(RequestMetricsMiddleware)

    @app.get("/items/{item_id}")
    async def get_item(item_id: int):
        return {"item_id": item_id}

    return app


def test_request_metrics_success(app_with_metrics, fake_metrics_factory):
    client = TestClient(app_with_metrics)
    response = client.get("/items/123")

    assert response.status_code == 200

    # Counter
    assert len(fake_metrics_factory.counter.calls) == 1
    _, labels = fake_metrics_factory.counter.calls[0]

    assert labels["method"] == "GET"
    assert labels["path"] == "/items/{item_id}"
    assert labels["status"] == "200"

    # Histogram
    assert len(fake_metrics_factory.histogram.calls) == 1
    duration, labels = fake_metrics_factory.histogram.calls[0]

    assert duration > 0
    assert labels["status"] == "200"


@pytest.fixture
def app_with_error(monkeypatch, fake_metrics_factory):
    app = FastAPI()

    monkeypatch.setattr(
        RequestMetricsMiddleware,
        "get_metrics_factory",
        lambda self: fake_metrics_factory,
    )

    app.add_middleware(RequestMetricsMiddleware)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    return app


def test_request_metrics_on_exception(app_with_error, fake_metrics_factory):
    client = TestClient(app_with_error)

    with pytest.raises(RuntimeError):
        client.get("/boom")

    # Metrics still recorded
    assert len(fake_metrics_factory.counter.calls) == 1
    _, labels = fake_metrics_factory.counter.calls[0]

    assert labels["method"] == "GET"
    assert labels["path"] == "/boom"
    assert labels["status"] == "500"
