import pytest

from memmachine.common.configuration.metrics_conf import (
    UnknownMetricsFactoryError,
    WithMetricsFactoryId,
)
from memmachine.common.metrics_factory import PrometheusMetricsFactory


@pytest.fixture(autouse=True)
def reset_factories(monkeypatch):
    """Clear global factories before each test."""
    monkeypatch.setattr(WithMetricsFactoryId, "_factories", {})


def test_default_factory_is_prometheus():
    cfg = WithMetricsFactoryId()
    factory = cfg.get_metrics_factory()

    assert isinstance(factory, PrometheusMetricsFactory)
    assert "prometheus" in cfg._factories
    assert cfg._factories["prometheus"] is factory  # same instance


def test_explicit_prometheus_id():
    cfg = WithMetricsFactoryId(metrics_factory_id="prometheus")
    factory = cfg.get_metrics_factory()

    assert isinstance(factory, PrometheusMetricsFactory)
    assert "prometheus" in cfg._factories


def test_memoized_factory_is_reused():
    cfg = WithMetricsFactoryId()

    factory1 = cfg.get_metrics_factory()
    factory2 = cfg.get_metrics_factory()

    assert factory1 is factory2  # cached


def test_unknown_factory_id_raises():
    cfg = WithMetricsFactoryId(metrics_factory_id="invalid_factory")

    with pytest.raises(UnknownMetricsFactoryError) as exc:
        cfg.get_metrics_factory()

    assert "Unknown MetricsFactory name" in str(exc.value)
