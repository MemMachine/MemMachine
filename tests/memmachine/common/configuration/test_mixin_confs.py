import pytest
import yaml
from pydantic import SecretStr

from memmachine.common.configuration.mixin_confs import (
    MetricsFactoryIdMixin,
    UnknownMetricsFactoryError,
    YamlSerializableMixin,
)
from memmachine.common.metrics_factory import PrometheusMetricsFactory


@pytest.fixture(autouse=True)
def reset_factories(monkeypatch):
    """Clear global factories before each test."""
    monkeypatch.setattr(MetricsFactoryIdMixin, "_factories", {})


def test_default_factory_is_prometheus():
    cfg = MetricsFactoryIdMixin()
    factory = cfg.get_metrics_factory()

    assert isinstance(factory, PrometheusMetricsFactory)
    assert "prometheus" in cfg._factories
    assert cfg._factories["prometheus"] is factory  # same instance


def test_explicit_prometheus_id():
    cfg = MetricsFactoryIdMixin(metrics_factory_id="prometheus")
    factory = cfg.get_metrics_factory()

    assert isinstance(factory, PrometheusMetricsFactory)
    assert "prometheus" in cfg._factories


def test_memoized_factory_is_reused():
    cfg = MetricsFactoryIdMixin()

    factory1 = cfg.get_metrics_factory()
    factory2 = cfg.get_metrics_factory()

    assert factory1 is factory2  # cached


def test_unknown_factory_id_raises():
    cfg = MetricsFactoryIdMixin(metrics_factory_id="invalid_factory")

    with pytest.raises(UnknownMetricsFactoryError) as exc:
        cfg.get_metrics_factory()

    assert "Unknown MetricsFactory name" in str(exc.value)


class SimpleConfig(YamlSerializableMixin):
    api_key: SecretStr
    name: str
    count: int


class NestedConfig(YamlSerializableMixin):
    secret_map: dict[str, SecretStr]
    secret_list: list[SecretStr]
    regular_value: str


def test_to_yaml_dict_basic_secret_unwrap():
    cfg = SimpleConfig(
        api_key=SecretStr("real-secret"),
        name="test",
        count=123,
    )

    d = cfg.to_yaml_dict()

    assert d["api_key"] == "real-secret"
    assert d["name"] == "test"
    assert d["count"] == 123


def test_to_yaml_dict_nested_secret_unwrap():
    cfg = NestedConfig(
        secret_map={"k1": SecretStr("v1"), "k2": SecretStr("v2")},
        secret_list=[SecretStr("a"), SecretStr("b")],
        regular_value="ok",
    )

    d = cfg.to_yaml_dict()

    assert d == {
        "secret_map": {"k1": "v1", "k2": "v2"},
        "secret_list": ["a", "b"],
        "regular_value": "ok",
    }


def test_to_yaml_output_matches_dict():
    cfg = SimpleConfig(
        api_key=SecretStr("abc123"),
        name="hello",
        count=42,
    )

    yaml_text = cfg.to_yaml()
    yaml_loaded = yaml.safe_load(yaml_text)

    assert yaml_loaded == cfg.to_yaml_dict()

    cfg_cp = SimpleConfig(**yaml_loaded)
    assert cfg_cp.api_key.get_secret_value() == "abc123"
    assert cfg_cp.name == "hello"
    assert cfg_cp.count == 42


def test_no_secretstr_instance_in_yaml_output():
    cfg = SimpleConfig(
        api_key=SecretStr("top-secret"),
        name="x",
        count=1,
    )

    yaml_text = cfg.to_yaml()

    # Should NOT contain the masked format or the class name
    assert "********" not in yaml_text
    assert "SecretStr" not in yaml_text

    # Should contain real value
    assert "top-secret" in yaml_text


def test_deep_recursive_unwrap():
    class DeepConfig(YamlSerializableMixin):
        value: dict

    cfg = DeepConfig(
        value={
            "level1": {
                "level2": {
                    "secret": SecretStr("deep-secret"),
                    "list": [1, SecretStr("abc"), {"x": SecretStr("yyy")}],
                }
            }
        }
    )

    d = cfg.to_yaml_dict()

    assert d["value"]["level1"]["level2"]["secret"] == "deep-secret"
    assert d["value"]["level1"]["level2"]["list"][1] == "abc"
    assert d["value"]["level1"]["level2"]["list"][2]["x"] == "yyy"
