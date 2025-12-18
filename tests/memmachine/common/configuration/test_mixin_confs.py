import pytest
import yaml
from pydantic import SecretStr, ValidationError

from memmachine.common.configuration.mixin_confs import (
    ApiKeyMixin,
    AWSCredentialsMixin,
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


@pytest.fixture(autouse=True)
def clear_key_password_env(monkeypatch):
    for var in ["MY_API_KEY", "MY_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "my-secret-value")
    mixin = ApiKeyMixin(api_key=SecretStr("$MY_API_KEY"))
    assert mixin.api_key.get_secret_value() == "my-secret-value"


def test_resolve_api_key_form2_from_env(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "another-secret-value")
    mixin = ApiKeyMixin(api_key=SecretStr("${MY_API_KEY}"))
    assert mixin.api_key.get_secret_value() == "another-secret-value"


def test_api_key_env_not_found(monkeypatch):
    monkeypatch.setenv("HER_API_KEY", "her-secret-value")
    mixin = ApiKeyMixin(api_key=SecretStr("$MY_API_KEY"))
    assert mixin.api_key.get_secret_value() == "$MY_API_KEY"


def test_api_key_without_env():
    mixin = ApiKeyMixin(api_key=SecretStr("plain-secret"))
    assert mixin.api_key.get_secret_value() == "plain-secret"


def test_password_mixin_resolve_from_env(monkeypatch):
    monkeypatch.setenv("MY_PASSWORD", "my-password-value")
    mixin = ApiKeyMixin(api_key=SecretStr("$MY_PASSWORD"))
    assert mixin.api_key.get_secret_value() == "my-password-value"


def test_password_mixin_env_not_found(monkeypatch):
    monkeypatch.setenv("HER_PASSWORD", "her-password-value")
    mixin = ApiKeyMixin(api_key=SecretStr("$MY_PASSWORD"))
    assert mixin.api_key.get_secret_value() == "$MY_PASSWORD"


def test_password_mixin_without_env():
    mixin = ApiKeyMixin(api_key=SecretStr("plain-password"))
    assert mixin.api_key.get_secret_value() == "plain-password"


def test_password_mixin_invalid_type():
    with pytest.raises(ValidationError) as exc:
        ApiKeyMixin(api_key=12345)  # type: ignore
    assert "should be a valid string" in str(exc.value)


@pytest.fixture(autouse=True)
def clear_aws_env(monkeypatch):
    for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        monkeypatch.delenv(var, raising=False)


def test_default_to_env_key_and_secret(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret-key")
    mixin = AWSCredentialsMixin()
    assert mixin.aws_access_key_id.get_secret_value() == "env-access-key"
    assert mixin.aws_secret_access_key.get_secret_value() == "env-secret-key"
    assert mixin.aws_session_token is None


def test_resolve_aws_credentials_from_env(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "my-api-key")
    monkeypatch.setenv("MY_PASSWORD", "my-password")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "my-token")
    mixin = AWSCredentialsMixin(
        aws_access_key_id=SecretStr("$MY_API_KEY"),
        aws_secret_access_key=SecretStr("${MY_PASSWORD}"),
    )
    assert mixin.aws_access_key_id.get_secret_value() == "my-api-key"
    assert mixin.aws_secret_access_key.get_secret_value() == "my-password"
    assert mixin.aws_session_token.get_secret_value() == "my-token"
