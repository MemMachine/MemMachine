from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from memmachine_common.api.config_spec import UpdateEpisodicMemorySpec
from pydantic import SecretStr

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.configuration.database_conf import (
    MilvusConf,
    Neo4jConf,
    QdrantConf,
    SqlAlchemyConf,
)
from memmachine_server.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConf,
    OpenAIEmbedderConf,
)
from memmachine_server.common.configuration.language_model_conf import (
    LiteLLMLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)
from memmachine_server.common.configuration.mixin_confs import YamlSerializableMixin
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.server.api_v2.config_service import ConfigService


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "LOG_FORMAT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Path:
    sample = Path("sample_configs/episodic_memory_config.cpu.sample")
    path = tmp_path / "config.yml"
    path.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "expression",
    [
        "$PERSISTENCE_TEST_KEY",
        "${PERSISTENCE_TEST_KEY}",
        "prefix-$PERSISTENCE_TEST_KEY-suffix",
    ],
)
@pytest.mark.parametrize(
    "provider", [OpenAIEmbedderConf, OpenAIResponsesLanguageModelConf, QdrantConf]
)
def test_secret_reference_preserved_in_yaml(
    provider: type[OpenAIEmbedderConf | OpenAIResponsesLanguageModelConf | QdrantConf],
    expression: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "synthetic-key")
    conf = provider(api_key=expression)
    assert conf.api_key.get_secret_value() == expression.replace(
        "${PERSISTENCE_TEST_KEY}", "synthetic-key"
    ).replace("$PERSISTENCE_TEST_KEY", "synthetic-key")
    assert conf.to_yaml_dict()["api_key"] == expression
    assert "synthetic-key" not in conf.to_yaml()
    assert "synthetic-key" not in conf.model_dump_json()
    assert "synthetic-key" not in repr(conf)


@pytest.mark.parametrize("provider", [Neo4jConf, SqlAlchemyConf])
def test_database_password_reference_preserved(
    provider: type[Neo4jConf | SqlAlchemyConf], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_PASSWORD", "synthetic-password")
    kwargs = {"password": "${PERSISTENCE_TEST_PASSWORD}"}
    if provider is SqlAlchemyConf:
        kwargs.update(dialect="sqlite", driver="aiosqlite", path=":memory:")
    conf = provider(**kwargs)
    assert conf.password == SecretStr("synthetic-password")
    assert SecretStr("synthetic-password") == conf.password
    assert hash(conf.password) == hash(SecretStr("synthetic-password"))
    assert conf.to_yaml_dict()["password"] == "${PERSISTENCE_TEST_PASSWORD}"


def test_explicit_replacement_discards_previous_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "synthetic-key")
    conf = OpenAIEmbedderConf(api_key="$PERSISTENCE_TEST_KEY")
    conf.api_key = SecretStr("synthetic-key")
    assert conf.to_yaml_dict()["api_key"] == "synthetic-key"
    changed = conf.model_copy(update={"api_key": SecretStr("replacement")})
    assert changed.to_yaml_dict()["api_key"] == "replacement"


def test_model_dump_revalidation_retains_original_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "synthetic-$nested-literal")
    conf = OpenAIEmbedderConf(api_key="$PERSISTENCE_TEST_KEY")
    monkeypatch.setenv("nested", "should-not-resolve-again")
    rebuilt = OpenAIEmbedderConf.model_validate(conf.model_dump())
    assert rebuilt.api_key.get_secret_value() == "synthetic-$nested-literal"
    assert rebuilt.to_yaml_dict()["api_key"] == "$PERSISTENCE_TEST_KEY"


def test_missing_and_empty_environment_references_are_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PERSISTENCE_TEST_KEY", raising=False)
    missing = OpenAIEmbedderConf(api_key="$PERSISTENCE_TEST_KEY")
    assert missing.api_key.get_secret_value() == "$PERSISTENCE_TEST_KEY"
    assert missing.to_yaml_dict()["api_key"] == "$PERSISTENCE_TEST_KEY"
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "")
    empty = OpenAIEmbedderConf(api_key="$PERSISTENCE_TEST_KEY")
    assert empty.api_key.get_secret_value() == ""
    assert empty.to_yaml_dict()["api_key"] == "$PERSISTENCE_TEST_KEY"


@pytest.mark.parametrize(
    ("field", "variable"),
    [
        ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
        ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
        ("aws_session_token", "AWS_SESSION_TOKEN"),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_aws_explicit_and_default_environment_credentials(
    field: str, variable: str, explicit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(variable, "synthetic-aws-value")
    kwargs = {field: "${" + variable + "}"} if explicit else {}
    conf = AmazonBedrockEmbedderConf(region="us-east-1", **kwargs)
    assert getattr(conf, field).get_secret_value() == "synthetic-aws-value"
    assert conf.to_yaml_dict()[field] == (
        "${" + variable + "}" if explicit else "$" + variable
    )
    assert "synthetic-aws-value" not in conf.to_yaml()


def test_milvus_references_survive_copy_and_nested_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_URI", "https://milvus.example.test")
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "synthetic-token")
    monkeypatch.setenv("PERSISTENCE_TEST_DB", "memory")
    conf = MilvusConf(
        uri="$PERSISTENCE_TEST_URI",
        token="${PERSISTENCE_TEST_KEY}",
        db_name="$PERSISTENCE_TEST_DB",
    )
    assert conf.uri == "https://milvus.example.test"
    assert conf.token == SecretStr("synthetic-token")
    for copied in (
        conf,
        deepcopy(conf),
        conf.model_copy(deep=True),
        MilvusConf.model_validate(conf.model_dump()),
    ):
        assert copied.to_yaml_dict()["uri"] == "$PERSISTENCE_TEST_URI"
        assert copied.to_yaml_dict()["token"] == "${PERSISTENCE_TEST_KEY}"
        assert copied.to_yaml_dict()["db_name"] == "$PERSISTENCE_TEST_DB"

    class Nested(YamlSerializableMixin):
        stores: list[MilvusConf]
        keys: dict[str, SecretStr]

    nested = Nested(stores=[conf], keys={"token": conf.token})
    assert nested.to_yaml_dict()["stores"][0]["uri"] == "$PERSISTENCE_TEST_URI"
    assert nested.to_yaml_dict()["keys"]["token"] == "${PERSISTENCE_TEST_KEY}"


def test_optional_literal_and_invalid_credentials_keep_existing_semantics() -> None:
    assert LiteLLMLanguageModelConf(model="openai/test", api_key=None).api_key is None
    assert (
        OpenAIEmbedderConf(api_key="literal-key").to_yaml_dict()["api_key"]
        == "literal-key"
    )
    with pytest.raises(ValueError, match="valid string"):
        OpenAIEmbedderConf.model_validate({"api_key": 123})


def test_real_configuration_service_save_and_reload_retain_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "first-synthetic-key")
    sample = Path("sample_configs/episodic_memory_config.cpu.sample")
    data = yaml.safe_load(sample.read_text(encoding="utf-8"))
    data["resources"]["embedders"]["openai_embedder"]["config"]["api_key"] = (
        "$PERSISTENCE_TEST_KEY"
    )
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    conf = Configuration.load_yml_file(str(path))
    resources = ResourceManagerImpl(conf)
    service = ConfigService(resources)
    service.update_episodic_memory_config(UpdateEpisodicMemorySpec(enabled=False))
    for _ in range(2):
        conf.save()
        persisted = path.read_text(encoding="utf-8")
        assert "$PERSISTENCE_TEST_KEY" in persisted
        assert "first-synthetic-key" not in persisted
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "rotated-synthetic-key")
    reloaded = Configuration.load_yml_file(str(path))
    assert (
        reloaded.resources.embedders.openai[
            "openai_embedder"
        ].api_key.get_secret_value()
        == "rotated-synthetic-key"
    )
    assert not reloaded.episodic_memory.enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_kind", ["embedder", "language_model"])
async def test_config_service_new_resources_retain_secret_references(
    resource_kind: str, sample_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "synthetic-added-key")
    path = sample_config_file
    conf = Configuration.load_yml_file(str(path))
    resources = ResourceManagerImpl(conf)
    service = ConfigService(resources)
    if resource_kind == "embedder":
        monkeypatch.setattr(resources.embedder_manager, "get_embedder", AsyncMock())
        await service.add_embedder(
            "added", "openai", {"api_key": "$PERSISTENCE_TEST_KEY"}
        )
        field = "embedders"
    else:
        monkeypatch.setattr(
            resources.language_model_manager, "get_language_model", AsyncMock()
        )
        await service.add_language_model(
            "added", "openai-responses", {"api_key": "$PERSISTENCE_TEST_KEY"}
        )
        field = "language_models"
    persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert (
        persisted["resources"][field]["added"]["config"]["api_key"]
        == "$PERSISTENCE_TEST_KEY"
    )
    assert "synthetic-added-key" not in path.read_text(encoding="utf-8")


def test_invalid_referenced_config_does_not_modify_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSISTENCE_TEST_KEY", "synthetic-key")
    path = tmp_path / "invalid.yml"
    original = "resources: {}\ninvalid: true\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="episodic_memory"):
        Configuration.load_yml_file(str(path))
    assert path.read_text(encoding="utf-8") == original
