import pytest
import yaml
from pydantic import SecretStr, ValidationError

from memmachine.common.configuration.language_model_conf import (
    AmazonBedrockLanguageModelConf,
    LanguageModelsConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)


@pytest.fixture
def openai_model_conf() -> dict:
    return {
        "provider": "openai-responses",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": "open-ai-key",
        },
    }


@pytest.fixture
def aws_model_conf() -> dict:
    return {
        "provider": "amazon-bedrock",
        "config": {
            "region": "us-west-2",
            "aws_access_key_id": "aws-key-id",
            "aws_secret_access_key": "aws-secret-key",
            "model_id": "openai.gpt-oss-20b-1:0",
        },
    }


@pytest.fixture
def ollama_model_conf() -> dict:
    return {
        "provider": "openai-chat-completions",
        "config": {
            "model": "llama3",
            "api_key": "EMPTY",
            "base_url": "http://host.docker.internal:11434/v1",
        },
    }


@pytest.fixture
def full_model_conf(openai_model_conf, aws_model_conf, ollama_model_conf) -> dict:
    return {
        "language_models": {
            "openai_model": openai_model_conf,
            "aws_model": aws_model_conf,
            "ollama_model": ollama_model_conf,
        },
    }


def test_valid_openai_model(openai_model_conf):
    conf = OpenAIResponsesLanguageModelConf(**openai_model_conf["config"])
    assert conf.model == "gpt-4o-mini"
    assert conf.api_key == SecretStr("open-ai-key")
    assert conf.max_retry_interval_seconds == 120


def test_valid_aws_model(aws_model_conf):
    conf = AmazonBedrockLanguageModelConf(**aws_model_conf["config"])
    assert conf.region == "us-west-2"
    assert conf.aws_access_key_id == SecretStr("aws-key-id")
    assert conf.aws_secret_access_key == SecretStr("aws-secret-key")
    assert conf.model_id == "openai.gpt-oss-20b-1:0"
    assert conf.max_retry_interval_seconds == 120


def test_valid_openai_chat_completions_model(ollama_model_conf):
    conf = OpenAIChatCompletionsLanguageModelConf(**ollama_model_conf["config"])
    assert conf.model == "llama3"
    assert conf.api_key == SecretStr("EMPTY")
    assert conf.base_url == "http://host.docker.internal:11434/v1"
    assert conf.max_retry_interval_seconds == 120


def test_full_language_model_conf(full_model_conf):
    conf = LanguageModelsConf.parse(full_model_conf)

    assert "openai_model" in conf.openai_responses_language_model_confs
    openai_conf = conf.openai_responses_language_model_confs["openai_model"]
    assert openai_conf.model == "gpt-4o-mini"

    assert "aws_model" in conf.amazon_bedrock_language_model_confs
    aws_conf = conf.amazon_bedrock_language_model_confs["aws_model"]
    assert aws_conf.region == "us-west-2"

    assert "ollama_model" in conf.openai_chat_completions_language_model_confs
    chat_completions_conf = conf.openai_chat_completions_language_model_confs[
        "ollama_model"
    ]
    assert chat_completions_conf.model == "llama3"


def test_get_language_model_names(full_model_conf):
    conf = LanguageModelsConf.parse(full_model_conf)

    assert conf.get_openai_responses_language_model_name() == "openai_model"
    assert conf.get_amazon_bedrock_language_model_name() == "aws_model"
    assert conf.get_openai_chat_completions_language_model_name() == "ollama_model"


def test_serialize_deserialize_language_model_conf(full_model_conf):
    conf = LanguageModelsConf.parse(full_model_conf)
    yaml_str = conf.to_yaml()
    conf_cp = LanguageModelsConf.parse(yaml.safe_load(yaml_str))
    assert conf == conf_cp
    assert len(conf.amazon_bedrock_language_model_confs) == len(
        conf_cp.amazon_bedrock_language_model_confs
    )
    assert len(conf.openai_responses_language_model_confs) == len(
        conf_cp.openai_responses_language_model_confs
    )
    assert len(conf.openai_chat_completions_language_model_confs) == len(
        conf_cp.openai_chat_completions_language_model_confs
    )


def test_missing_required_field_openai_model():
    conf_dict = {"model": "gpt-4o-mini"}
    with pytest.raises(ValidationError) as exc_info:
        OpenAIResponsesLanguageModelConf(**conf_dict)
    assert "field required" in str(exc_info.value).lower()


def test_invalid_base_url_in_openai_chat_completions_model():
    conf_dict = {
        "model": "llama3",
        "api_key": "EMPTY",
        "base_url": "invalid-url",
    }
    with pytest.raises(ValidationError) as exc_info:
        OpenAIChatCompletionsLanguageModelConf(**conf_dict)
    assert "invalid base url" in str(exc_info.value).lower()


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in [
        "MY_API_KEY",
        "MY_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_read_api_key_from_env(monkeypatch, openai_model_conf):
    monkeypatch.setenv("MY_API_KEY", "env-open-ai-key")
    openai_model_conf["config"]["api_key"] = "${MY_API_KEY}"
    conf = OpenAIResponsesLanguageModelConf(**openai_model_conf["config"])
    assert conf.model == "gpt-4o-mini"
    assert conf.api_key == SecretStr("env-open-ai-key")


def test_read_aws_keys_from_env(monkeypatch, aws_model_conf):
    monkeypatch.setenv("MY_KEY_ID", "my-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "access-key")
    aws_model_conf["config"]["aws_access_key_id"] = "${MY_KEY_ID}"
    aws_model_conf["config"]["aws_secret_access_key"] = ""
    conf = AmazonBedrockLanguageModelConf(**aws_model_conf["config"])
    assert conf.aws_access_key_id.get_secret_value() == "my-key-id"
    assert conf.aws_secret_access_key.get_secret_value() == "access-key"
    assert conf.aws_session_token is None
