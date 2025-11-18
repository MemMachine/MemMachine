import pytest
from pydantic import SecretStr, ValidationError

from memmachine.common.configuration.model_conf import (
    AmazonBedrockLanguageModelConf,
    LanguageModelConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)


@pytest.fixture
def openai_model_conf() -> dict:
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "open-ai-key",
    }


@pytest.fixture
def aws_model_conf() -> dict:
    return {
        "provider": "amazon-bedrock",
        "region": "us-west-2",
        "aws_access_key_id": "aws-key-id",
        "aws_secret_access_key": "aws-secret-key",
        "model_id": "openai.gpt-oss-20b-1:0",
    }


@pytest.fixture
def ollama_model_conf() -> dict:
    return {
        "provider": "openai-chat-completions",
        "model": "llama3",
        "api_key": "EMPTY",
        "base_url": "http://host.docker.internal:11434/v1",
    }


@pytest.fixture
def full_model_conf(openai_model_conf, aws_model_conf, ollama_model_conf) -> dict:
    return {
        "model": {
            "openai_model": openai_model_conf,
            "aws_model": aws_model_conf,
            "ollama_model": ollama_model_conf,
        },
    }


def test_valid_openai_model(openai_model_conf):
    conf = OpenAIResponsesLanguageModelConf(**openai_model_conf)
    assert conf.model == "gpt-4o-mini"
    assert conf.api_key == SecretStr("open-ai-key")
    assert conf.max_retry_interval_seconds == 120


def test_valid_aws_model(aws_model_conf):
    conf = AmazonBedrockLanguageModelConf(**aws_model_conf)
    assert conf.region == "us-west-2"
    assert conf.aws_access_key_id == SecretStr("aws-key-id")
    assert conf.aws_secret_access_key == SecretStr("aws-secret-key")
    assert conf.model_id == "openai.gpt-oss-20b-1:0"
    assert conf.max_retry_interval_seconds == 120


def test_valid_openai_chat_completions_model(ollama_model_conf):
    conf = OpenAIChatCompletionsLanguageModelConf(**ollama_model_conf)
    assert conf.model == "llama3"
    assert conf.api_key == SecretStr("EMPTY")
    assert conf.base_url == "http://host.docker.internal:11434/v1"
    assert conf.max_retry_interval_seconds == 120


def test_full_language_model_conf(full_model_conf):
    conf = LanguageModelConf.parse_language_model_conf(full_model_conf)

    assert "openai_model" in conf.openai_confs
    openai_conf = conf.openai_confs["openai_model"]
    assert openai_conf.model == "gpt-4o-mini"

    assert "aws_model" in conf.aws_bedrock_confs
    aws_conf = conf.aws_bedrock_confs["aws_model"]
    assert aws_conf.region == "us-west-2"

    assert "ollama_model" in conf.openai_chat_completions_confs
    chat_completions_conf = conf.openai_chat_completions_confs["ollama_model"]
    assert chat_completions_conf.model == "llama3"


def test_missing_required_field_openai_model():
    conf_dict = {"provider": "openai", "model": "gpt-4o-mini"}  # Missing api_key
    with pytest.raises(ValidationError) as exc_info:
        OpenAIResponsesLanguageModelConf(**conf_dict)
    assert "field required" in str(exc_info.value).lower()


def test_invalid_base_url_in_openai_chat_completions_model():
    conf_dict = {
        "provider": "openai-chat-completions",
        "model": "llama3",
        "api_key": "EMPTY",
        "base_url": "invalid-url",
    }
    with pytest.raises(ValidationError) as exc_info:
        OpenAIChatCompletionsLanguageModelConf(**conf_dict)
    assert "invalid base url" in str(exc_info.value).lower()
