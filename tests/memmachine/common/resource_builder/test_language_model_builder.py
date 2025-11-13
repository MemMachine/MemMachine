import pytest
from pydantic import SecretStr

from memmachine.common.configuration.model_conf import (
    AwsBedrockModelConf,
    LanguageModelConf,
    OpenAICompatibleModelConf,
    OpenAIModelConf,
)
from memmachine.common.resource_mgr.language_model_mgr import LanguageModelMgr


@pytest.fixture
def mock_conf():
    """Mock LanguageModelConf with dummy configurations."""
    conf = LanguageModelConf(
        openai_confs={
            "openai_4o_mini": OpenAIModelConf(
                model="gpt-4o-mini",
                api_key=SecretStr("DUMMY_OPENAI_API_KEY_1"),
            ),
            "openai_3_5_turbo": OpenAIModelConf(
                model="gpt-3.5-turbo",
                api_key=SecretStr("DUMMY_OPENAI_API_KEY_2"),
            ),
        },
        aws_bedrock_confs={
            "aws_model": AwsBedrockModelConf(
                region="us-west-2",
                aws_access_key_id=SecretStr("DUMMY_AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=SecretStr("DUMMY_AWS_SECRET_ACCESS_KEY"),
                model_id="amazon.titan-embed-text-v2:0",
                additional_model_request_fields={},
            ),
        },
        openai_compatible_confs={
            "ollama_model": OpenAICompatibleModelConf(
                model="llama3",
                api_key=SecretStr("DUMMY_OLLAMA_API_KEY"),
                base_url="http://localhost:11434/v1",
            ),
        },
    )
    return conf


def test_build_open_ai_model(mock_conf):
    builder = LanguageModelMgr(mock_conf)
    builder._build_openai_model()

    assert "openai_4o_mini" in builder.openai_model
    assert "openai_3_5_turbo" in builder.openai_model

    model = builder.get_model("openai_4o_mini")
    assert model is not None


def test_build_aws_bedrock_model(mock_conf):
    builder = LanguageModelMgr(mock_conf)
    builder._build_aws_bedrock_model()

    assert "aws_model" in builder.aws_bedrock_model

    model = builder.get_model("aws_model")
    assert model is not None


def test_build_openai_compatible_model(mock_conf):
    builder = LanguageModelMgr(mock_conf)
    builder._build_openai_compatible_model()

    assert "ollama_model" in builder.openai_compatible_model

    model = builder.get_model("ollama_model")
    assert model is not None
