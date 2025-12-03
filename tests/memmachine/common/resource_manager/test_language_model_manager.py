import pytest
from pydantic import SecretStr

from memmachine.common.configuration.language_model_conf import (
    AmazonBedrockLanguageModelConf,
    LanguageModelsConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)
from memmachine.common.resource_manager.language_model_manager import (
    LanguageModelManager,
)


@pytest.fixture
def mock_conf():
    """Mock LanguageModelsConf with dummy configurations."""
    conf = LanguageModelsConf(
        openai_responses_language_model_confs={
            "openai_4o_mini": OpenAIResponsesLanguageModelConf(
                model="gpt-4o-mini",
                api_key=SecretStr("DUMMY_OPENAI_API_KEY_1"),
            ),
            "openai_3_5_turbo": OpenAIResponsesLanguageModelConf(
                model="gpt-3.5-turbo",
                api_key=SecretStr("DUMMY_OPENAI_API_KEY_2"),
            ),
        },
        amazon_bedrock_language_model_confs={
            "aws_model": AmazonBedrockLanguageModelConf(
                region="us-west-2",
                aws_access_key_id=SecretStr("DUMMY_AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=SecretStr("DUMMY_AWS_SECRET_ACCESS_KEY"),
                model_id="amazon.titan-embed-text-v2:0",
                additional_model_request_fields={},
            ),
        },
        openai_chat_completions_language_model_confs={
            "ollama_model": OpenAIChatCompletionsLanguageModelConf(
                model="llama3",
                api_key=SecretStr("DUMMY_OLLAMA_API_KEY"),
                base_url="http://localhost:11434/v1",
            ),
        },
    )
    return conf


@pytest.mark.asyncio
async def test_build_open_ai_model(mock_conf):
    builder = LanguageModelManager(mock_conf)
    await builder.build_all()

    assert "openai_4o_mini" in builder._language_models
    assert "openai_3_5_turbo" in builder._language_models

    model = builder.get_language_model("openai_4o_mini")
    assert model is not None


@pytest.mark.asyncio
async def test_build_aws_bedrock_model(mock_conf):
    builder = LanguageModelManager(mock_conf)
    await builder.build_all()

    assert "aws_model" in builder._language_models

    model = builder.get_language_model("aws_model")
    assert model is not None


@pytest.mark.asyncio
async def test_build_openai_chat_completions_model(mock_conf):
    builder = LanguageModelManager(mock_conf)
    await builder.build_all()

    assert "ollama_model" in builder._language_models

    model = builder.get_language_model("ollama_model")
    assert model is not None
