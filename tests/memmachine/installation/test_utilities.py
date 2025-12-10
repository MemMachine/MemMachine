import pytest

from memmachine.installation.utilities import ModelProvider


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("openai", ModelProvider.OPENAI),
        ("OpenAI", ModelProvider.OPENAI),
        ("OPENAI", ModelProvider.OPENAI),
        ("bedrock", ModelProvider.BEDROCK),
        ("Bedrock", ModelProvider.BEDROCK),
        ("ollama", ModelProvider.OLLAMA),
        ("Ollama", ModelProvider.OLLAMA),
        ("", ModelProvider.OPENAI),  # empty input defaults
        ("   ", ModelProvider.OPENAI),  # whitespace defaults
        (None, ModelProvider.OPENAI),  # None defaults
        ("invalid-provider", ModelProvider.OPENAI),  # invalid => OPENAI
    ],
)
def test_parse_model_provider(input_value, expected):
    assert ModelProvider.parse(input_value) == expected
