import os

PROVIDER_MODEL_MAP = {
    "openai": [
        "gpt-4.1-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    "anthropic": [
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-opus-4-20250514-v1:0",
    ],
    "google": [
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
    ],
}


MODEL_TO_PROVIDER = {
    model: provider
    for provider, models in PROVIDER_MODEL_MAP.items()
    for model in models
}

# Model display names with categories
MODEL_DISPLAY_NAMES = {
    "gpt-4.1-mini": "OpenAI - GPT-4.1 Mini",
    "gpt-5": "OpenAI - GPT-5",
    "gpt-5-mini": "OpenAI - GPT-5 Mini",
    "gpt-5-nano": "OpenAI - GPT-5 Nano",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "AWS Bedrock - Anthropic - Claude Haiku 4.5",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "AWS Bedrock - Anthropic - Claude Sonnet 4.5",
    "anthropic.claude-opus-4-20250514-v1:0": "AWS Bedrock - Anthropic - Claude Opus 4",
    "gemini-3-pro-preview": "Google - Gemini 3 Pro (Preview)",
    "gemini-2.5-pro": "Google - Gemini 2.5 Pro",
}

MODEL_CHOICES = [model for models in PROVIDER_MODEL_MAP.values() for model in models]

# Inference profile ARNs for provisioned throughput models
# Read from environment variables
MODEL_TO_INFERENCE_PROFILE_ARN = {}
# Claude Haiku 4.5
haiku_arn = os.getenv("BEDROCK_HAIKU_4_5_ARN", "").strip()
if haiku_arn:
    MODEL_TO_INFERENCE_PROFILE_ARN["anthropic.claude-haiku-4-5-20251001-v1:0"] = (
        haiku_arn
    )

# Claude Sonnet 4.5
sonnet_arn = os.getenv("BEDROCK_SONNET_4_5_ARN", "").strip()
if sonnet_arn:
    MODEL_TO_INFERENCE_PROFILE_ARN["anthropic.claude-sonnet-4-5-20250929-v1:0"] = (
        sonnet_arn
    )

# Claude Opus 4
opus_arn = os.getenv("BEDROCK_OPUS_4_ARN", "").strip()
if opus_arn:
    MODEL_TO_INFERENCE_PROFILE_ARN["anthropic.claude-opus-4-20250514-v1:0"] = opus_arn
