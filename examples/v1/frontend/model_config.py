PROVIDER_MODEL_MAP = {
    "openai": ["gpt-4.1-mini"],
    "anthropic": [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
    ],
    "deepseek": ["us.deepseek.r1-v1:0"],
    "meta": ["meta.llama3-8b-instruct-v1:0", "meta.llama3-70b-instruct-v1:0"],
    "mistral": [
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-7b-instruct-v0:2",
    ],
}
# "meta.llama4-maverick-17b-instruct-v1:0" (not currently working)

MODEL_TO_PROVIDER = {
    model: provider
    for provider, models in PROVIDER_MODEL_MAP.items()
    for model in models
}

MODEL_CHOICES = [model for models in PROVIDER_MODEL_MAP.values() for model in models]
