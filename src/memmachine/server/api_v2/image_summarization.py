"""Image summarization helpers for the API v2 server."""

from __future__ import annotations

import base64
import logging

import openai

from memmachine.common.configuration import Configuration
from memmachine.common.errors import ConfigurationError
from memmachine.common.resource_manager.resource_manager import ResourceManagerImpl

logger = logging.getLogger(__name__)


_IMAGE_SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes images. "
    "Respond in concise English."
)

_IMAGE_SUMMARY_USER_PROMPT = (
    "Summarize the key information in this image.\n"
    "Requirements: concise and objective; do not guess; if the image is unclear or the information is insufficient, say so."
)


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


async def summarize_image(
    *,
    config: Configuration,
    resources: ResourceManagerImpl,
    image_bytes: bytes,
    mime_type: str,
) -> str:
    """Summarize an uploaded image using an OpenAI-compatible chat-completions model."""
    model_id = (config.image_summarization_model or "").strip()
    if not model_id:
        raise ConfigurationError(
            "image_summarization_model is not configured, but an image was provided"
        )

    lm_confs = resources.config.resources.language_models
    if model_id not in lm_confs.openai_chat_completions_language_model_confs:
        raise ConfigurationError(
            "image_summarization_model must reference an 'openai-chat-completions' "
            f"language model id, got: {model_id!r}"
        )

    conf = lm_confs.get_openai_chat_completions_language_model_conf(model_id)

    client = openai.AsyncOpenAI(
        api_key=conf.api_key.get_secret_value(),
        base_url=conf.base_url,
    )

    data_url = _to_data_url(image_bytes, mime_type)

    print("Data URL:", data_url)  # Debugging line to check the data URL

    messages = [
        {"role": "system", "content": _IMAGE_SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _IMAGE_SUMMARY_USER_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    response = await client.chat.completions.create(
        model=conf.model,
        messages=messages,
        temperature=0,
    )

    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        logger.warning("Empty image summary returned by model '%s'", model_id)
    return summary
