"""
Image summarizer service.

Owned by `MemMachine` instances.
"""

from __future__ import annotations

import base64
import logging

import openai

from memmachine.common.configuration import Configuration
from memmachine.common.errors import ConfigurationError
from memmachine.common.resource_manager.resource_manager import ResourceManagerImpl

logger = logging.getLogger(__name__)


class ImageSummarizer:
    """Summarize images using the configured OpenAI-compatible vision model."""

    _IMAGE_SUMMARY_SYSTEM_PROMPT = (
        "You are a helpful assistant that summarizes images. "
        "Respond in concise English."
    )
    _IMAGE_SUMMARY_USER_PROMPT = (
        "Summarize the key information in this image.\n"
        "Requirements: concise and objective; do not guess; if the image is unclear or the information is insufficient, say so."
    )

    def __init__(self, *, config: Configuration, resources: ResourceManagerImpl) -> None:
        """Create an ImageSummarizer bound to a config and resource manager."""
        self._config = config
        self._resources = resources

    @staticmethod
    def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    async def summarize_image(self, *, image_bytes: bytes, mime_type: str) -> str:
        """Summarize an uploaded image using a chat-completions model."""
        model_id = (self._config.image_summarization_model or "").strip()
        if not model_id:
            raise ConfigurationError(
                "image_summarization_model is not configured, but an image was provided"
            )

        lm_confs = self._resources.config.resources.language_models
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

        data_url = self._to_data_url(image_bytes, mime_type)
        messages = [
            {"role": "system", "content": self._IMAGE_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._IMAGE_SUMMARY_USER_PROMPT},
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
