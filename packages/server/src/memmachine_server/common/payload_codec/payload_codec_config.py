"""Payload codec configuration models and serialization helpers."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    TypeAdapter,
)


class PlaintextPayloadCodecConfig(BaseModel):
    """Codec config for plaintext-serialized payloads."""

    type: Literal["plaintext"] = "plaintext"


PayloadCodecConfigUnion = PlaintextPayloadCodecConfig

PayloadCodecConfig = Annotated[PayloadCodecConfigUnion, Field(discriminator="type")]

_PAYLOAD_CODEC_CONFIG_ADAPTER = TypeAdapter(PayloadCodecConfig)


def encode_payload_codec_config(config: PayloadCodecConfig) -> dict[str, JsonValue]:
    """Encode a codec config to JSON-compatible data."""
    return _PAYLOAD_CODEC_CONFIG_ADAPTER.dump_python(config, mode="json")


def decode_payload_codec_config(
    encoded: Mapping[str, JsonValue],
) -> PayloadCodecConfig:
    """Decode a codec config from JSON-compatible data."""
    return _PAYLOAD_CODEC_CONFIG_ADAPTER.validate_python(encoded)
