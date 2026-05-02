"""Payload codec configuration models and serialization helpers."""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    TypeAdapter,
    field_serializer,
    field_validator,
)


class PlaintextPayloadCodecConfig(BaseModel):
    """Codec config for plaintext-serialized payloads."""

    type: Literal["plaintext"] = "plaintext"


class KMSEnvelopePayloadCodecConfig(BaseModel):
    """Shared fields for codec configs that rely on KMS envelope encryption."""

    key_ref: str
    wrapped_dek: bytes
    associated_data: bytes | None = None

    @field_validator("wrapped_dek", mode="before")
    @classmethod
    def _decode_wrapped_dek(cls, value: object) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return urlsafe_b64decode(value.encode("ascii"))
        raise TypeError("wrapped_dek must be bytes or a base64url string")

    @field_serializer("wrapped_dek")
    def _serialize_wrapped_dek(self, value: bytes) -> str:
        return urlsafe_b64encode(value).decode("ascii")

    @field_validator("associated_data", mode="before")
    @classmethod
    def _decode_associated_data(cls, value: object) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return urlsafe_b64decode(value.encode("ascii"))
        raise TypeError("associated_data must be bytes, base64url string, or None")

    @field_serializer("associated_data")
    def _serialize_associated_data(self, value: bytes | None) -> str | None:
        if value is None:
            return None
        return urlsafe_b64encode(value).decode("ascii")


class AESGCMPayloadCodecConfig(KMSEnvelopePayloadCodecConfig):
    """Codec config for AES-GCM payload encryption."""

    type: Literal["aes_gcm"] = "aes_gcm"
    nonce_size: int = 12


PayloadCodecConfigUnion = PlaintextPayloadCodecConfig | AESGCMPayloadCodecConfig

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
