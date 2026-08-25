"""Payload codec configuration models and serialization helpers."""

import binascii
from base64 import b64decode, urlsafe_b64encode
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_serializer,
    field_validator,
)


def _decode_urlsafe_b64(value: str, field_name: str) -> bytes:
    try:
        return b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            f"{field_name} is not a valid base64url string: {exc}"
        ) from exc


class KMSEnvelopeParams(BaseModel):
    """Parameters for KMS envelope encryption/decryption."""

    model_config = ConfigDict(frozen=True)

    key_ref: str
    wrapped_dek: bytes
    associated_data: bytes | None = None

    @field_validator("wrapped_dek", mode="before")
    @classmethod
    def _decode_wrapped_dek(cls, value: object) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return _decode_urlsafe_b64(value, "wrapped_dek")
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
            return _decode_urlsafe_b64(value, "associated_data")
        raise TypeError("associated_data must be bytes, base64url string, or None")

    @field_serializer("associated_data")
    def _serialize_associated_data(self, value: bytes | None) -> str | None:
        if value is None:
            return None
        return urlsafe_b64encode(value).decode("ascii")


class PlaintextPayloadCodecConfig(BaseModel):
    """Codec config for plaintext-serialized payloads."""

    model_config = ConfigDict(frozen=True)

    type: Literal["plaintext"] = "plaintext"


class AESGCMPayloadCodecConfig(BaseModel):
    """Codec config for AES-GCM payload encryption."""

    model_config = ConfigDict(frozen=True)

    type: Literal["aes_gcm"] = "aes_gcm"
    envelope: KMSEnvelopeParams
    nonce_size: int = Field(default=12, ge=8, le=128)


# Union of codec configs whose DEK is recovered via KMS envelope decryption.
type KMSEnvelopePayloadCodecConfig = AESGCMPayloadCodecConfig


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
