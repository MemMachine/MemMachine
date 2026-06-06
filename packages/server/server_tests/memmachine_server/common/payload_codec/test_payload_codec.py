"""Tests for payload codecs."""

import pytest
from cryptography.exceptions import InvalidTag

from memmachine_server.common.payload_codec.aes_gcm_payload_codec import (
    AESGCMPayloadCodec,
)
from memmachine_server.common.payload_codec.plaintext_payload_codec import (
    PlaintextPayloadCodec,
)


def test_plaintext_payload_codec_round_trip() -> None:
    codec = PlaintextPayloadCodec()
    value = b'{"type":"message","source":"User"}'

    encoded = codec.encode(value)
    decoded = codec.decode(encoded)

    assert encoded == value
    assert decoded == value


def test_aes_gcm_payload_codec_round_trip() -> None:
    codec = AESGCMPayloadCodec(b"0" * 32, associated_data=b"partition:context")
    value = b'{"type":"message","source":"User"}'

    encoded = codec.encode(value)
    decoded = codec.decode(encoded)

    assert encoded != value
    assert decoded == value

    wrong_codec = AESGCMPayloadCodec(
        b"0" * 32,
        associated_data=b"partition:block",
    )
    with pytest.raises(InvalidTag):
        wrong_codec.decode(encoded)


def test_aes_gcm_payload_codec_rejects_invalid_nonce_size() -> None:
    with pytest.raises(ValueError, match="nonce_size"):
        AESGCMPayloadCodec(b"0" * 32, nonce_size=0)


def test_aes_gcm_payload_codec_rejects_short_payload() -> None:
    codec = AESGCMPayloadCodec(b"0" * 32, associated_data=b"partition:context")

    # 12-byte nonce present but no room for the 16-byte tag.
    with pytest.raises(ValueError, match="too short"):
        codec.decode(b"\x00" * 12)


def test_aes_gcm_payload_codec_detects_tampered_ciphertext() -> None:
    codec = AESGCMPayloadCodec(b"0" * 32, associated_data=b"partition:context")
    value = b'{"type":"message","source":"User"}'

    encoded = bytearray(codec.encode(value))
    encoded[20] ^= 1  # flip a bit inside the ciphertext body

    with pytest.raises(InvalidTag):
        codec.decode(bytes(encoded))
