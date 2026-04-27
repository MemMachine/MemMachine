"""Tests for payload codec configuration."""

from base64 import urlsafe_b64encode

from memmachine_server.common.payload_codec import (
    decode_payload_codec_config,
    encode_payload_codec_config,
)
from memmachine_server.common.payload_codec.payload_codec_config import (
    AESGCMPayloadCodecConfig,
    PlaintextPayloadCodecConfig,
)


def test_plaintext_payload_codec_config_round_trip() -> None:
    config = PlaintextPayloadCodecConfig()

    serialized = encode_payload_codec_config(config)
    deserialized = decode_payload_codec_config(serialized)

    assert deserialized == config


def test_aes_gcm_payload_codec_config_round_trip() -> None:
    config = AESGCMPayloadCodecConfig(
        key_ref="partition_key",
        wrapped_dek=b"wrapped-dek-bytes",
        associated_data=b"partition:context",
        nonce_size=12,
    )

    serialized = encode_payload_codec_config(config)
    deserialized = decode_payload_codec_config(serialized)

    assert serialized["wrapped_dek"] == urlsafe_b64encode(b"wrapped-dek-bytes").decode(
        "ascii"
    )
    assert serialized["associated_data"] == urlsafe_b64encode(
        b"partition:context"
    ).decode("ascii")
    assert deserialized == config
