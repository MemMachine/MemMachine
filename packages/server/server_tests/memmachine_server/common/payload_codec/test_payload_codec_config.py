"""Tests for payload codec configuration."""

from memmachine_server.common.payload_codec import (
    decode_payload_codec_config,
    encode_payload_codec_config,
)
from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)


def test_plaintext_payload_codec_config_round_trip() -> None:
    config = PlaintextPayloadCodecConfig()

    serialized = encode_payload_codec_config(config)
    deserialized = decode_payload_codec_config(serialized)

    assert deserialized == config
