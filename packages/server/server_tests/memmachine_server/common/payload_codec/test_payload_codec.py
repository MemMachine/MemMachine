"""Tests for payload codecs."""

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
