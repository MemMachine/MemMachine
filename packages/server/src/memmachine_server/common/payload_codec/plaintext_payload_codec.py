"""Plaintext payload codec."""

from typing import override

from .payload_codec import PayloadCodec


class PlaintextPayloadCodec(PayloadCodec):
    """Codec for plaintext-serialized payloads."""

    @override
    def encode(self, value: bytes) -> bytes:
        return value

    @override
    def decode(self, value: bytes) -> bytes:
        return value
