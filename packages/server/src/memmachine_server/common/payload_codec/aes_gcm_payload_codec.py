"""AES-GCM payload codec."""

from secrets import token_bytes
from typing import override

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .payload_codec import (
    PayloadCodec,
)


class AESGCMPayloadCodec(PayloadCodec):
    """Codec for AES-GCM payload encryption."""

    def __init__(
        self,
        key: bytes,
        *,
        nonce_size: int = 12,
        associated_data: bytes | None = None,
    ) -> None:
        """Initialize with a raw AES key."""
        if nonce_size <= 0:
            raise ValueError("nonce_size must be positive")

        self._aead = AESGCM(key)
        self._nonce_size = nonce_size
        self._associated_data = associated_data

    @override
    def encode(self, value: bytes) -> bytes:
        nonce = token_bytes(self._nonce_size)
        ciphertext = self._aead.encrypt(nonce, value, self._associated_data)
        return nonce + ciphertext

    @override
    def decode(self, value: bytes) -> bytes:
        if len(value) < self._nonce_size:
            raise ValueError("Encrypted payload is too short")
        nonce = value[: self._nonce_size]
        ciphertext = value[self._nonce_size :]
        return self._aead.decrypt(nonce, ciphertext, self._associated_data)
