"""AES-GCM payload codec."""

from secrets import token_bytes
from typing import override

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .payload_codec import (
    PayloadCodec,
)

# AESGCM in `cryptography` produces a fixed 128-bit authentication tag.
_AES_GCM_TAG_SIZE = 16


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
        if not 8 <= nonce_size <= 128:
            raise ValueError("nonce_size must be between 8 and 128 bytes")

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
        if len(value) < self._nonce_size + _AES_GCM_TAG_SIZE:
            raise ValueError("Encrypted payload is too short")
        nonce = value[: self._nonce_size]
        ciphertext = value[self._nonce_size :]
        return self._aead.decrypt(nonce, ciphertext, self._associated_data)
