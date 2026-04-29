"""Shared payload codec abstractions."""

from abc import ABC, abstractmethod


class PayloadCodec(ABC):
    """Byte-level codec for payloads."""

    @abstractmethod
    def encode(
        self,
        value: bytes,
    ) -> bytes:
        """
        Encode a payload.

        Args:
            value (bytes):
                Payload bytes to encode.

        Returns:
            bytes:
                Encoded payload bytes.
        """

    @abstractmethod
    def decode(
        self,
        value: bytes,
    ) -> bytes:
        """
        Decode a serialized payload.

        Args:
            value (bytes):
                Encoded payload bytes to decode.

        Returns:
            bytes:
                Payload bytes.
        """
