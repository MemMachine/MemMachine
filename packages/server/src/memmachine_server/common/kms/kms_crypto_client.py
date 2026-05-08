"""Abstract base class for a KMS cryptography client."""

from abc import ABC, abstractmethod


class KMSCryptoClient(ABC):
    """Abstract base class for a KMS cryptography client."""

    @abstractmethod
    async def encrypt(
        self,
        key_ref: str,
        plaintext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        """
        Encrypt bytes with the referenced key.

        Args:
            key_ref (str):
                Opaque backend-managed key reference.
            plaintext (bytes):
                Bytes to encrypt.
            associated_data (bytes | None):
                Additional authenticated data.

        Returns:
            bytes:
                Ciphertext for the supplied plaintext.
        """
        raise NotImplementedError

    @abstractmethod
    async def decrypt(
        self,
        key_ref: str,
        ciphertext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        """
        Decrypt bytes with the referenced key.

        Args:
            key_ref (str):
                Opaque backend-managed key reference.
            ciphertext (bytes):
                Bytes to decrypt.
            associated_data (bytes | None):
                Additional authenticated data.

        Returns:
            bytes:
                Plaintext for the supplied ciphertext.
        """
        raise NotImplementedError
