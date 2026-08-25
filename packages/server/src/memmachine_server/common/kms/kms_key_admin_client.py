"""Abstract base class for a KMS key admin client."""

from abc import ABC, abstractmethod


class KMSKeyAdminClient(ABC):
    """Abstract base class for a KMS key admin client."""

    @abstractmethod
    async def create_key(self) -> str:
        """
        Create a new key in the backend.

        Returns:
            str:
                Opaque backend-managed key reference for the new key.
                The caller persists this value and uses it against
                exists_key, delete_key, and KMSCryptoClient.
        """
        raise NotImplementedError

    @abstractmethod
    async def exists_key(self, key_ref: str) -> bool:
        """
        Check whether a key exists.

        Args:
            key_ref (str):
                Opaque backend-managed key reference.

        Returns:
            bool:
                True if a key with key_ref currently exists and is
                usable, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_key(self, key_ref: str) -> None:
        """
        Delete a key.

        Deletion may not be immediate on all backends.

        Args:
            key_ref (str):
                Opaque backend-managed key reference.
        """
        raise NotImplementedError
