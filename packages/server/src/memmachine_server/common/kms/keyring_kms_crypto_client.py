"""KMS cryptography client backed by OS-native keyrings."""

from asyncio import to_thread
from base64 import urlsafe_b64decode
from typing import override

import keyring
from pydantic import BaseModel, Field, field_validator

from memmachine_server.common.kms.kms_crypto_client import KMSCryptoClient
from memmachine_server.common.payload_codec.default_payload_codec_factory import (
    default_payload_codec_factory,
)
from memmachine_server.common.payload_codec.payload_codec import (
    PayloadCodec,
    PayloadCodecFactory,
)

DEFAULT_KEYRING_SERVICE_NAME = "memmachine-kms-crypto-client"


class KeyringKMSCryptoClientParams(BaseModel):
    """
    Parameters for KeyringKMSCryptoClient.

    Attributes:
        service_name (str):
            Keyring service namespace used to store key material.
        payload_codec_factory (PayloadCodecFactory):
            Factory used to create a payload codec from raw key material.
    """

    service_name: str = Field(
        default=DEFAULT_KEYRING_SERVICE_NAME,
        description="Keyring service namespace used to store key material",
        min_length=1,
    )
    payload_codec_factory: PayloadCodecFactory = Field(
        default=default_payload_codec_factory,
        description="Factory used to create a payload codec from raw key material",
    )

    @field_validator("service_name")
    @classmethod
    def _normalize_service_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("service_name cannot be empty")
        return normalized


class KeyringKMSCryptoClient(KMSCryptoClient):
    """KMS cryptography client backed by OS-native keyrings."""

    def __init__(
        self,
        params: KeyringKMSCryptoClientParams,
    ) -> None:
        """
        Initialize the keyring-backed client.

        Args:
            params (KeyringKMSCryptoClientParams):
                Initialization parameters.
        """
        self._service_name = params.service_name
        self._payload_codec_factory = params.payload_codec_factory

    @override
    async def encrypt(
        self,
        key_ref: str,
        plaintext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        """Encrypt bytes with the referenced key."""
        normalized_key_ref = self._normalize_key_ref(key_ref)
        return await to_thread(
            self._encrypt_sync,
            normalized_key_ref,
            plaintext,
            associated_data,
        )

    def _encrypt_sync(
        self,
        key_ref: str,
        plaintext: bytes,
        associated_data: bytes | None,
    ) -> bytes:
        return self._build_codec(key_ref, associated_data).encode(plaintext)

    @override
    async def decrypt(
        self,
        key_ref: str,
        ciphertext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        """Decrypt bytes with the referenced key."""
        normalized_key_ref = self._normalize_key_ref(key_ref)
        return await to_thread(
            self._decrypt_sync,
            normalized_key_ref,
            ciphertext,
            associated_data,
        )

    def _decrypt_sync(
        self,
        key_ref: str,
        ciphertext: bytes,
        associated_data: bytes | None,
    ) -> bytes:
        return self._build_codec(key_ref, associated_data).decode(ciphertext)

    @staticmethod
    def _normalize_key_ref(key_ref: str) -> str:
        if not key_ref.strip():
            raise ValueError("key_ref cannot be empty")
        return key_ref

    def _build_codec(
        self,
        key_ref: str,
        associated_data: bytes | None,
    ) -> PayloadCodec:
        encoded_key = keyring.get_password(self._service_name, key_ref)
        if encoded_key is None:
            raise KeyError(f"No key material is stored for key_ref {key_ref!r}")

        key = urlsafe_b64decode(encoded_key.encode("ascii"))
        return self._payload_codec_factory(key, associated_data)
