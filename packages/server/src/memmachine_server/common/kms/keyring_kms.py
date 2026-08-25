"""KMS clients backed by OS-native keyrings."""

from asyncio import to_thread
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable
from typing import override
from uuid import uuid4

import keyring
from pydantic import BaseModel, Field, field_validator

from memmachine_server.common.kms.kms_crypto_client import KMSCryptoClient
from memmachine_server.common.kms.kms_key_admin_client import KMSKeyAdminClient
from memmachine_server.common.payload_codec.defaults import (
    default_key_material_generator,
    default_payload_codec_factory,
)
from memmachine_server.common.payload_codec.payload_codec import (
    PayloadCodec,
    PayloadCodecFactory,
)

DEFAULT_KEYRING_KMS_SERVICE_NAME = "memmachine-kms"


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
        default=DEFAULT_KEYRING_KMS_SERVICE_NAME,
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


class KeyringKMSKeyAdminClientParams(BaseModel):
    """
    Parameters for KeyringKMSKeyAdminClient.

    Attributes:
        service_name (str):
            Keyring service namespace used to store key material.
        key_material_generator (Callable[[], bytes]):
            Generator for raw key material. Must be paired with a matching
            payload_codec_factory on the KeyringKMSCryptoClient that will
            consume the keys.
    """

    service_name: str = Field(
        default=DEFAULT_KEYRING_KMS_SERVICE_NAME,
        description="Keyring service namespace used to store key material",
        min_length=1,
    )
    key_material_generator: Callable[[], bytes] = Field(
        default=default_key_material_generator,
        description="Generator for raw key material",
    )

    @field_validator("service_name")
    @classmethod
    def _normalize_service_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("service_name cannot be empty")
        return normalized


class KeyringKMSKeyAdminClient(KMSKeyAdminClient):
    """KMS key admin client backed by OS-native keyrings."""

    def __init__(
        self,
        params: KeyringKMSKeyAdminClientParams,
    ) -> None:
        """
        Initialize the keyring-backed admin client.

        Args:
            params (KeyringKMSKeyAdminClientParams):
                Initialization parameters.
        """
        self._service_name = params.service_name
        self._key_material_generator = params.key_material_generator

    @override
    async def create_key(self) -> str:
        key_ref = str(uuid4())
        encoded_key = urlsafe_b64encode(
            self._key_material_generator(),
        ).decode("ascii")
        await to_thread(
            keyring.set_password,
            self._service_name,
            key_ref,
            encoded_key,
        )
        return key_ref

    @override
    async def exists_key(self, key_ref: str) -> bool:
        encoded_key = await to_thread(
            keyring.get_password,
            self._service_name,
            key_ref,
        )
        return encoded_key is not None

    @override
    async def delete_key(self, key_ref: str) -> None:
        await to_thread(
            keyring.delete_password,
            self._service_name,
            key_ref,
        )
