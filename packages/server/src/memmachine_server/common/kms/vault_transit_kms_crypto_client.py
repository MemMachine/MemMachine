"""HashiCorp Vault/OpenBao Transit-backed KMS crypto client implementation."""

from asyncio import to_thread
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Mapping
from threading import Lock
from typing import Any, cast, override

import hvac
from pydantic import BaseModel, Field, field_validator

from memmachine_server.common.kms.kms_crypto_client import KMSCryptoClient

DEFAULT_MOUNT_POINT = "transit"


class VaultTransitKMSCryptoClientParams(BaseModel):
    """
    Parameters for VaultTransitKMSCryptoClient.

    Attributes:
        client_args (tuple[Any, ...]):
            Positional arguments forwarded to `hvac.Client`.
        client_kwargs (dict[str, Any]):
            Keyword arguments forwarded to `hvac.Client`.
        mount_point (str):
            Transit mount point to use for crypto operations.
    """

    client_args: tuple[Any, ...] = Field(
        default_factory=tuple,
        description="Positional arguments forwarded to hvac.Client",
    )
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to hvac.Client",
    )
    mount_point: str = Field(
        default=DEFAULT_MOUNT_POINT,
        description="Transit mount point to use for crypto operations",
        min_length=1,
    )

    @field_validator("mount_point")
    @classmethod
    def _normalize_path_component(cls, value: str) -> str:
        normalized = value.strip("/")
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized


class VaultTransitKMSCryptoClient(KMSCryptoClient):
    """KMS client backed by a HashiCorp Vault/OpenBao transit engine."""

    def __init__(self, params: VaultTransitKMSCryptoClientParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._client = hvac.Client(*params.client_args, **params.client_kwargs)
        self._mount_point = params.mount_point

        self._transit = self._client.secrets.transit

        self._client_lock = Lock()

    @staticmethod
    def _encode_bytes(value: bytes) -> str:
        return urlsafe_b64encode(value).decode("ascii")

    @staticmethod
    def _decode_bytes(value: str) -> bytes:
        return urlsafe_b64decode(value.encode("ascii"))

    @staticmethod
    def _ensure_string(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Transit {field_name} must be a string")
        return value

    @staticmethod
    def _get_data_field(response: Mapping[str, object], field_name: str) -> object:
        try:
            data = response["data"]
        except (KeyError, TypeError) as err:
            raise ValueError(
                f"Transit response did not contain the expected `{field_name}` field"
            ) from err
        if not isinstance(data, Mapping):
            raise TypeError("Transit response data must be a mapping")
        try:
            return cast(Mapping[str, object], data)[field_name]
        except KeyError as err:
            raise ValueError(
                f"Transit response did not contain the expected `{field_name}` field"
            ) from err

    @override
    async def encrypt(
        self,
        key_ref: str,
        plaintext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        return await to_thread(
            self._encrypt_sync,
            key_ref,
            plaintext,
            associated_data,
        )

    def _encrypt_sync(
        self,
        key_ref: str,
        plaintext: bytes,
        associated_data: bytes | None,
    ) -> bytes:
        with self._client_lock:
            response = self._transit.encrypt_data(
                name=key_ref,
                plaintext=self._encode_bytes(plaintext),
                mount_point=self._mount_point,
                associated_data=(
                    self._encode_bytes(associated_data)
                    if associated_data is not None
                    else None
                ),
            )
        ciphertext = self._ensure_string(
            self._get_data_field(response, "ciphertext"),
            "ciphertext",
        )
        return ciphertext.encode("utf-8")

    @override
    async def decrypt(
        self,
        key_ref: str,
        ciphertext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        return await to_thread(
            self._decrypt_sync,
            key_ref,
            ciphertext,
            associated_data,
        )

    def _decrypt_sync(
        self,
        key_ref: str,
        ciphertext: bytes,
        associated_data: bytes | None,
    ) -> bytes:
        with self._client_lock:
            response = self._transit.decrypt_data(
                name=key_ref,
                ciphertext=ciphertext.decode("utf-8"),
                mount_point=self._mount_point,
                associated_data=(
                    self._encode_bytes(associated_data)
                    if associated_data is not None
                    else None
                ),
            )
        plaintext = self._ensure_string(
            self._get_data_field(response, "plaintext"),
            "plaintext",
        )
        return self._decode_bytes(plaintext)
