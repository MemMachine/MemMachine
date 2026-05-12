"""Tests for the keyring-backed KMS crypto client."""

from base64 import urlsafe_b64encode

import pytest
from cryptography.exceptions import InvalidTag

import memmachine_server.common.kms.keyring_kms_crypto_client as kms_module
from memmachine_server.common.kms.keyring_kms_crypto_client import (
    KeyringKMSCryptoClient,
    KeyringKMSCryptoClientParams,
)
from memmachine_server.common.payload_codec.plaintext_payload_codec import (
    PlaintextPayloadCodec,
)


class InMemoryKeyring:
    """Minimal in-memory keyring used to test the client logic."""

    def __init__(self) -> None:
        self.passwords: dict[tuple[str, str], str] = {}
        self.get_calls: list[tuple[str, str]] = []

    def get_password(self, service: str, username: str) -> str | None:
        self.get_calls.append((service, username))
        return self.passwords.get((service, username))


@pytest.fixture
def keyring_backend(monkeypatch: pytest.MonkeyPatch) -> InMemoryKeyring:
    backend = InMemoryKeyring()
    monkeypatch.setattr(kms_module.keyring, "get_password", backend.get_password)
    return backend


def seed_key_material(
    backend: InMemoryKeyring,
    service_name: str,
    key_ref: str,
    key_material: bytes,
) -> None:
    backend.passwords[(service_name, key_ref)] = urlsafe_b64encode(
        key_material,
    ).decode("ascii")


def plaintext_payload_codec_factory(
    key: bytes,
    associated_data: bytes | None = None,
) -> PlaintextPayloadCodec:
    del key, associated_data
    return PlaintextPayloadCodec()


@pytest.mark.asyncio
async def test_keyring_kms_client_round_trip(keyring_backend: InMemoryKeyring) -> None:
    seed_key_material(
        keyring_backend,
        "test-service",
        "partition_key",
        b"0" * 32,
    )
    kms_client = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    ciphertext = await kms_client.encrypt(
        "partition_key",
        b"hello world",
        associated_data=b"partition:context",
    )
    plaintext = await kms_client.decrypt(
        "partition_key",
        ciphertext,
        associated_data=b"partition:context",
    )

    assert plaintext == b"hello world"
    assert keyring_backend.get_calls == [
        ("test-service", "partition_key"),
        ("test-service", "partition_key"),
    ]


@pytest.mark.asyncio
async def test_keyring_kms_client_uses_configured_payload_codec(
    keyring_backend: InMemoryKeyring,
) -> None:
    seed_key_material(
        keyring_backend,
        "test-service",
        "partition_key",
        b"not-an-aes-key-size",
    )
    kms_client = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(
            service_name="test-service",
            payload_codec_factory=plaintext_payload_codec_factory,
        ),
    )

    ciphertext = await kms_client.encrypt(
        "partition_key",
        b"hello world",
        associated_data=b"partition:context",
    )
    plaintext = await kms_client.decrypt(
        "partition_key",
        ciphertext,
        associated_data=b"partition:context",
    )

    assert ciphertext == b"hello world"
    assert plaintext == b"hello world"


@pytest.mark.asyncio
async def test_keyring_kms_client_rejects_bad_associated_data(
    keyring_backend: InMemoryKeyring,
) -> None:
    seed_key_material(
        keyring_backend,
        "test-service",
        "partition_key",
        b"0" * 32,
    )
    kms_client = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    ciphertext = await kms_client.encrypt(
        "partition_key",
        b"hello world",
        associated_data=b"partition:context",
    )

    with pytest.raises(InvalidTag):
        await kms_client.decrypt(
            "partition_key",
            ciphertext,
            associated_data=b"partition:wrong",
        )


@pytest.mark.asyncio
async def test_keyring_kms_client_requires_existing_key_for_decrypt(
    keyring_backend: InMemoryKeyring,
) -> None:
    kms_client = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    with pytest.raises(LookupError, match="No key material is stored"):
        await kms_client.decrypt("missing_key", b"ciphertext")


@pytest.mark.asyncio
async def test_keyring_kms_client_requires_existing_key_for_encrypt(
    keyring_backend: InMemoryKeyring,
) -> None:
    kms_client = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    with pytest.raises(LookupError, match="No key material is stored"):
        await kms_client.encrypt("missing_key", b"ciphertext")
