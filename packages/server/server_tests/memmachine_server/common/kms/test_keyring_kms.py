"""Tests for the keyring-backed KMS clients."""

from base64 import urlsafe_b64decode
from uuid import UUID

import pytest
from cryptography.exceptions import InvalidTag
from keyring.errors import PasswordDeleteError

import memmachine_server.common.kms.keyring_kms as keyring_kms
from memmachine_server.common.kms.keyring_kms import (
    KeyringKMSCryptoClient,
    KeyringKMSCryptoClientParams,
    KeyringKMSKeyAdminClient,
    KeyringKMSKeyAdminClientParams,
)
from memmachine_server.common.payload_codec.aes_gcm_payload_codec import (
    AESGCMPayloadCodec,
)
from memmachine_server.common.payload_codec.payload_codec import PayloadCodec
from memmachine_server.common.payload_codec.plaintext_payload_codec import (
    PlaintextPayloadCodec,
)


class InMemoryKeyring:
    """Minimal in-memory keyring used to test the client logic."""

    def __init__(self) -> None:
        self.passwords: dict[tuple[str, str], str] = {}
        self.set_calls: list[tuple[str, str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []

    def set_password(self, service: str, username: str, password: str) -> None:
        self.set_calls.append((service, username, password))
        self.passwords[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        self.get_calls.append((service, username))
        return self.passwords.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self.delete_calls.append((service, username))
        if (service, username) not in self.passwords:
            raise PasswordDeleteError(f"Password not found in {service} for {username}")
        del self.passwords[(service, username)]


@pytest.fixture
def keyring_backend(monkeypatch: pytest.MonkeyPatch) -> InMemoryKeyring:
    backend = InMemoryKeyring()
    monkeypatch.setattr(keyring_kms.keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring_kms.keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring_kms.keyring, "delete_password", backend.delete_password)
    return backend


@pytest.mark.asyncio
async def test_keyring_kms_client_round_trip(keyring_backend: InMemoryKeyring) -> None:
    del keyring_backend
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )
    crypto = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    key_ref = await admin.create_key()
    ciphertext = await crypto.encrypt(
        key_ref,
        b"hello world",
        associated_data=b"partition:context",
    )
    plaintext = await crypto.decrypt(
        key_ref,
        ciphertext,
        associated_data=b"partition:context",
    )

    assert plaintext == b"hello world"


@pytest.mark.asyncio
async def test_keyring_kms_client_honors_paired_factory_and_generator(
    keyring_backend: InMemoryKeyring,
) -> None:
    del keyring_backend

    def aes_gcm_nonce16_factory(
        key: bytes,
        associated_data: bytes | None = None,
    ) -> PayloadCodec:
        return AESGCMPayloadCodec(
            key,
            nonce_size=16,
            associated_data=associated_data,
        )

    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )
    crypto = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(
            service_name="test-service",
            payload_codec_factory=aes_gcm_nonce16_factory,
        ),
    )

    key_ref = await admin.create_key()
    ciphertext = await crypto.encrypt(key_ref, b"hello world")
    plaintext = await crypto.decrypt(key_ref, ciphertext)

    assert plaintext == b"hello world"


@pytest.mark.asyncio
async def test_keyring_kms_client_honors_plaintext_pair(
    keyring_backend: InMemoryKeyring,
) -> None:
    del keyring_backend

    def empty_key_generator() -> bytes:
        return b""

    def plaintext_factory(
        key: bytes,
        associated_data: bytes | None = None,
    ) -> PayloadCodec:
        del key, associated_data
        return PlaintextPayloadCodec()

    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(
            service_name="test-service",
            key_material_generator=empty_key_generator,
        ),
    )
    crypto = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(
            service_name="test-service",
            payload_codec_factory=plaintext_factory,
        ),
    )

    key_ref = await admin.create_key()
    ciphertext = await crypto.encrypt(key_ref, b"hello world")
    plaintext = await crypto.decrypt(key_ref, ciphertext)

    assert ciphertext == b"hello world"
    assert plaintext == b"hello world"


@pytest.mark.asyncio
async def test_keyring_kms_client_rejects_bad_associated_data(
    keyring_backend: InMemoryKeyring,
) -> None:
    del keyring_backend
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )
    crypto = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    key_ref = await admin.create_key()
    ciphertext = await crypto.encrypt(
        key_ref,
        b"hello world",
        associated_data=b"partition:context",
    )

    with pytest.raises(InvalidTag):
        await crypto.decrypt(
            key_ref,
            ciphertext,
            associated_data=b"partition:wrong",
        )


@pytest.mark.asyncio
async def test_keyring_kms_client_requires_existing_key_for_decrypt(
    keyring_backend: InMemoryKeyring,
) -> None:
    del keyring_backend
    crypto = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    with pytest.raises(LookupError, match="No key material is stored"):
        await crypto.decrypt("missing_key", b"ciphertext")


@pytest.mark.asyncio
async def test_keyring_kms_client_requires_existing_key_for_encrypt(
    keyring_backend: InMemoryKeyring,
) -> None:
    del keyring_backend
    crypto = KeyringKMSCryptoClient(
        KeyringKMSCryptoClientParams(service_name="test-service"),
    )

    with pytest.raises(LookupError, match="No key material is stored"):
        await crypto.encrypt("missing_key", b"ciphertext")


@pytest.mark.asyncio
async def test_create_key_stores_random_32_byte_material(
    keyring_backend: InMemoryKeyring,
) -> None:
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )

    key_ref = await admin.create_key()

    assert len(keyring_backend.set_calls) == 1
    service, ref, encoded = keyring_backend.set_calls[0]
    assert service == "test-service"
    assert ref == key_ref
    assert len(urlsafe_b64decode(encoded.encode("ascii"))) == 32


@pytest.mark.asyncio
async def test_create_key_returns_distinct_uuid_refs(
    keyring_backend: InMemoryKeyring,
) -> None:
    del keyring_backend
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )

    refs = {await admin.create_key() for _ in range(20)}

    assert len(refs) == 20
    for ref in refs:
        UUID(ref)  # raises ValueError if not a valid UUID


@pytest.mark.asyncio
async def test_exists_key_returns_true_after_create(
    keyring_backend: InMemoryKeyring,
) -> None:
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )

    key_ref = await admin.create_key()

    assert await admin.exists_key(key_ref) is True
    assert keyring_backend.get_calls[-1] == ("test-service", key_ref)


@pytest.mark.asyncio
async def test_exists_key_returns_false_for_unknown_ref(
    keyring_backend: InMemoryKeyring,
) -> None:
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )

    assert await admin.exists_key("nonexistent_key") is False
    assert keyring_backend.get_calls == [("test-service", "nonexistent_key")]


@pytest.mark.asyncio
async def test_delete_key_removes_existing_entry(
    keyring_backend: InMemoryKeyring,
) -> None:
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )

    key_ref = await admin.create_key()
    assert await admin.exists_key(key_ref) is True

    await admin.delete_key(key_ref)

    assert await admin.exists_key(key_ref) is False
    assert keyring_backend.delete_calls == [("test-service", key_ref)]


@pytest.mark.asyncio
async def test_delete_key_raises_for_unknown_ref(
    keyring_backend: InMemoryKeyring,
) -> None:
    del keyring_backend
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name="test-service"),
    )

    with pytest.raises(PasswordDeleteError):
        await admin.delete_key("nonexistent_key")
