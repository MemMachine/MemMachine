"""Integration tests for the Vault/OpenBao transit KMS clients."""

from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from testcontainers.vault import VaultContainer

from server_tests.memmachine_server.conftest import is_docker_available

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def vault_client_config() -> Iterator[tuple[Any, dict[str, object]]]:
    """Start a dev Vault container and return an authenticated hvac client."""
    if not is_docker_available():
        pytest.skip("Docker is not available")

    hvac = pytest.importorskip("hvac")
    with VaultContainer("hashicorp/vault:1.16.1").with_command(
        "server -dev -dev-root-token-id=toor -dev-listen-address=0.0.0.0:8200"
    ) as vault_container:
        client = hvac.Client(
            url=vault_container.get_connection_url(),
            token=vault_container.root_token,
        )
        if not client.is_authenticated():
            pytest.fail("Vault client failed to authenticate")

        client.sys.enable_secrets_engine(
            backend_type="transit",
            path="transit",
        )
        yield (
            client,
            {
                "url": vault_container.get_connection_url(),
                "token": vault_container.root_token,
            },
        )


@pytest.fixture
def admin_client(
    vault_client_config: tuple[Any, dict[str, object]],
) -> Any:
    from memmachine_server.common.kms.vault_transit_kms import (
        VaultTransitKMSKeyAdminClient,
        VaultTransitKMSKeyAdminClientParams,
    )

    _, client_kwargs = vault_client_config
    return VaultTransitKMSKeyAdminClient(
        VaultTransitKMSKeyAdminClientParams(
            client_kwargs=client_kwargs,
            mount_point="transit",
        ),
    )


@pytest.fixture
def crypto_client(
    vault_client_config: tuple[Any, dict[str, object]],
) -> Any:
    from memmachine_server.common.kms.vault_transit_kms import (
        VaultTransitKMSCryptoClient,
        VaultTransitKMSCryptoClientParams,
    )

    _, client_kwargs = vault_client_config
    return VaultTransitKMSCryptoClient(
        VaultTransitKMSCryptoClientParams(
            client_kwargs=client_kwargs,
            mount_point="transit",
        ),
    )


@pytest.mark.asyncio
async def test_transit_kms_client_round_trip(
    admin_client: Any,
    crypto_client: Any,
) -> None:
    key_ref = await admin_client.create_key()
    try:
        ciphertext = await crypto_client.encrypt(
            key_ref,
            b"hello vault",
            associated_data=b"partition:context",
        )
        plaintext = await crypto_client.decrypt(
            key_ref,
            ciphertext,
            associated_data=b"partition:context",
        )

        assert plaintext == b"hello vault"
    finally:
        await admin_client.delete_key(key_ref)


@pytest.mark.asyncio
async def test_transit_kms_admin_create_key_returns_distinct_uuid_refs(
    admin_client: Any,
) -> None:
    refs: list[str] = []
    try:
        for _ in range(5):
            ref = await admin_client.create_key()
            UUID(ref)  # raises ValueError if not a valid UUID
            refs.append(ref)

        assert len(set(refs)) == len(refs)
    finally:
        for ref in refs:
            await admin_client.delete_key(ref)


@pytest.mark.asyncio
async def test_transit_kms_admin_exists_key_true_after_create(
    admin_client: Any,
) -> None:
    key_ref = await admin_client.create_key()
    try:
        assert await admin_client.exists_key(key_ref) is True
    finally:
        await admin_client.delete_key(key_ref)


@pytest.mark.asyncio
async def test_transit_kms_admin_exists_key_false_for_unknown_ref(
    admin_client: Any,
) -> None:
    assert await admin_client.exists_key(str(UUID(int=0))) is False


@pytest.mark.asyncio
async def test_transit_kms_admin_delete_key_removes_key(
    admin_client: Any,
) -> None:
    key_ref = await admin_client.create_key()
    assert await admin_client.exists_key(key_ref) is True

    await admin_client.delete_key(key_ref)

    assert await admin_client.exists_key(key_ref) is False
