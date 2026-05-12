"""Integration tests for the Vault/OpenBao transit KMS crypto client."""

from collections.abc import Iterator
from typing import Any

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


@pytest.fixture(scope="module")
def transit_key_ref(vault_client_config: tuple[Any, dict[str, object]]) -> str:
    vault_client, _ = vault_client_config
    key_ref = "mmachine"
    vault_client.secrets.transit.create_key(
        name=key_ref,
        mount_point="transit",
        key_type="aes256-gcm96",
    )
    return key_ref


@pytest.mark.asyncio
async def test_transit_kms_client_round_trip(
    vault_client_config: tuple[Any, dict[str, object]],
    transit_key_ref: str,
) -> None:
    from memmachine_server.common.kms.vault_transit_kms_crypto_client import (
        VaultTransitKMSCryptoClient,
        VaultTransitKMSCryptoClientParams,
    )

    _, client_kwargs = vault_client_config
    params = VaultTransitKMSCryptoClientParams(
        client_kwargs=client_kwargs,
        mount_point="transit",
    )
    kms_client = VaultTransitKMSCryptoClient(params)

    ciphertext = await kms_client.encrypt(
        transit_key_ref,
        b"hello vault",
        associated_data=b"partition:context",
    )
    plaintext = await kms_client.decrypt(
        transit_key_ref,
        ciphertext,
        associated_data=b"partition:context",
    )

    assert plaintext == b"hello vault"
