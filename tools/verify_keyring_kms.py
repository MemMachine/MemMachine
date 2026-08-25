"""Verify the keyring-backed KMS clients against the real OS keyring."""

import argparse
import asyncio

from cryptography.exceptions import InvalidTag
from memmachine_server.common.kms.keyring_kms import (
    DEFAULT_KEYRING_KMS_SERVICE_NAME,
    KeyringKMSCryptoClient,
    KeyringKMSCryptoClientParams,
    KeyringKMSKeyAdminClient,
    KeyringKMSKeyAdminClientParams,
)

DEFAULT_VERIFY_SERVICE_NAME = f"{DEFAULT_KEYRING_KMS_SERVICE_NAME}-verify"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Provision a temporary key via KeyringKMSKeyAdminClient and verify "
            "KeyringKMSCryptoClient encrypt/decrypt behavior against the real "
            "OS keyring."
        ),
    )
    parser.add_argument(
        "--service-name",
        default=DEFAULT_VERIFY_SERVICE_NAME,
        help=(
            "Keyring service namespace to use. Defaults to a verification-only "
            "namespace, not the production client default."
        ),
    )
    parser.add_argument(
        "--keep-key",
        action="store_true",
        help="Leave the temporary keyring entry in place for manual inspection.",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause before cleanup so you can inspect the OS keyring entry.",
    )
    return parser.parse_args()


async def verify_keyring_clients(
    *,
    service_name: str,
    keep_key: bool,
    pause: bool,
) -> None:
    admin = KeyringKMSKeyAdminClient(
        KeyringKMSKeyAdminClientParams(service_name=service_name),
    )
    key_ref = await admin.create_key()

    try:
        if not await admin.exists_key(key_ref):
            raise RuntimeError(
                f"Newly created key {key_ref!r} did not report as existing"
            )

        client = KeyringKMSCryptoClient(
            KeyringKMSCryptoClientParams(service_name=service_name),
        )
        plaintext = b"hello keyring kms crypto client"
        associated_data = b"verify:keyring:kms"

        ciphertext = await client.encrypt(
            key_ref,
            plaintext,
            associated_data=associated_data,
        )
        decrypted = await client.decrypt(
            key_ref,
            ciphertext,
            associated_data=associated_data,
        )
        if decrypted != plaintext:
            raise RuntimeError("Round-trip decrypted bytes did not match plaintext")

        try:
            await client.decrypt(
                key_ref,
                ciphertext,
                associated_data=b"verify:keyring:wrong",
            )
        except InvalidTag:
            pass
        else:
            raise RuntimeError("Decrypt unexpectedly succeeded with wrong AAD")

        print("Keyring KMS clients verification succeeded.")
        print(f"Service: {service_name}")
        print(f"Key ref: {key_ref}")
        if pause:
            await asyncio.to_thread(
                input,
                "Inspect the OS keyring entry now, then press Enter to continue...",
            )
    finally:
        if keep_key:
            print("Leaving keyring entry in place because --keep-key was set.")
        else:
            await admin.delete_key(key_ref)


def main() -> None:
    args = parse_args()
    asyncio.run(
        verify_keyring_clients(
            service_name=args.service_name,
            keep_key=args.keep_key,
            pause=args.pause,
        )
    )


if __name__ == "__main__":
    main()
