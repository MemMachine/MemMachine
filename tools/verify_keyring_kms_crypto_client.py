"""Verify the keyring-backed KMS crypto client against the real OS keyring."""

import argparse
import asyncio
from base64 import urlsafe_b64encode
from secrets import token_bytes
from uuid import uuid4

import keyring
from cryptography.exceptions import InvalidTag
from memmachine_server.common.kms.keyring_kms_crypto_client import (
    DEFAULT_KEYRING_SERVICE_NAME,
    KeyringKMSCryptoClient,
    KeyringKMSCryptoClientParams,
)

DEFAULT_VERIFY_SERVICE_NAME = f"{DEFAULT_KEYRING_SERVICE_NAME}-verify"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed a temporary key in the real OS keyring and verify "
            "KeyringKMSCryptoClient encrypt/decrypt behavior."
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
        "--key-ref",
        default=f"verify-{uuid4()}",
        help="Key reference to create temporarily. Defaults to a random value.",
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


async def verify_keyring_client(
    *,
    service_name: str,
    key_ref: str,
    keep_key: bool,
    pause: bool,
) -> None:
    existing_key = keyring.get_password(service_name, key_ref)
    if existing_key is not None:
        raise RuntimeError(
            f"Refusing to overwrite existing keyring entry "
            f"({service_name!r}, {key_ref!r})"
        )

    key_material = token_bytes(32)
    encoded_key = urlsafe_b64encode(key_material).decode("ascii")
    keyring.set_password(service_name, key_ref, encoded_key)

    try:
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

        print("Keyring KMS crypto client verification succeeded.")
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
            keyring.delete_password(service_name, key_ref)


def main() -> None:
    args = parse_args()
    asyncio.run(
        verify_keyring_client(
            service_name=args.service_name,
            key_ref=args.key_ref,
            keep_key=args.keep_key,
            pause=args.pause,
        )
    )


if __name__ == "__main__":
    main()
