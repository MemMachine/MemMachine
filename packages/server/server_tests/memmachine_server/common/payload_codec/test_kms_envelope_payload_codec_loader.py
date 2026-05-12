"""Tests for the KMS envelope payload codec loader."""

from typing import override

import pytest
from cryptography.exceptions import InvalidTag

from memmachine_server.common.kms.kms_crypto_client import KMSCryptoClient
from memmachine_server.common.payload_codec import KMSEnvelopePayloadCodecLoader
from memmachine_server.common.payload_codec.aes_gcm_payload_codec import (
    AESGCMPayloadCodec,
)
from memmachine_server.common.payload_codec.payload_codec_config import (
    AESGCMPayloadCodecConfig,
    KMSEnvelopePayloadCodecConfig,
)


class FakeKMSCryptoClient(KMSCryptoClient):
    """Minimal fake KMS client for loader tests."""

    def __init__(self, plaintext_key: bytes) -> None:
        self.plaintext_key = plaintext_key
        self.decrypt_calls: list[tuple[str, bytes, bytes | None]] = []

    @override
    async def encrypt(
        self,
        key_ref: str,
        plaintext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        raise NotImplementedError

    @override
    async def decrypt(
        self,
        key_ref: str,
        ciphertext: bytes,
        *,
        associated_data: bytes | None = None,
    ) -> bytes:
        self.decrypt_calls.append((key_ref, ciphertext, associated_data))
        return self.plaintext_key


@pytest.mark.asyncio
async def test_loader_returns_aes_gcm_codec() -> None:
    kms_client = FakeKMSCryptoClient(b"0" * 32)
    loader = KMSEnvelopePayloadCodecLoader(kms_client)
    config = AESGCMPayloadCodecConfig(
        key_ref="partition_key",
        wrapped_dek=b"wrapped-dek-bytes",
        nonce_size=12,
        associated_data=b"partition:context",
    )

    codec = await loader.load(config)

    assert isinstance(codec, AESGCMPayloadCodec)
    assert kms_client.decrypt_calls == [
        ("partition_key", b"wrapped-dek-bytes", b"partition:context")
    ]

    encoded = codec.encode(b"payload")
    decoded = codec.decode(encoded)

    assert decoded == b"payload"

    wrong_codec = AESGCMPayloadCodec(
        b"0" * 32,
        nonce_size=12,
        associated_data=b"partition:block",
    )
    with pytest.raises(InvalidTag):
        wrong_codec.decode(encoded)


class UnknownKMSEnvelopePayloadCodecConfig(KMSEnvelopePayloadCodecConfig):
    """Envelope config whose concrete type the loader does not handle."""


@pytest.mark.asyncio
async def test_loader_rejects_unsupported_codec_config() -> None:
    loader = KMSEnvelopePayloadCodecLoader(FakeKMSCryptoClient(b"0" * 32))

    with pytest.raises(
        NotImplementedError,
        match="Unsupported KMS envelope payload codec config",
    ):
        await loader.load(
            UnknownKMSEnvelopePayloadCodecConfig(
                key_ref="partition_key",
                wrapped_dek=b"wrapped-dek-bytes",
            ),
        )
