"""Tests for the KMS envelope payload codec loader."""

from typing import Literal, cast, override

import pytest
from cryptography.exceptions import InvalidTag
from pydantic import BaseModel, ConfigDict

from memmachine_server.common.kms.kms_crypto_client import KMSCryptoClient
from memmachine_server.common.payload_codec import KMSEnvelopePayloadCodecLoader
from memmachine_server.common.payload_codec.aes_gcm_payload_codec import (
    AESGCMPayloadCodec,
)
from memmachine_server.common.payload_codec.payload_codec_config import (
    AESGCMPayloadCodecConfig,
    KMSEnvelopeParams,
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
        envelope=KMSEnvelopeParams(
            key_ref="partition_key",
            wrapped_dek=b"wrapped-dek-bytes",
            associated_data=b"partition:context",
        ),
        nonce_size=12,
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


@pytest.mark.asyncio
async def test_loader_returns_aes_gcm_codec_without_associated_data() -> None:
    kms_client = FakeKMSCryptoClient(b"0" * 32)
    loader = KMSEnvelopePayloadCodecLoader(kms_client)
    config = AESGCMPayloadCodecConfig(
        envelope=KMSEnvelopeParams(
            key_ref="partition_key",
            wrapped_dek=b"wrapped-dek-bytes",
            associated_data=None,
        ),
        nonce_size=12,
    )

    codec = await loader.load(config)

    assert isinstance(codec, AESGCMPayloadCodec)
    assert kms_client.decrypt_calls == [("partition_key", b"wrapped-dek-bytes", None)]

    encoded = codec.encode(b"payload")
    decoded = codec.decode(encoded)

    assert decoded == b"payload"

    wrong_codec = AESGCMPayloadCodec(
        b"0" * 32,
        nonce_size=12,
        associated_data=b"partition:context",
    )
    with pytest.raises(InvalidTag):
        wrong_codec.decode(encoded)


class UnknownEnvelopePayloadCodecConfig(BaseModel):
    """Envelope codec config whose concrete type the loader does not handle."""

    model_config = ConfigDict(frozen=True)

    type: Literal["unknown"] = "unknown"
    envelope: KMSEnvelopeParams


@pytest.mark.asyncio
async def test_loader_rejects_unsupported_codec_config() -> None:
    kms_client = FakeKMSCryptoClient(b"0" * 32)
    loader = KMSEnvelopePayloadCodecLoader(kms_client)
    config = UnknownEnvelopePayloadCodecConfig(
        envelope=KMSEnvelopeParams(
            key_ref="partition_key",
            wrapped_dek=b"wrapped-dek-bytes",
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="Unsupported KMS envelope payload codec config",
    ):
        await loader.load(cast(KMSEnvelopePayloadCodecConfig, config))

    # Unsupported configs must not trigger a KMS decrypt.
    assert kms_client.decrypt_calls == []
