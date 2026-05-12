"""Payload codec loader for KMS envelope-encrypted codec configs."""

from memmachine_server.common.kms.kms_crypto_client import KMSCryptoClient

from .aes_gcm_payload_codec import AESGCMPayloadCodec
from .payload_codec import PayloadCodec
from .payload_codec_config import (
    AESGCMPayloadCodecConfig,
    KMSEnvelopePayloadCodecConfig,
)


class KMSEnvelopePayloadCodecLoader:
    """Materialize codecs from KMS envelope-encrypted codec configs."""

    def __init__(self, kms_crypto_client: KMSCryptoClient) -> None:
        """Initialize with a KMS crypto client."""
        self._kms_crypto_client = kms_crypto_client

    async def load(self, config: KMSEnvelopePayloadCodecConfig) -> PayloadCodec:
        """
        Materialize a live codec from a KMS envelope-encrypted codec config.

        Args:
            config (KMSEnvelopePayloadCodecConfig):
                Codec configuration carrying a KMS key reference and wrapped DEK.

        Returns:
            PayloadCodec:
                A live codec instance.
        """
        match config:
            case AESGCMPayloadCodecConfig():
                dek = await self._kms_crypto_client.decrypt(
                    config.key_ref,
                    config.wrapped_dek,
                    associated_data=config.associated_data,
                )
                return AESGCMPayloadCodec(
                    dek,
                    nonce_size=config.nonce_size,
                    associated_data=config.associated_data,
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported KMS envelope payload codec config: "
                    f"{type(config).__name__}"
                )
