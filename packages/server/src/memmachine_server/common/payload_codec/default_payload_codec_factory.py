"""Default payload codec factory."""

from .aes_gcm_payload_codec import AESGCMPayloadCodec
from .payload_codec import PayloadCodec


def default_payload_codec_factory(
    key: bytes,
    associated_data: bytes | None = None,
) -> PayloadCodec:
    """Build the default payload codec from raw key material."""
    return AESGCMPayloadCodec(key, associated_data=associated_data)
