"""Default payload codec factory and matching key material generator."""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .aes_gcm_payload_codec import AESGCMPayloadCodec
from .payload_codec import PayloadCodec

_AES_GCM_KEY_BITS = 256


def default_payload_codec_factory(
    key: bytes,
    associated_data: bytes | None = None,
) -> PayloadCodec:
    """Build the default payload codec from raw key material."""
    return AESGCMPayloadCodec(key, associated_data=associated_data)


def default_key_material_generator() -> bytes:
    """Generate key material."""
    return AESGCM.generate_key(bit_length=_AES_GCM_KEY_BITS)
