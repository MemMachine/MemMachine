"""
The MemMachine property key namespace.

Keys beginning with `RESERVED_PROPERTY_KEY_PREFIX` belong to MemMachine and are
built by `reserved_property_key` as `<prefix><system>_<field>`; every other key
belongs to the caller. The prefix is the distribution name, so its uniqueness
is the package registry's. Every key is bounded by `validate_identifier`, the
vector store's naming contract.
"""

from typing import Final

from memmachine_server.common.vector_store.utils import validate_identifier

RESERVED_PROPERTY_KEY_PREFIX: Final[str] = "memmachine_"


def is_reserved_property_key(key: str) -> bool:
    """Return whether `key` belongs to MemMachine rather than to the caller."""
    return key.startswith(RESERVED_PROPERTY_KEY_PREFIX)


def reserved_property_key(system: str, field: str) -> str:
    """
    Build the reserved property key naming `field` of `system`.

    Raises ValueError if the key is not a valid identifier.
    """
    key = f"{RESERVED_PROPERTY_KEY_PREFIX}{system}_{field}"
    if not validate_identifier(key):
        raise ValueError(
            f"Reserved property key {key!r} ({len(key.encode())} bytes) does not "
            f"satisfy the vector store naming contract: [a-z0-9_], at most 32 bytes."
        )
    return key


def validate_caller_property_key(key: str) -> None:
    """
    Raise ValueError unless `key` is a legal caller property key.

    A caller key is a valid identifier outside the reserved namespace. It is
    used exactly as given, never rewritten, so an illegal key is rejected
    rather than repaired.
    """
    if is_reserved_property_key(key):
        raise ValueError(
            f"Property key {key!r} is reserved: keys beginning with "
            f"{RESERVED_PROPERTY_KEY_PREFIX!r} belong to MemMachine."
        )
    if not validate_identifier(key):
        raise ValueError(
            f"Property key {key!r} must match [a-z0-9_] and be at most 32 bytes."
        )
