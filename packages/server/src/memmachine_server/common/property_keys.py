"""
Ownership of the MemMachine property key namespace.

Property keys are shared between callers and MemMachine itself: a caller names
its own metadata, and each memory system needs somewhere to put the fields it
maintains. Keys beginning with `RESERVED_PROPERTY_KEY_PREFIX` belong to
MemMachine; everything else belongs to the caller.

The prefix is the distribution name, so its uniqueness is enforced by the
package registry rather than assumed, and memory systems subdivide within it --
`memmachine_event_timestamp`, `memmachine_<system>_<field>` -- rather than each
reserving a further prefix from the caller.

Build reserved keys with `reserved_property_key` rather than by concatenation:
the alphabet and length available to a key are set by the vector store's naming
contract, and the budget is easy to overrun once a prefix and a system name are
spent.
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

    Raises ValueError if the result does not satisfy the vector store's naming
    contract, which is how a prefix plus a system name overrunning the length
    budget is caught at import time rather than at first write.
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
    Raise ValueError unless `key` is a legal caller-supplied property key.

    A caller key must satisfy the vector store's naming contract -- `[a-z0-9_]`,
    at most 32 bytes -- and must not fall in MemMachine's reserved namespace.
    Keys are stored and filtered exactly as given: MemMachine never rewrites,
    prefixes, or encodes them, so the key a caller writes is the key it filters
    on, and an illegal key is rejected rather than repaired.

    Properties are the only filterable surface, and they are stored in the
    clear. An event's context and block go through the payload codec and are
    never filterable, so choosing which data is safe to expose as a property is
    the application's decision: to filter on provenance, project it into a
    property rather than expecting `context` to be reachable.
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
