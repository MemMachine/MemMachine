"""Collection registry interface and implementations."""

from .collection_registry import (
    PurgeClaim,
    RegisteredCollection,
    VectorStoreCollectionRegistry,
)

__all__ = [
    "PurgeClaim",
    "RegisteredCollection",
    "VectorStoreCollectionRegistry",
]
