"""Segment store data models and interfaces."""

from .data_types import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStoreEventAlreadyExistsError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
    Tombstone,
)
from .segment_store import (
    PendingWriteClaim,
    SegmentStore,
    SegmentStorePartition,
    TombstoneClaim,
)

__all__ = [
    "PendingWriteClaim",
    "SegmentStore",
    "SegmentStoreAttemptsExhaustedError",
    "SegmentStoreEventAlreadyExistsError",
    "SegmentStorePartition",
    "SegmentStorePartitionAlreadyExistsError",
    "SegmentStorePartitionConfig",
    "SegmentStorePartitionConfigMismatchError",
    "SegmentStorePartitionHandleStaleError",
    "Tombstone",
    "TombstoneClaim",
]
