"""Segment store data models and interfaces."""

from .data_types import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStoreEventAlreadyStoredError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
)
from .segment_store import (
    SegmentStore,
    SegmentStorePartition,
    SegmentStorePartitionWriter,
)

__all__ = [
    "SegmentStore",
    "SegmentStoreAttemptsExhaustedError",
    "SegmentStoreEventAlreadyStoredError",
    "SegmentStorePartition",
    "SegmentStorePartitionAlreadyExistsError",
    "SegmentStorePartitionConfig",
    "SegmentStorePartitionConfigMismatchError",
    "SegmentStorePartitionHandleStaleError",
    "SegmentStorePartitionWriter",
]
