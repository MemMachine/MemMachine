"""Segment store data models and interfaces."""

from .data_types import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
)
from .segment_store import (
    SegmentStore,
    SegmentStorePartition,
)

__all__ = [
    "SegmentStore",
    "SegmentStoreAttemptsExhaustedError",
    "SegmentStorePartition",
    "SegmentStorePartitionAlreadyExistsError",
    "SegmentStorePartitionConfig",
    "SegmentStorePartitionConfigMismatchError",
    "SegmentStorePartitionHandleStaleError",
]
