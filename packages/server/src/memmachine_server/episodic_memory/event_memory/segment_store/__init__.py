"""Segment store data models and interfaces."""

from .data_types import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
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
    "SegmentStorePartitionHandleStaleError",
]
