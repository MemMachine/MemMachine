"""Segment store data models and interfaces."""

from .data_types import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStoreEventAlreadyExistsError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
)
from .segment_store import (
    IndexingJobClaim,
    SegmentStore,
    SegmentStorePartition,
    TombstoneClaim,
)

__all__ = [
    "IndexingJobClaim",
    "SegmentStore",
    "SegmentStoreAttemptsExhaustedError",
    "SegmentStoreEventAlreadyExistsError",
    "SegmentStorePartition",
    "SegmentStorePartitionAlreadyExistsError",
    "SegmentStorePartitionConfig",
    "SegmentStorePartitionConfigMismatchError",
    "SegmentStorePartitionHandleStaleError",
    "TombstoneClaim",
]
