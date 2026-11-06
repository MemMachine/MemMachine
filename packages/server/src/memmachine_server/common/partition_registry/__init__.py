"""Partition registry interface and implementations."""

from .data_types import (
    AttemptsExhaustedError,
    PartitionAlreadyExistsError,
    PartitionDeletedError,
    PartitionHandleStaleError,
    PartitionPendingError,
)
from .partition_registry import (
    PurgeRound,
    Registration,
    Reservation,
    VectorStorePartitionRegistry,
)

__all__ = [
    "AttemptsExhaustedError",
    "PartitionAlreadyExistsError",
    "PartitionDeletedError",
    "PartitionHandleStaleError",
    "PartitionPendingError",
    "PurgeRound",
    "Registration",
    "Reservation",
    "VectorStorePartitionRegistry",
]
