"""Partition registry interface and implementations."""

from .data_types import (
    AttemptsExhaustedError,
    PartitionAlreadyExistsError,
    PartitionDeletedError,
    PartitionHandleStaleError,
    PartitionPendingError,
)
from .partition_registry import (
    PartitionRegistry,
    PurgeRound,
    Registration,
    Reservation,
)

__all__ = [
    "AttemptsExhaustedError",
    "PartitionAlreadyExistsError",
    "PartitionDeletedError",
    "PartitionHandleStaleError",
    "PartitionPendingError",
    "PartitionRegistry",
    "PurgeRound",
    "Registration",
    "Reservation",
]
