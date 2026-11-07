"""Partition registry interface and implementations."""

from .data_types import (
    AttemptsExhaustedError,
    PartitionAlreadyExistsError,
    PartitionDeletedError,
    PartitionHandleStaleError,
    PartitionPendingError,
    validate_partition_name,
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
    "validate_partition_name",
]
