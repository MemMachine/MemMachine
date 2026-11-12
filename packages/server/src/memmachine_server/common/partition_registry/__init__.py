"""Partition registry interface and implementations."""

from .data_types import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionDeletedError,
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
    "PartitionAlreadyExistsError",
    "PartitionCreateAttemptsExhaustedError",
    "PartitionDeletedError",
    "PartitionPendingError",
    "PartitionRegistry",
    "PurgeRound",
    "Registration",
    "Reservation",
    "validate_partition_name",
]
