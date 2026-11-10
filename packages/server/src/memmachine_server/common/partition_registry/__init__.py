"""Partition registry interface and implementations."""

from .data_types import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionIncarnationEndedError,
    PartitionPendingError,
    validate_partition_name,
)
from .partition_owner import open_or_create_partition
from .partition_registry import (
    PartitionRegistry,
    PurgeRound,
    Registration,
    Reservation,
)

__all__ = [
    "PartitionAlreadyExistsError",
    "PartitionCreateAttemptsExhaustedError",
    "PartitionIncarnationEndedError",
    "PartitionPendingError",
    "PartitionRegistry",
    "PurgeRound",
    "Registration",
    "Reservation",
    "open_or_create_partition",
    "validate_partition_name",
]
