"""Partition registry interface and implementations."""

from .data_types import (
    AttemptsExhaustedError,
    PartitionAlreadyExistsError,
    PartitionIncarnationEndedError,
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
    "PartitionIncarnationEndedError",
    "PartitionPendingError",
    "PartitionRegistry",
    "PurgeRound",
    "Registration",
    "Reservation",
    "validate_partition_name",
]
