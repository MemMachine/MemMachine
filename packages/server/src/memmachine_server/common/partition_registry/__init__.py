"""Partition registry interface and implementations."""

from .data_types import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionDeletedError,
    PartitionPendingError,
    validate_partition_name,
)
from .partition_manager import PartitionManager
from .partition_registry import PartitionRegistry, Registration
from .partitioned_store import PartitionedStore

__all__ = [
    "PartitionAlreadyExistsError",
    "PartitionCreateAttemptsExhaustedError",
    "PartitionDeletedError",
    "PartitionManager",
    "PartitionPendingError",
    "PartitionRegistry",
    "PartitionedStore",
    "Registration",
    "validate_partition_name",
]
