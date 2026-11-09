"""
Data types for partition registries.

Imports nothing from this package; the rest of it builds on these types.
"""

from datetime import datetime
from uuid import UUID

# The longest partition namespace and partition name, in bytes, a registry
# holds.
_PARTITION_NAMESPACE_MAX_BYTES = 32
_PARTITION_NAME_MAX_BYTES = 255


def validate_partition_namespace(partition_namespace: str) -> None:
    """Raise ValueError unless the namespace is non-empty, at most `_PARTITION_NAMESPACE_MAX_BYTES` bytes, and free of NUL."""
    if (
        not partition_namespace
        or "\x00" in partition_namespace
        or len(partition_namespace.encode()) > _PARTITION_NAMESPACE_MAX_BYTES
    ):
        raise ValueError(
            f"Partition namespace {partition_namespace!r} must be non-empty, at "
            f"most {_PARTITION_NAMESPACE_MAX_BYTES} bytes, and free of NUL "
            "characters"
        )


def validate_partition_name(partition_name: str) -> None:
    """Raise ValueError unless the partition name is non-empty, at most `_PARTITION_NAME_MAX_BYTES` bytes, and free of NUL."""
    if (
        not partition_name
        or "\x00" in partition_name
        or len(partition_name.encode()) > _PARTITION_NAME_MAX_BYTES
    ):
        raise ValueError(
            f"Partition name {partition_name!r} must be non-empty, at most "
            f"{_PARTITION_NAME_MAX_BYTES} bytes, and free of NUL characters"
        )


class PartitionAlreadyExistsError(Exception):
    """Raised when creating a partition that already exists."""

    def __init__(self, partition_namespace: str, partition_name: str) -> None:
        """Initialize with the namespace and the name of the existing partition."""
        self.partition_namespace = partition_namespace
        self.partition_name = partition_name
        super().__init__(
            f"Partition {partition_name!r} in namespace {partition_namespace!r} already exists."
        )


class PartitionPendingError(Exception):
    """Raised when opening a partition whose creation has not completed."""

    def __init__(
        self,
        partition_namespace: str,
        partition_name: str,
        registered_at: datetime,
    ) -> None:
        """Initialize with the pending partition's namespace, name, and registration time."""
        self.partition_namespace = partition_namespace
        self.partition_name = partition_name
        self.registered_at = registered_at
        super().__init__(
            f"Partition {partition_name!r} in namespace {partition_namespace!r} has "
            f"been pending since {registered_at.isoformat()}; if its creation was "
            "abandoned, delete it to create it again."
        )


class PartitionIncarnationEndedError(Exception):
    """Raised when an incarnation of a partition is no longer the life registered under its name: the partition was deleted, before or after its creation completed."""

    def __init__(
        self, partition_namespace: str, partition_name: str, incarnation: UUID
    ) -> None:
        """Initialize with the partition's namespace and name, and the incarnation that ended."""
        self.partition_namespace = partition_namespace
        self.partition_name = partition_name
        self.incarnation = incarnation
        super().__init__(
            f"Incarnation {incarnation} of partition {partition_name!r} in "
            f"namespace {partition_namespace!r} has ended: the partition was "
            "deleted, or deleted and created again, after the incarnation was "
            "reserved"
        )


class PartitionCreateAttemptsExhaustedError(Exception):
    """Raised when creating a partition, or opening it by creating it, gave up after repeated attempts that made no progress."""
