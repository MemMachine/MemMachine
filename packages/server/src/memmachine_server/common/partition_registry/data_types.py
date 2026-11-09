"""
Data types for partition registries.

Imports nothing from this package; the rest of it builds on these types.
"""

from datetime import datetime

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


class PartitionDeletedError(Exception):
    """Raised when a partition is deleted before its creation completes."""

    def __init__(self, partition_namespace: str, partition_name: str) -> None:
        """Initialize with the namespace and the name of the deleted partition."""
        self.partition_namespace = partition_namespace
        self.partition_name = partition_name
        super().__init__(
            f"Partition {partition_name!r} in namespace {partition_namespace!r} was "
            "deleted before its creation completed"
        )


class PartitionHandleStaleError(Exception):
    """Raised when a handle is used after its partition was deleted."""

    def __init__(self, partition_namespace: str, partition_name: str) -> None:
        """Record the namespace and the name the stale handle belonged to."""
        self.partition_namespace = partition_namespace
        self.partition_name = partition_name
        super().__init__(
            f"Stale handle for partition {partition_name!r} in namespace "
            f"{partition_namespace!r}: the partition was deleted (or re-created) after this "
            "handle was bound"
        )


class AttemptsExhaustedError(Exception):
    """Raised when an operation gave up after repeated attempts that made no progress."""
