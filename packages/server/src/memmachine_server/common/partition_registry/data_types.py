"""
Data types for partition registries.

Imports nothing from this package.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

# The longest partition namespace and partition name a registry holds, in
# bytes.
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


@dataclass(frozen=True)
class ListedPartition:
    """
    A registered partition, as one listing found it.

    The partition's state is as the listing's read found it, and may have
    changed since: a pending partition is confirmed or deleted, and a live
    one deleted, under any holder of this object.

    Attributes:
        partition_name (str):
            The name the partition is registered under.
        partition_uuid (UUID):
            The partition's UUID.
        state (Literal["pending", "live"]):
            "pending" from the reservation of the partition's name until the
            reservation is confirmed, and "live" from then on.
        reserved_at (datetime):
            When the partition's name was reserved, on the database clock.
    """

    partition_name: str
    partition_uuid: UUID
    state: Literal["pending", "live"]
    reserved_at: datetime


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
    """Raised when getting a partition whose creation has not completed."""

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
    """Raised when a partition was deleted, before or after its creation completed."""

    def __init__(
        self, partition_namespace: str, partition_name: str, partition_uuid: UUID
    ) -> None:
        """Initialize with the deleted partition's namespace, name, and UUID."""
        self.partition_namespace = partition_namespace
        self.partition_name = partition_name
        self.partition_uuid = partition_uuid
        super().__init__(
            f"Partition {partition_uuid} under name {partition_name!r} in "
            f"namespace {partition_namespace!r} was deleted"
        )


class PartitionAlreadyConfirmedError(Exception):
    """Raised when a reservation's partition was confirmed already."""

    def __init__(
        self, partition_namespace: str, partition_name: str, partition_uuid: UUID
    ) -> None:
        """Initialize with the confirmed partition's namespace, name, and UUID."""
        self.partition_namespace = partition_namespace
        self.partition_name = partition_name
        self.partition_uuid = partition_uuid
        super().__init__(
            f"Partition {partition_uuid} under name {partition_name!r} in "
            f"namespace {partition_namespace!r} was confirmed already"
        )


class PartitionCreateAttemptsExhaustedError(Exception):
    """Raised when creating a partition gave up after repeated attempts that made no progress."""


class TombstoneClaimStaleError(Exception):
    """Raised when a tombstone claim is ended after it stopped being the tombstone's latest, unended claim."""

    def __init__(self, partition_uuid: UUID) -> None:
        """Initialize with the UUID of the partition whose tombstone was claimed."""
        self.partition_uuid = partition_uuid
        super().__init__(
            f"Tombstone claim on partition {partition_uuid} is stale, since it "
            "ended already or the tombstone was claimed again after its lease "
            "and backoff"
        )
