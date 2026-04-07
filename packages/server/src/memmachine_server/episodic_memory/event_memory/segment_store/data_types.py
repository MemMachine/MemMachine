"""Data types for segment store."""


class SegmentStorePartitionAlreadyExistsError(Exception):
    """Raised when creating a partition that already exists."""

    def __init__(self, partition_key: str) -> None:
        """Initialize with the key of the existing partition."""
        self.partition_key = partition_key
        super().__init__(f"Partition {partition_key!r} already exists.")
