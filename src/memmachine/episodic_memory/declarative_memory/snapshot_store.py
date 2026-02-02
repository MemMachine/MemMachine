from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from uuid import UUID

from memmachine.common.filter.filter_parser import FilterExpr

from .data_types import Snapshot


class SnapshotStore(ABC):
    @abstractmethod
    async def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def add_snapshots(
        self,
        session_key: str,
        snapshots: Mapping[Snapshot, Iterable[UUID]],
    ) -> None:
        """
        Add a mapping of snapshots to the UUIDs of all embedding records.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_snapshot_contexts(
        self,
        session_key: str,
        seed_snapshot_uuids: Iterable[UUID],
        *,
        max_backward_snapshots: int = 0,
        max_forward_snapshots: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> Iterable[Iterable[Snapshot]]:
        """
        Returns a window of snapshots around each of the seed snapshots.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_episodes_snapshots(
        self,
        session_key: str,
        episode_uuids: Iterable[UUID],
    ) -> Iterable[UUID]:
        """
        Delete all snapshots associated with the episodes given by their UUIDs.
        Returns the UUIDs of the embedding records associated with the deleted snapshots.
        """
        raise NotImplementedError
