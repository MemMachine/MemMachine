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
        snapshots: Iterable[Snapshot],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def add_snapshot_derivative_uuids(
        self,
        session_key: str,
        snapshot_derivative_uuids: Mapping[UUID, Iterable[UUID]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_snapshot_derivative_uuids(
        self,
        session_key: str,
        snapshot_uuids: Iterable[UUID],
    ) -> Iterable[Iterable[UUID]]:
        raise NotImplementedError

    @abstractmethod
    async def get_episodes_snapshots(
        self,
        session_key: str,
        episode_uuids: Iterable[UUID],
    ) -> Iterable[Snapshot]:
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
        raise NotImplementedError

    @abstractmethod
    async def delete_snapshots(
        self,
        session_key: str,
        snapshot_uuids: Iterable[UUID],
    ) -> None:
        raise NotImplementedError
