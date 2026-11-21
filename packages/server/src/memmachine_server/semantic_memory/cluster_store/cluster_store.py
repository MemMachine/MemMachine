"""Abstract interface for persisting clustering state."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from memmachine_server.common.lifecycle import Lifecycle
from memmachine_server.semantic_memory.cluster_manager import ClusterState
from memmachine_server.semantic_memory.semantic_model import SetIdT


@runtime_checkable
class ClusterStateStorage(Lifecycle, Protocol):
    """Contract for persisting clustering state per semantic set."""

    async def prepare(self) -> None:
        """
        Create the storage's missing durable resources.

        The caller runs it before `start`, while no other process uses the
        storage. Running it again changes nothing.
        """
        ...

    async def delete_all(self) -> None: ...

    async def get_state(self, *, set_id: SetIdT) -> ClusterState | None: ...

    async def save_state(self, *, set_id: SetIdT, state: ClusterState) -> None: ...

    async def delete_state(self, *, set_id: SetIdT) -> None: ...
