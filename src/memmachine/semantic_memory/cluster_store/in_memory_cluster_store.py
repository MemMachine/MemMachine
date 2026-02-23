"""In-memory clustering state storage for tests and local use."""

from __future__ import annotations

import copy
from collections.abc import MutableMapping

from memmachine.semantic_memory.cluster_manager import ClusterState
from memmachine.semantic_memory.cluster_store.cluster_store import ClusterStateStorage
from memmachine.semantic_memory.semantic_model import SetIdT


class InMemoryClusterStateStorage(ClusterStateStorage):
    """Simple in-memory implementation of ClusterStateStorage."""

    def __init__(self) -> None:
        """Initialize empty in-memory state map."""
        self._states: MutableMapping[str, ClusterState] = {}

    async def startup(self) -> None:
        return None

    async def delete_all(self) -> None:
        self._states.clear()

    async def get_state(self, *, set_id: SetIdT) -> ClusterState | None:
        state = self._states.get(str(set_id))
        if state is None:
            return None
        return copy.deepcopy(state)

    async def save_state(self, *, set_id: SetIdT, state: ClusterState) -> None:
        self._states[str(set_id)] = copy.deepcopy(state)

    async def delete_state(self, *, set_id: SetIdT) -> None:
        self._states.pop(str(set_id), None)
