"""Clustering logic for semantic memory ingestion."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass
class ClusterInfo:
    """Centroid stats for a single cluster."""

    centroid: Sequence[float]
    count: int
    last_ts: datetime


@dataclass
class ClusterSplitRecord:
    """Records a completed split so it is not re-run on re-ingestion."""

    original_cluster_id: str
    segment_ids: list[str]
    input_hash: str


@dataclass
class ClusterState:
    """Mutable clustering state for a set."""

    clusters: MutableMapping[str, ClusterInfo] = field(default_factory=dict)
    event_to_cluster: MutableMapping[str, str] = field(default_factory=dict)
    pending_events: dict[str, dict[str, datetime]] = field(
        default_factory=dict,
    )
    next_cluster_id: int = 0
    split_records: MutableMapping[str, ClusterSplitRecord] = field(
        default_factory=dict,
    )


@dataclass(frozen=True)
class ClusterAssignment:
    """Result of assigning an event to a cluster."""

    cluster_id: str
    similarity: float | None
    created_new: bool


@dataclass(frozen=True)
class ClusterParams:
    """Configuration for cluster assignment decisions."""

    similarity_threshold: float = 0.3
    max_time_gap: timedelta | None = None
    id_prefix: str = "cluster_"


@dataclass(frozen=True)
class ClusterSplitParams:
    """Tuning parameters for the cluster split phase."""

    enabled: bool = False
    min_cluster_size: int = 6
    max_messages_in_prompt: int = 20
    low_similarity_threshold: float = 0.5
    time_gap_seconds: float | None = None
    cohesion_drop_zscore: float = 2.0
    debug_fail_loudly: bool = False


class ClusterManager:
    """Assigns events to clusters and updates state."""

    def __init__(self, params: ClusterParams) -> None:
        """Validate and store cluster parameters."""
        if not 0.0 <= params.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if params.id_prefix == "":
            raise ValueError("id_prefix must be non-empty")
        self._params = params

    def assign(
        self,
        *,
        event_id: str,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState | None = None,
    ) -> tuple[ClusterAssignment, ClusterState]:
        if state is None:
            state = ClusterState()

        if event_id in state.event_to_cluster:
            cluster_id = state.event_to_cluster[event_id]
            return ClusterAssignment(
                cluster_id=cluster_id,
                similarity=None,
                created_new=False,
            ), state

        selected_id, similarity = self._select_cluster(
            embedding=embedding,
            timestamp=timestamp,
            state=state,
        )

        if selected_id is None or similarity < self._params.similarity_threshold:
            cluster_id = self._create_cluster(
                embedding=embedding,
                timestamp=timestamp,
                state=state,
            )
            state.event_to_cluster[event_id] = cluster_id
            return ClusterAssignment(
                cluster_id=cluster_id,
                similarity=None,
                created_new=True,
            ), state

        self._update_cluster(
            cluster_id=selected_id,
            embedding=embedding,
            timestamp=timestamp,
            state=state,
        )
        state.event_to_cluster[event_id] = selected_id
        return ClusterAssignment(
            cluster_id=selected_id,
            similarity=similarity,
            created_new=False,
        ), state

    def _select_cluster(
        self,
        *,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState,
    ) -> tuple[str | None, float]:
        best_id: str | None = None
        best_similarity = -1.0

        for cluster_id, info in state.clusters.items():
            if not self._cluster_is_eligible(info, timestamp):
                continue
            similarity = self._cosine_similarity(info.centroid, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = cluster_id

        return best_id, best_similarity

    def _cluster_is_eligible(self, info: ClusterInfo, timestamp: datetime) -> bool:
        if self._params.max_time_gap is None:
            return True
        gap = timestamp - info.last_ts
        if gap.total_seconds() < 0:
            gap = info.last_ts - timestamp
        return gap <= self._params.max_time_gap

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        a_vec = np.array(a, dtype=float)
        b_vec = np.array(b, dtype=float)
        if a_vec.shape != b_vec.shape:
            raise ValueError("Embedding dimension mismatch")
        denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / denom)

    def _create_cluster(
        self,
        *,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState,
    ) -> str:
        cluster_id = f"{self._params.id_prefix}{state.next_cluster_id}"
        state.next_cluster_id += 1
        state.clusters[cluster_id] = ClusterInfo(
            centroid=list(map(float, embedding)),
            count=1,
            last_ts=timestamp,
        )
        return cluster_id

    def _update_cluster(
        self,
        *,
        cluster_id: str,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState,
    ) -> None:
        info = state.clusters[cluster_id]
        centroid = np.array(info.centroid, dtype=float)
        new_vec = np.array(embedding, dtype=float)
        if centroid.shape != new_vec.shape:
            raise ValueError("Embedding dimension mismatch")
        new_count = info.count + 1
        updated = (centroid * info.count + new_vec) / new_count
        info.centroid = [float(x) for x in updated.tolist()]
        info.count = new_count
        info.last_ts = timestamp
