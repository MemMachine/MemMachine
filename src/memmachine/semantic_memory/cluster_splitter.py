"""Reranker-guided cluster splitting for semantic ingestion."""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from memmachine.common.episode_store import Episode
from memmachine.common.reranker import Reranker
from memmachine.semantic_memory.cluster_manager import (
    ClusterInfo,
    ClusterSplitParams,
    ClusterSplitRecord,
    ClusterState,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContinuitySignals:
    """Pre-computed similarity and time-gap metrics for a cluster."""

    adjacent_similarities: Sequence[float]
    time_gaps_seconds: Sequence[float]
    min_adjacent_similarity: float
    max_time_gap_seconds: float


@runtime_checkable
class ClusterSplitterProtocol(Protocol):
    """Contract for the split phase in the ingestion pipeline."""

    async def maybe_split_clusters(
        self,
        *,
        cluster_messages: Sequence[tuple[str, Sequence[Episode]]],
        cluster_embeddings: Mapping[str, Sequence[float]],
        state: ClusterState,
        reranker: Reranker | None,
    ) -> tuple[list[tuple[str, Sequence[Episode]]], ClusterState]: ...


class NoOpClusterSplitter:
    """Satisfies ClusterSplitterProtocol but performs no splitting."""

    async def maybe_split_clusters(
        self,
        *,
        cluster_messages: Sequence[tuple[str, Sequence[Episode]]],
        cluster_embeddings: Mapping[str, Sequence[float]],  # noqa: ARG002
        state: ClusterState,
        reranker: Reranker | None,  # noqa: ARG002
    ) -> tuple[list[tuple[str, Sequence[Episode]]], ClusterState]:
        return list(cluster_messages), state


class SplitGate:
    """Evaluates whether a cluster is a candidate for split scoring."""

    def __init__(self, params: ClusterSplitParams) -> None:  # noqa: D107
        self._params = params

    def is_candidate(
        self,
        messages: Sequence[Episode],
        signals: ContinuitySignals,
    ) -> bool:
        """Return True only when min-size and at least one signal condition is met."""
        if len(messages) < self._params.min_cluster_size:
            return False

        if signals.min_adjacent_similarity < self._params.low_similarity_threshold:
            return True

        if (
            self._params.time_gap_seconds is not None
            and signals.max_time_gap_seconds > self._params.time_gap_seconds
        ):
            return True

        sims = np.array(signals.adjacent_similarities)
        if len(sims) >= 2:
            std = float(sims.std())
            if std > 0.0:
                mean = float(sims.mean())
                z_scores = (mean - sims) / std
                if float(z_scores.max()) >= self._params.cohesion_drop_zscore:
                    return True

        return False


class RerankerClusterSplitter:
    """Production splitter: gates on heuristics, scores with reranker, applies splits."""

    def __init__(self, params: ClusterSplitParams) -> None:  # noqa: D107
        self._params = params
        self._gate = SplitGate(params)

    @staticmethod
    def _collect_embeddings(
        messages: Sequence[Episode],
        cluster_embeddings: Mapping[str, Sequence[float]],
    ) -> tuple[list[Sequence[float]], list[str]]:
        ordered_embeddings = [
            cluster_embeddings[m.uid] for m in messages if m.uid is not None
        ]
        event_ids = [m.uid for m in messages if m.uid is not None]
        return ordered_embeddings, event_ids

    def _replay_if_unchanged(
        self,
        cluster_id: str,
        messages: Sequence[Episode],
        ordered_embeddings: Sequence[Sequence[float]],
        current_hash: str,
        state: ClusterState,
    ) -> list[tuple[str, Sequence[Episode]]] | None:
        record = state.split_records.get(cluster_id)
        if record is None:
            return None

        if record.input_hash != current_hash:
            state.split_records.pop(cluster_id, None)
            return None

        if record.segment_ids:
            return self._replay_split(
                cluster_id,
                messages,
                ordered_embeddings,
                record,
                state,
            )

        return [(cluster_id, messages)]

    @staticmethod
    def _record_split(
        state: ClusterState,
        cluster_id: str,
        segment_ids: Sequence[str],
        input_hash: str,
    ) -> None:
        state.split_records[cluster_id] = ClusterSplitRecord(
            original_cluster_id=cluster_id,
            segment_ids=list(segment_ids),
            input_hash=input_hash,
        )

    def _keep_cluster_intact(
        self,
        state: ClusterState,
        cluster_id: str,
        messages: Sequence[Episode],
        current_hash: str,
    ) -> list[tuple[str, Sequence[Episode]]]:
        self._record_split(state, cluster_id, [], current_hash)
        return [(cluster_id, messages)]

    async def _get_reranker_scores(
        self,
        window: Sequence[Episode],
        reranker: Reranker,
        cluster_id: str,
    ) -> list[float] | None:
        try:
            return await self._adjacent_reranker_scores(window, reranker)
        except Exception:
            logger.exception(
                "Reranker split failed for cluster %s, keeping intact",
                cluster_id,
            )
            if self._params.debug_fail_loudly:
                raise
            return None

    async def _maybe_split_cluster(
        self,
        *,
        cluster_id: str,
        messages: Sequence[Episode],
        cluster_embeddings: Mapping[str, Sequence[float]],
        state: ClusterState,
        reranker: Reranker | None,
    ) -> list[tuple[str, Sequence[Episode]]]:
        ordered_embeddings, event_ids = self._collect_embeddings(
            messages,
            cluster_embeddings,
        )
        current_hash = self._input_hash(event_ids)
        replayed = self._replay_if_unchanged(
            cluster_id,
            messages,
            ordered_embeddings,
            current_hash,
            state,
        )
        if replayed is not None:
            return replayed

        signals = self._compute_signals(messages, ordered_embeddings)

        if not self._gate.is_candidate(messages, signals):
            return self._keep_cluster_intact(
                state,
                cluster_id,
                messages,
                current_hash,
            )

        if reranker is None:
            logger.warning(
                "Cluster %s is a split candidate but no reranker is configured",
                cluster_id,
            )
            return self._keep_cluster_intact(
                state,
                cluster_id,
                messages,
                current_hash,
            )

        logger.info(
            "Cluster %s is a split candidate (size=%d), scoring with reranker",
            cluster_id,
            len(messages),
        )

        n = min(len(messages), self._params.max_messages_in_prompt)
        window = messages[:n]
        reranker_scores = await self._get_reranker_scores(
            window,
            reranker,
            cluster_id,
        )
        if not reranker_scores:
            return self._keep_cluster_intact(
                state,
                cluster_id,
                messages,
                current_hash,
            )

        time_gaps = signals.time_gaps_seconds[: max(len(window) - 1, 0)]
        split_indices = self._select_split_indices(
            reranker_scores,
            time_gaps,
        )
        valid_indices = validate_split_indices(split_indices, n)

        if not valid_indices:
            logger.info("Reranker decided no split for cluster %s", cluster_id)
            return self._keep_cluster_intact(
                state,
                cluster_id,
                messages,
                current_hash,
            )

        segments, segment_ids = apply_cluster_split(
            cluster_id,
            messages,
            ordered_embeddings,
            valid_indices,
            state,
        )
        self._record_split(state, cluster_id, segment_ids, current_hash)
        return segments

    async def maybe_split_clusters(
        self,
        *,
        cluster_messages: Sequence[tuple[str, Sequence[Episode]]],
        cluster_embeddings: Mapping[str, Sequence[float]],
        state: ClusterState,
        reranker: Reranker | None,
    ) -> tuple[list[tuple[str, Sequence[Episode]]], ClusterState]:
        result: list[tuple[str, Sequence[Episode]]] = []
        for cluster_id, messages in cluster_messages:
            segments = await self._maybe_split_cluster(
                cluster_id=cluster_id,
                messages=messages,
                cluster_embeddings=cluster_embeddings,
                state=state,
                reranker=reranker,
            )
            result.extend(segments)

        return result, state

    @staticmethod
    async def _adjacent_reranker_scores(
        messages: Sequence[Episode],
        reranker: Reranker,
    ) -> list[float]:
        scores: list[float] = []
        for i in range(len(messages) - 1):
            query = RerankerClusterSplitter._format_reranker_message(messages[i])
            candidate = RerankerClusterSplitter._format_reranker_message(
                messages[i + 1]
            )
            pair_scores = await reranker.score(query, [candidate])
            scores.append(float(pair_scores[0]) if pair_scores else 0.0)
        return scores

    def _select_split_indices(
        self,
        scores: Sequence[float],
        time_gaps_seconds: Sequence[float],
    ) -> list[int]:
        normalized = self._normalize_scores(scores)
        split_indices: set[int] = set()

        for i, score in enumerate(normalized):
            if score <= self._params.low_similarity_threshold:
                split_indices.add(i + 1)

        if self._params.time_gap_seconds is not None:
            for i, gap in enumerate(time_gaps_seconds):
                if gap > self._params.time_gap_seconds:
                    split_indices.add(i + 1)

        sims = np.array(normalized)
        if len(sims) >= 2:
            std = float(sims.std())
            if std > 0.0:
                mean = float(sims.mean())
                z_scores = (mean - sims) / std
                for i, z in enumerate(z_scores.tolist()):
                    if float(z) >= self._params.cohesion_drop_zscore:
                        split_indices.add(i + 1)

        return sorted(split_indices)

    @staticmethod
    def _normalize_scores(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        min_score = float(min(scores))
        max_score = float(max(scores))
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [
            (float(score) - min_score) / (max_score - min_score) for score in scores
        ]

    @staticmethod
    def _format_reranker_message(message: Episode) -> str:
        ts = message.created_at.isoformat()
        return f"[{ts}] {message.producer_role}: {message.content}"

    def _replay_split(
        self,
        cluster_id: str,
        messages: Sequence[Episode],
        embeddings: Sequence[Sequence[float]],
        record: ClusterSplitRecord,
        state: ClusterState,
    ) -> list[tuple[str, Sequence[Episode]]]:
        """Re-apply a previously recorded split without rescoring."""
        segment_ids = record.segment_ids
        n_segments = len(segment_ids)

        boundaries: list[int] = [0]
        search_from = 0
        for seg_id in segment_ids[1:]:
            for i in range(search_from, len(messages)):
                msg = messages[i]
                if (
                    msg.uid is not None
                    and state.event_to_cluster.get(msg.uid) == seg_id
                ):
                    boundaries.append(i)
                    search_from = i
                    break
        boundaries.append(len(messages))

        if len(boundaries) != n_segments + 1:
            return [(cluster_id, messages)]

        segments: list[tuple[str, Sequence[Episode]]] = []
        for seg_idx in range(n_segments):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            seg_messages = messages[start:end]
            seg_embeddings = embeddings[start:end]
            seg_id = segment_ids[seg_idx]

            if seg_messages:
                arr = np.array(seg_embeddings, dtype=float)
                centroid = arr.mean(axis=0).tolist()
                state.clusters[seg_id] = ClusterInfo(
                    centroid=centroid,
                    count=len(seg_messages),
                    last_ts=max(m.created_at for m in seg_messages),
                )
            segments.append((seg_id, seg_messages))

        return segments

    @staticmethod
    def _compute_signals(
        messages: Sequence[Episode],
        embeddings: Sequence[Sequence[float]],
    ) -> ContinuitySignals:
        adj_sims = _adjacent_similarities(embeddings)
        time_gaps: list[float] = []
        for i in range(len(messages) - 1):
            gap = abs(
                (messages[i + 1].created_at - messages[i].created_at).total_seconds()
            )
            time_gaps.append(gap)

        return ContinuitySignals(
            adjacent_similarities=adj_sims,
            time_gaps_seconds=time_gaps,
            min_adjacent_similarity=min(adj_sims) if adj_sims else 1.0,
            max_time_gap_seconds=max(time_gaps) if time_gaps else 0.0,
        )

    @staticmethod
    def _input_hash(event_ids: Sequence[str]) -> str:
        raw = ",".join(event_ids)
        return hashlib.sha256(raw.encode()).hexdigest()


def segment_cluster_id(original_id: str, segment_index: int) -> str:
    """Return a deterministic, collision-resistant child cluster ID."""
    raw = f"{original_id}:{segment_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def validate_split_indices(indices: Sequence[int], n: int) -> list[int]:
    """Validate and normalize split indices.

    Requirements: ints, sorted ascending, unique, in open interval (0, n).
    Returns empty list on any fundamental validation failure.
    """
    try:
        as_ints = [int(x) for x in indices]
    except (TypeError, ValueError):
        logger.warning("Invalid split indices (non-integer): %s", indices)
        return []

    valid = sorted({i for i in as_ints if 0 < i < n})
    if len(valid) != len(as_ints):
        logger.warning(
            "Split indices adjusted (original=%s, valid=%s)",
            as_ints,
            valid,
        )
    return valid


def apply_cluster_split(
    cluster_id: str,
    messages: Sequence[Episode],
    embeddings: Sequence[Sequence[float]],
    split_indices: list[int],
    state: ClusterState,
) -> tuple[list[tuple[str, Sequence[Episode]]], list[str]]:
    """Partition messages at split_indices and update ClusterState.

    The first segment retains the original cluster_id.
    Subsequent segments get deterministic child IDs.
    Returns (segments, segment_ids).
    """
    boundaries = [0, *split_indices, len(messages)]
    segments: list[tuple[str, Sequence[Episode]]] = []
    segment_ids: list[str] = []

    for seg_idx, (start, end) in enumerate(itertools.pairwise(boundaries)):
        seg_messages = messages[start:end]
        seg_embeddings = embeddings[start:end]

        seg_id = cluster_id if seg_idx == 0 else segment_cluster_id(cluster_id, seg_idx)

        arr = np.array(seg_embeddings, dtype=float)
        centroid = arr.mean(axis=0).tolist()

        state.clusters[seg_id] = ClusterInfo(
            centroid=centroid,
            count=len(seg_messages),
            last_ts=max(m.created_at for m in seg_messages),
        )
        for msg in seg_messages:
            if msg.uid is not None:
                state.event_to_cluster[msg.uid] = seg_id

        segments.append((seg_id, seg_messages))
        segment_ids.append(seg_id)

    logger.info(
        "Split cluster %s into %d segments: %s",
        cluster_id,
        len(segments),
        segment_ids,
    )

    return segments, segment_ids


def _adjacent_similarities(
    embeddings: Sequence[Sequence[float]],
) -> list[float]:
    """Return cosine similarities between consecutive embeddings."""
    result: list[float] = []
    for i in range(len(embeddings) - 1):
        a = np.array(embeddings[i], dtype=float)
        b = np.array(embeddings[i + 1], dtype=float)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        result.append(float(np.dot(a, b) / denom) if denom != 0.0 else 0.0)
    return result
