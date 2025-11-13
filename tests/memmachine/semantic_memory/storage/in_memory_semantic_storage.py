from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import InstanceOf

from memmachine.history_store.history_model import HistoryIdT
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.semantic_memory.storage.storage_base import (
    FeatureIdT,
    SemanticStorageBase,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    lnorm = float(np.linalg.norm(lhs))
    rnorm = float(np.linalg.norm(rhs))
    if lnorm == 0 or rnorm == 0:
        return None
    return float(np.dot(lhs, rhs) / (lnorm * rnorm))


@dataclass
class _FeatureEntry:
    id: FeatureIdT
    set_id: str
    semantic_type_id: str
    tag: str
    feature: str
    value: str
    embedding: np.ndarray
    metadata: dict[str, Any] | None = None
    citations: list[HistoryIdT] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


class InMemorySemanticStorage(SemanticStorageBase):
    """In-memory implementation of :class:`SemanticStorageBase` used for testing."""

    def __init__(self):
        self._features_by_id: dict[FeatureIdT, _FeatureEntry] = {}
        self._feature_ids_by_set: dict[str, list[FeatureIdT]] = {}
        # History tracking mirrors the SetIngestedHistory table
        self._set_history_map: dict[str, dict[HistoryIdT, bool]] = {}
        self._history_to_sets: dict[HistoryIdT, dict[str, bool]] = {}
        self._next_feature_id = 1
        self._next_history_id = 1
        self._lock = asyncio.Lock()

    async def startup(self):
        return None

    async def cleanup(self):
        return None

    async def delete_all(self):
        async with self._lock:
            self._features_by_id.clear()
            self._feature_ids_by_set.clear()
            self._set_history_map.clear()
            self._history_to_sets.clear()
            self._next_feature_id = 1
            self._next_history_id = 1

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return None
            return self._feature_to_model(entry, load_citations=load_citations)

    async def add_feature(
        self,
        *,
        set_id: str,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> FeatureIdT:
        metadata = dict(metadata or {}) or None
        async with self._lock:
            feature_id = FeatureIdT(str(self._next_feature_id))
            self._next_feature_id += 1
            assert isinstance(feature_id, FeatureIdT)
            entry = _FeatureEntry(
                id=feature_id,
                set_id=set_id,
                semantic_type_id=category_name,
                tag=tag,
                feature=feature,
                value=value,
                embedding=np.array(embedding, dtype=float, copy=True),
                metadata=metadata,
            )
            self._features_by_id[feature_id] = entry
            self._feature_ids_by_set.setdefault(set_id, []).append(feature_id)
            return feature_id

    async def update_feature(
        self,
        feature_id: FeatureIdT,
        *,
        set_id: str | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        embedding: InstanceOf[np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            if set_id is not None and set_id != entry.set_id:
                old_set_ids = self._feature_ids_by_set.get(entry.set_id, [])
                if feature_id in old_set_ids:
                    old_set_ids.remove(feature_id)
                    if not old_set_ids:
                        self._feature_ids_by_set.pop(entry.set_id, None)
                self._feature_ids_by_set.setdefault(set_id, []).append(feature_id)
                entry.set_id = set_id

            if category_name is not None:
                entry.semantic_type_id = category_name
            if feature is not None:
                entry.feature = feature
            if value is not None:
                entry.value = value
            if tag is not None:
                entry.tag = tag
            if embedding is not None:
                entry.embedding = np.array(embedding, dtype=float, copy=True)
            if metadata is not None:
                entry.metadata = dict(metadata) if metadata else None

            entry.updated_at = _utcnow()

    async def delete_features(self, feature_ids: list[FeatureIdT]):
        if not feature_ids:
            return

        async with self._lock:
            for feature_id in feature_ids:
                feature_id = self._normalize_feature_id(feature_id)
                entry = self._features_by_id.pop(feature_id, None)
                if entry is None:
                    continue
                ids = self._feature_ids_by_set.get(entry.set_id)
                if ids and feature_id in ids:
                    ids.remove(feature_id)
                    if not ids:
                        self._feature_ids_by_set.pop(entry.set_id, None)

    async def get_feature_set(
        self,
        *,
        set_ids: list[str] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        vector_search_opts: SemanticStorageBase.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        async with self._lock:
            entries = self._filter_features(
                set_ids=set_ids,
                type_names=category_names,
                feature_names=feature_names,
                tags=tags,
                k=limit,
                vector_search_opts=vector_search_opts,
                tag_threshold=tag_threshold,
            )
            return [
                self._feature_to_model(entry, load_citations=load_citations)
                for entry in entries
            ]

    async def delete_feature_set(
        self,
        *,
        set_ids: list[str] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        thresh: int | None = None,
        limit: int | None = None,
        vector_search_opts: SemanticStorageBase.VectorSearchOpts | None = None,
    ):
        async with self._lock:
            to_remove = self._filter_features(
                set_ids=set_ids,
                type_names=category_names,
                feature_names=feature_names,
                tags=tags,
                k=limit,
                vector_search_opts=vector_search_opts,
                tag_threshold=thresh,
            )
            for entry in to_remove:
                self._features_by_id.pop(entry.id, None)
                ids = self._feature_ids_by_set.get(entry.set_id)
                if ids and entry.id in ids:
                    ids.remove(entry.id)
                    if not ids:
                        self._feature_ids_by_set.pop(entry.set_id, None)

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: list[HistoryIdT],
    ):
        if not history_ids:
            return

        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            existing: set[HistoryIdT] = set(entry.citations)
            for history_id in history_ids:
                history_id = HistoryIdT(history_id)
                if history_id not in existing:
                    entry.citations.append(history_id)
                    existing.add(history_id)
            entry.updated_at = _utcnow()

    async def get_history_messages(
        self,
        *,
        set_ids: list[str] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> list[HistoryIdT]:
        async with self._lock:
            rows: list[tuple[HistoryIdT, bool]] = []
            for set_id, history_map in self._set_history_map.items():
                if set_ids is not None and set_id not in set_ids:
                    continue
                for history_id, ingested in history_map.items():
                    rows.append((history_id, ingested))

            rows.sort(key=lambda pair: pair[0])

            if is_ingested is not None:
                rows = [pair for pair in rows if pair[1] == is_ingested]

            history_ids = [history_id for history_id, _ in rows]

            if limit is not None:
                history_ids = history_ids[:limit]

            return history_ids

    async def get_history_messages_count(
        self,
        *,
        set_ids: list[str] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        async with self._lock:
            count = 0
            for set_id, history_map in self._set_history_map.items():
                if set_ids is not None and set_id not in set_ids:
                    continue
                if is_ingested is None:
                    count += len(history_map)
                else:
                    count += sum(
                        1 for status in history_map.values() if status == is_ingested
                    )
            return count

    async def add_history_to_set(
        self,
        set_id: str,
        history_id: HistoryIdT,
    ) -> None:
        async with self._lock:
            history_map = self._set_history_map.setdefault(set_id, {})
            history_id = HistoryIdT(history_id)
            history_map[history_id] = history_map.get(history_id, False)
            self._history_to_sets.setdefault(history_id, {})[set_id] = history_map[
                history_id
            ]

    async def mark_messages_ingested(
        self,
        *,
        set_id: str,
        history_ids: list[HistoryIdT],
    ) -> None:
        if not history_ids:
            raise ValueError("No ids provided")

        async with self._lock:
            set_map = self._set_history_map.get(set_id)
            if set_map is None:
                return

            for history_id in history_ids:
                history_id = HistoryIdT(history_id)
                if history_id in set_map:
                    set_map[history_id] = True
                    self._history_to_sets.setdefault(history_id, {})[set_id] = True

    def _feature_to_model(
        self,
        entry: _FeatureEntry,
        *,
        load_citations: bool,
    ) -> SemanticFeature:
        citations: list[HistoryIdT] | None = None
        if load_citations:
            citations = list(entry.citations)

        feature_id = entry.id
        assert isinstance(feature_id, FeatureIdT)

        return SemanticFeature(
            set_id=entry.set_id,
            category=entry.semantic_type_id,
            tag=entry.tag,
            feature_name=entry.feature,
            value=entry.value,
            metadata=SemanticFeature.Metadata(
                id=feature_id,
                citations=citations,
                other=dict(entry.metadata) if entry.metadata else None,
            ),
        )

    @staticmethod
    def _normalize_feature_id(feature_id: FeatureIdT) -> FeatureIdT:
        return FeatureIdT(feature_id)

    def _filter_features(
        self,
        *,
        set_ids: list[str] | None,
        type_names: list[str] | None,
        feature_names: list[str] | None,
        tags: list[str] | None,
        k: int | None,
        vector_search_opts: SemanticStorageBase.VectorSearchOpts | None,
        tag_threshold: int | None,
    ) -> list[_FeatureEntry]:
        entries: list[_FeatureEntry] = list(self._features_by_id.values())

        if set_ids:
            allowed = set(set_ids)
            entries = [entry for entry in entries if entry.set_id in allowed]
        if type_names:
            allowed = set(type_names)
            entries = [entry for entry in entries if entry.semantic_type_id in allowed]
        if feature_names:
            allowed = set(feature_names)
            entries = [entry for entry in entries if entry.feature in allowed]
        if tags:
            allowed = set(tags)
            entries = [entry for entry in entries if entry.tag in allowed]

        if vector_search_opts is not None:
            filtered: list[tuple[float, _FeatureEntry]] = []
            for entry in entries:
                similarity = _cosine_similarity(
                    entry.embedding, vector_search_opts.query_embedding
                )
                if similarity is None:
                    similarity = float("-inf")
                min_cos = vector_search_opts.min_distance
                if min_cos is not None and similarity < min_cos:
                    continue
                filtered.append((similarity, entry))

            filtered.sort(key=lambda pair: pair[0], reverse=True)
            entries = [entry for _, entry in filtered]
        else:
            entries.sort(key=lambda e: (e.created_at, e.id))

        if k is not None:
            entries = entries[:k]

        if tag_threshold is not None:
            counts = Counter(entry.tag for entry in entries)
            entries = [entry for entry in entries if counts[entry.tag] >= tag_threshold]

        return entries
