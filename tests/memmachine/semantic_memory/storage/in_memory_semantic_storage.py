from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np

from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


@dataclass
class _ProfileEntry:
    id: int
    set_id: str
    semantic_type_id: str
    tag: str
    feature: str
    value: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    citations: list[int]
    created_at: float = field(default_factory=time.time)


@dataclass
class _HistoryEntry:
    id: int
    set_id: str
    content: str
    metadata: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ingested: bool = False


class InMemorySemanticStorage(SemanticStorageBase):
    """In-memory implementation of :class:`SemanticStorageBase` used for testing."""

    def __init__(self):
        self._profiles_by_set: dict[str, list[_ProfileEntry]] = {}
        self._profiles_by_id: dict[int, _ProfileEntry] = {}
        self._history_by_set: dict[str, list[_HistoryEntry]] = {}
        self._history_by_id: dict[int, _HistoryEntry] = {}
        self._next_profile_id = 1
        self._next_history_id = 1
        self._lock = asyncio.Lock()

    async def startup(self):
        return None

    async def cleanup(self):
        return None

    async def delete_all(self):
        async with self._lock:
            self._profiles_by_set.clear()
            self._profiles_by_id.clear()
            self._history_by_set.clear()
            self._history_by_id.clear()
            self._next_profile_id = 1
            self._next_history_id = 1

    async def get_set_features(
        self,
        *,
        set_id: str,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> dict[str, dict[str, Any | list[Any]]]:
        async with self._lock:
            result: dict[str, dict[str, Any | list[Any]]] = {}
            entries = self._profiles_by_set.get(set_id, [])
            for entry in entries:
                if semantic_type_id is not None and (
                    entry.semantic_type_id != semantic_type_id
                ):
                    continue
                if tag is not None and entry.tag != tag:
                    continue
                payload = {"value": entry.value}
                if entry.metadata:
                    payload["metadata"] = dict(entry.metadata)

                tag_bucket = result.setdefault(entry.tag, {})
                values = tag_bucket.setdefault(entry.feature, [])
                values.append(payload)

            for tag_name, features in result.items():
                for feature_name, values in list(features.items()):
                    if len(values) == 1:
                        features[feature_name] = values[0]
            return result

    async def get_citation_list(
        self,
        *,
        set_id: str,
        feature: str,
        value: str,
        tag: str,
    ) -> list[int]:
        async with self._lock:
            citations: list[int] = []
            for entry in self._profiles_by_set.get(set_id, []):
                if entry.feature != feature or entry.tag != tag:
                    continue
                if entry.value != str(value):
                    continue
                citations.extend(entry.citations)
            return citations

    async def delete_feature_set(
        self,
        *,
        set_id: str,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        async with self._lock:
            keep: list[_ProfileEntry] = []
            for entry in self._profiles_by_set.get(set_id, []):
                if semantic_type_id is not None and (
                    entry.semantic_type_id != semantic_type_id
                ):
                    keep.append(entry)
                    continue
                if tag is not None and entry.tag != tag:
                    keep.append(entry)
                    continue
                self._profiles_by_id.pop(entry.id, None)

            if keep:
                self._profiles_by_set[set_id] = keep
            else:
                self._profiles_by_set.pop(set_id, None)

    async def add_feature(
        self,
        *,
        set_id: str,
        semantic_type_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        citations: list[int] | None = None,
    ):
        metadata = metadata or {}
        citations = citations or []
        async with self._lock:
            entry = _ProfileEntry(
                id=self._next_profile_id,
                set_id=set_id,
                semantic_type_id=semantic_type_id,
                tag=tag,
                feature=feature,
                value=str(value),
                embedding=np.array(embedding, dtype=float, copy=True),
                metadata=dict(metadata),
                citations=list(citations),
            )
            self._next_profile_id += 1
            self._profiles_by_set.setdefault(set_id, []).append(entry)
            self._profiles_by_id[entry.id] = entry
            return entry.id

    async def semantic_search(
        self,
        *,
        set_id: str,
        qemb: np.ndarray,
        k: int,
        min_cos: float,
        include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            haystack = list(self._profiles_by_set.get(set_id, []))
            qnorm = float(np.linalg.norm(qemb))
            if qnorm == 0:
                return []

            hits: list[tuple[float, _ProfileEntry]] = []
            for entry in haystack:
                denom = float(np.linalg.norm(entry.embedding)) * qnorm
                if denom == 0:
                    continue
                score = float(np.dot(entry.embedding, qemb) / denom)
                if score > min_cos:
                    hits.append((score, entry))

            hits.sort(key=lambda item: item[0], reverse=True)
            if k > 0:
                hits = hits[:k]

            results: list[dict[str, Any]] = []
            for score, entry in hits:
                payload: dict[str, Any] = {
                    "tag": entry.tag,
                    "feature": entry.feature,
                    "value": entry.value,
                    "metadata": {
                        "id": entry.id,
                        "similarity_score": score,
                    },
                }
                if include_citations:
                    payload["metadata"]["citations"] = [
                        self._history_by_id[cid].content
                        for cid in entry.citations
                        if cid in self._history_by_id
                    ]
                if entry.metadata:
                    payload["metadata"].update(entry.metadata)
                results.append(payload)
            return results

    async def delete_features(self, feature_ids: list[int]):
        if not feature_ids:
            return
        async with self._lock:
            for fid in feature_ids:
                entry = self._profiles_by_id.pop(fid, None)
                if entry is None:
                    continue
                items = [
                    e
                    for e in self._profiles_by_set.get(entry.set_id, [])
                    if e.id != fid
                ]
                if items:
                    self._profiles_by_set[entry.set_id] = items
                else:
                    self._profiles_by_set.pop(entry.set_id, None)

    async def get_all_citations_for_ids(self, feature_ids: list[int]) -> list[int]:
        if not feature_ids:
            return []
        async with self._lock:
            citations: list[int] = []
            seen: set[int] = set()
            for fid in feature_ids:
                entry = self._profiles_by_id.get(fid)
                if entry is None:
                    continue
                for cid in entry.citations:
                    if cid in seen:
                        continue
                    seen.add(cid)
                    citations.append(cid)
            return citations

    async def delete_feature_with_filter(
        self,
        *,
        set_id: str,
        semantic_type_id: str,
        feature: str,
        tag: str,
    ):
        async with self._lock:
            keep: list[_ProfileEntry] = []
            for entry in self._profiles_by_set.get(set_id, []):
                if entry.semantic_type_id != semantic_type_id:
                    keep.append(entry)
                    continue
                if entry.feature != feature or entry.tag != tag:
                    keep.append(entry)
                    continue
                self._profiles_by_id.pop(entry.id, None)

            if keep:
                self._profiles_by_set[set_id] = keep
            else:
                self._profiles_by_set.pop(set_id, None)

    async def get_large_feature_sections(
        self,
        *,
        set_id: str,
        thresh: int,
    ) -> list[list[dict[str, Any]]]:
        async with self._lock:
            sections: dict[str, list[_ProfileEntry]] = {}
            for entry in self._profiles_by_set.get(set_id, []):
                sections.setdefault(entry.tag, []).append(entry)

            result: list[list[dict[str, Any]]] = []
            for entries in sections.values():
                if len(entries) < thresh:
                    continue
                section = [
                    {
                        "tag": entry.tag,
                        "feature": entry.feature,
                        "value": entry.value,
                        "metadata": {"id": entry.id},
                    }
                    for entry in entries
                ]
                result.append(section)
            return result

    async def add_history(
        self,
        set_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> Mapping[str, Any]:
        metadata = dict(metadata or {})
        async with self._lock:
            entry = _HistoryEntry(
                id=self._next_history_id,
                set_id=set_id,
                content=content,
                metadata=metadata,
            )
            self._next_history_id += 1
            self._history_by_set.setdefault(set_id, []).append(entry)
            self._history_by_id[entry.id] = entry
            return self._history_entry_to_mapping(entry)

    async def delete_history(
        self,
        *,
        set_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        async with self._lock:
            start = start_time.timestamp() if start_time else float("-inf")
            end = end_time.timestamp() if end_time else float("inf")
            keep: list[_HistoryEntry] = []
            for entry in self._history_by_set.get(set_id, []):
                if start <= entry.timestamp <= end:
                    self._history_by_id.pop(entry.id, None)
                else:
                    keep.append(entry)
            if keep:
                self._history_by_set[set_id] = keep
            else:
                self._history_by_set.pop(set_id, None)

    async def get_history_messages_by_ingestion_status(
        self,
        *,
        set_id: str,
        k: int = 0,
        is_ingested: bool = False,
    ) -> list[Mapping[str, Any]]:
        return await self._get_history_messages(
            set_id=set_id,
            k=k,
            is_ingested=is_ingested,
        )

    async def get_uningested_history_messages_count(self) -> int:
        async with self._lock:
            return sum(
                1 for entry in self._history_by_id.values() if not entry.ingested
            )

    async def mark_messages_ingested(self, *, ids: list[int]) -> None:
        if not ids:
            return
        async with self._lock:
            for mid in ids:
                entry = self._history_by_id.get(mid)
                if entry is None:
                    continue
                entry.ingested = True

    async def get_history_message(
        self,
        *,
        set_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[str]:
        async with self._lock:
            start = start_time.timestamp() if start_time else float("-inf")
            end = end_time.timestamp() if end_time else float("inf")
            entries = [
                entry
                for entry in self._history_by_set.get(set_id, [])
                if start <= entry.timestamp <= end
            ]
            entries.sort(key=lambda item: item.timestamp)
            return [entry.content for entry in entries]

    async def get_ingested_history_messages(
        self,
        set_id: str,
        k: int = 0,
        is_ingested: bool = False,
    ) -> list[Mapping[str, Any]]:
        return await self._get_history_messages(
            set_id=set_id,
            k=k,
            is_ingested=is_ingested,
        )

    def _history_entry_to_mapping(self, entry: _HistoryEntry) -> Mapping[str, Any]:
        return {
            "id": entry.id,
            "set_id": entry.set_id,
            "content": entry.content,
            "metadata": dict(entry.metadata),
        }

    async def _get_history_messages(
        self,
        *,
        set_id: str,
        k: int,
        is_ingested: bool,
    ) -> list[Mapping[str, Any]]:
        async with self._lock:
            entries = [
                entry
                for entry in self._history_by_set.get(set_id, [])
                if entry.ingested == is_ingested
            ]
            entries.sort(key=lambda item: item.timestamp)
            if k > 0:
                entries = entries[:k]
            return [self._history_entry_to_mapping(entry) for entry in entries]
