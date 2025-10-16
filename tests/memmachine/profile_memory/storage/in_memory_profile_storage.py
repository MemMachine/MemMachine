from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from memmachine.profile_memory.storage.storage_base import ProfileStorageBase


@dataclass
class _ProfileEntry:
    id: int
    user_id: str
    tag: str
    feature: str
    value: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    isolations: dict[str, bool | int | float | str]
    citations: list[int]
    created_at: float = field(default_factory=time.time)


@dataclass
class _HistoryEntry:
    id: int
    user_id: str
    content: str
    metadata: dict[str, Any]
    isolations: dict[str, bool | int | float | str]
    timestamp: float = field(default_factory=time.time)
    ingested: bool = False


class InMemoryProfileStorage(ProfileStorageBase):
    """In-memory implementation of ``ProfileStorageBase`` used for testing."""

    def __init__(self):
        self._profiles_by_user: dict[str, list[_ProfileEntry]] = {}
        self._profiles_by_id: dict[int, _ProfileEntry] = {}
        self._history_by_user: dict[str, list[_HistoryEntry]] = {}
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
            self._profiles_by_user.clear()
            self._profiles_by_id.clear()
            self._history_by_user.clear()
            self._history_by_id.clear()
            self._next_profile_id = 1
            self._next_history_id = 1

    async def get_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, dict[str, Any | list[Any]]]:
        isolations = isolations or {}
        async with self._lock:
            result: dict[str, dict[str, Any | list[Any]]] = {}
            for entry in self._profiles_by_user.get(user_id, []):
                if not self._isolations_match(entry.isolations, isolations):
                    continue
                payload = {"value": entry.value}
                tag_bucket = result.setdefault(entry.tag, {})
                values = tag_bucket.setdefault(entry.feature, [])
                values.append(payload)

            for tag, features in result.items():
                for feature, values in list(features.items()):
                    if len(values) == 1:
                        features[feature] = values[0]
            return result

    async def get_citation_list(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[int]:
        isolations = isolations or {}
        async with self._lock:
            citations: list[int] = []
            for entry in self._profiles_by_user.get(user_id, []):
                if entry.feature != feature or entry.tag != tag:
                    continue
                if entry.value != str(value):
                    continue
                if self._isolations_match(entry.isolations, isolations):
                    citations.extend(entry.citations)
            return citations

    async def delete_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations = isolations or {}
        async with self._lock:
            keep: list[_ProfileEntry] = []
            for entry in self._profiles_by_user.get(user_id, []):
                if self._isolations_match(entry.isolations, isolations):
                    self._profiles_by_id.pop(entry.id, None)
                else:
                    keep.append(entry)
            if keep:
                self._profiles_by_user[user_id] = keep
            else:
                self._profiles_by_user.pop(user_id, None)

    async def add_profile_feature(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
        citations: list[int] | None = None,
    ):
        metadata = metadata or {}
        isolations = isolations or {}
        citations = citations or []
        async with self._lock:
            entry = _ProfileEntry(
                id=self._next_profile_id,
                user_id=user_id,
                tag=tag,
                feature=feature,
                value=str(value),
                embedding=np.array(embedding, dtype=float, copy=True),
                metadata=dict(metadata),
                isolations=dict(isolations),
                citations=list(citations),
            )
            self._next_profile_id += 1
            self._profiles_by_user.setdefault(user_id, []).append(entry)
            self._profiles_by_id[entry.id] = entry

    async def semantic_search(
        self,
        user_id: str,
        qemb: np.ndarray,
        k: int,
        min_cos: float,
        isolations: dict[str, bool | int | float | str] | None = None,
        include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        isolations = isolations or {}
        async with self._lock:
            haystack = [
                entry
                for entry in self._profiles_by_user.get(user_id, [])
                if self._isolations_match(entry.isolations, isolations)
            ]
            qnorm = float(np.linalg.norm(qemb))
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
                results.append(payload)
            return results

    async def delete_profile_feature(
        self,
        user_id: str,
        feature: str,
        tag: str,
        value: str | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations = isolations or {}
        async with self._lock:
            keep: list[_ProfileEntry] = []
            for entry in self._profiles_by_user.get(user_id, []):
                if entry.feature != feature or entry.tag != tag:
                    keep.append(entry)
                    continue
                if not self._isolations_match(entry.isolations, isolations):
                    keep.append(entry)
                    continue
                if value is not None and entry.value != str(value):
                    keep.append(entry)
                    continue
                self._profiles_by_id.pop(entry.id, None)
            if keep:
                self._profiles_by_user[user_id] = keep
            else:
                self._profiles_by_user.pop(user_id, None)

    async def delete_profile_feature_by_id(self, pid: int):
        async with self._lock:
            entry = self._profiles_by_id.pop(pid, None)
            if entry is None:
                return
            user_entries = [
                e for e in self._profiles_by_user.get(entry.user_id, []) if e.id != pid
            ]
            if user_entries:
                self._profiles_by_user[entry.user_id] = user_entries
            else:
                self._profiles_by_user.pop(entry.user_id, None)

    async def get_all_citations_for_ids(
        self, pids: list[int]
    ) -> list[tuple[int, dict[str, bool | int | float | str]]]:
        async with self._lock:
            result: list[tuple[int, dict[str, bool | int | float | str]]] = []
            seen: set[int] = set()
            for pid in pids:
                entry = self._profiles_by_id.get(pid)
                if entry is None:
                    continue
                for cid in entry.citations:
                    if cid in seen:
                        continue
                    history = self._history_by_id.get(cid)
                    if history is None:
                        continue
                    seen.add(cid)
                    result.append((cid, dict(history.isolations)))
            return result

    async def get_large_profile_sections(
        self,
        user_id: str,
        thresh: int,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        isolations = isolations or {}
        async with self._lock:
            sections: dict[str, list[_ProfileEntry]] = {}
            for entry in self._profiles_by_user.get(user_id, []):
                if self._isolations_match(entry.isolations, isolations):
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
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        isolations = isolations or {}
        async with self._lock:
            entry = _HistoryEntry(
                id=self._next_history_id,
                user_id=user_id,
                content=content,
                metadata=dict(metadata),
                isolations=dict(isolations),
            )
            self._next_history_id += 1
            self._history_by_user.setdefault(user_id, []).append(entry)
            self._history_by_id[entry.id] = entry
            return self._history_entry_to_mapping(entry)

    async def delete_history(
        self,
        user_id: str,
        start_time: float = 0,
        end_time: float = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations = isolations or {}
        async with self._lock:
            start = start_time if start_time else float("-inf")
            end = end_time if end_time else float("inf")
            keep: list[_HistoryEntry] = []
            for entry in self._history_by_user.get(user_id, []):
                if not self._isolations_match(entry.isolations, isolations):
                    keep.append(entry)
                    continue
                if start <= entry.timestamp <= end:
                    self._history_by_id.pop(entry.id, None)
                else:
                    keep.append(entry)
            if keep:
                self._history_by_user[user_id] = keep
            else:
                self._history_by_user.pop(user_id, None)

    async def get_history_messages_by_ingestion_status(
        self,
        user_id: str,
        k: int = 0,
        is_ingested: bool = False,
    ) -> list[dict[str, Any]]:
        return await self._get_history_messages(
            user_id=user_id,
            k=k,
            is_ingested=is_ingested,
            isolations=None,
        )

    async def get_uningested_history_messages_count(self) -> int:
        async with self._lock:
            return sum(
                1 for entry in self._history_by_id.values() if not entry.ingested
            )

    async def mark_messages_ingested(
        self,
        ids: list[int],
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[dict[str, Any]]:
        isolations = isolations or None
        async with self._lock:
            updated: list[dict[str, Any]] = []
            for mid in ids:
                entry = self._history_by_id.get(mid)
                if entry is None:
                    continue
                if isolations is not None and not self._isolations_match(
                    entry.isolations, isolations
                ):
                    continue
                entry.ingested = True
                updated.append(self._history_entry_to_mapping(entry))
            return updated

    async def get_history_message(
        self,
        user_id: str,
        start_time: float = 0,
        end_time: float = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[str]:
        isolations = isolations or {}
        async with self._lock:
            start = start_time if start_time else float("-inf")
            end = end_time if end_time else float("inf")
            entries = [
                entry
                for entry in self._history_by_user.get(user_id, [])
                if self._isolations_match(entry.isolations, isolations)
                and start <= entry.timestamp <= end
            ]
            entries.sort(key=lambda item: item.timestamp)
            return [entry.content for entry in entries]

    async def purge_history(
        self,
        user_id: str,
        start_time: float = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations = isolations or {}
        async with self._lock:
            threshold = start_time if start_time else float("-inf")
            keep: list[_HistoryEntry] = []
            for entry in self._history_by_user.get(user_id, []):
                if not self._isolations_match(entry.isolations, isolations):
                    keep.append(entry)
                    continue
                if entry.timestamp <= threshold:
                    self._history_by_id.pop(entry.id, None)
                else:
                    keep.append(entry)
            if keep:
                self._history_by_user[user_id] = keep
            else:
                self._history_by_user.pop(user_id, None)

    async def get_ingested_history_messages(
        self,
        user_id: str,
        k: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
        is_ingested: bool = False,
    ) -> list[dict[str, Any]]:
        isolations = isolations or {}
        return await self._get_history_messages(
            user_id=user_id,
            k=k,
            is_ingested=is_ingested,
            isolations=isolations,
        )

    def _history_entry_to_mapping(self, entry: _HistoryEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "user_id": entry.user_id,
            "content": entry.content,
            "metadata": json.dumps(entry.metadata),
            "isolations": json.dumps(entry.isolations),
        }

    async def _get_history_messages(
        self,
        user_id: str,
        k: int,
        is_ingested: bool,
        isolations: dict[str, bool | int | float | str] | None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            entries = [
                entry
                for entry in self._history_by_user.get(user_id, [])
                if entry.ingested == is_ingested
                and self._isolations_match(entry.isolations, isolations)
            ]
            entries.sort(key=lambda item: item.timestamp, reverse=True)
            if k > 0:
                entries = entries[:k]
            return [self._history_entry_to_mapping(entry) for entry in entries]

    @staticmethod
    def _isolations_match(
        source: dict[str, bool | int | float | str],
        expected: dict[str, bool | int | float | str] | None,
    ) -> bool:
        if expected is None:
            return True
        for key, value in expected.items():
            if source.get(key) != value:
                return False
        return True
