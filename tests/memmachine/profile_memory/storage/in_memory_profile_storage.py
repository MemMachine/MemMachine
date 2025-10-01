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


class InMemoryProfileStorage(ProfileStorageBase):
    """Simple in-memory implementation of ``ProfileStorageBase``."""

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
        if isolations is None:
            isolations = {}
        async with self._lock:
            entries = self._profiles_by_user.get(user_id, [])
            result: dict[str, dict[str, Any | list[Any]]] = {}
            for entry in entries:
                if not self._matches(entry.isolations, isolations):
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

    async def delete_profile(
            self,
            user_id: str,
            isolations: dict[str, bool | int | float | str] | None = None,
    ):
        if isolations is None:
            isolations = {}
        async with self._lock:
            keep: list[_ProfileEntry] = []
            for entry in self._profiles_by_user.get(user_id, []):
                if self._matches(entry.isolations, isolations):
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
        if metadata is None:
            metadata = {}
        if isolations is None:
            isolations = {}
        if citations is None:
            citations = []
        async with self._lock:
            entry = _ProfileEntry(
                id=self._next_profile_id,
                user_id=user_id,
                tag=tag,
                feature=feature,
                value=str(value),
                embedding=np.array(embedding, copy=True),
                metadata=dict(metadata),
                isolations=dict(isolations),
                citations=list(citations),
            )
            self._next_profile_id += 1
            self._profiles_by_user.setdefault(user_id, []).append(entry)
            self._profiles_by_id[entry.id] = entry

    async def delete_profile_feature(
            self,
            user_id: str,
            feature: str,
            tag: str,
            value: str | None = None,
            isolations: dict[str, bool | int | float | str] | None = None,
    ):
        if isolations is None:
            isolations = {}
        async with self._lock:
            keep: list[_ProfileEntry] = []
            for entry in self._profiles_by_user.get(user_id, []):
                if entry.feature != feature or entry.tag != tag:
                    keep.append(entry)
                    continue
                if not self._matches(entry.isolations, isolations):
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
            user_entries = self._profiles_by_user.get(entry.user_id, [])
            user_entries = [e for e in user_entries if e.id != pid]
            if user_entries:
                self._profiles_by_user[entry.user_id] = user_entries
            else:
                self._profiles_by_user.pop(entry.user_id, None)

    async def get_citation_list(
            self,
            user_id: str,
            feature: str,
            value: str,
            tag: str,
            isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[int]:
        if isolations is None:
            isolations = {}
        async with self._lock:
            citations: list[int] = []
            for entry in self._profiles_by_user.get(user_id, []):
                if entry.feature != feature or entry.tag != tag:
                    continue
                if entry.value != str(value):
                    continue
                if self._matches(entry.isolations, isolations):
                    citations.extend(entry.citations)
            return citations

    async def get_all_citations_for_ids(
            self, pids: list[int]
    ) -> list[tuple[int, dict[str, bool | int | float | str]]]:
        async with self._lock:
            out: list[tuple[int, dict[str, bool | int | float | str]]] = []
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
                    out.append((cid, dict(history.isolations)))
            return out

    async def get_large_profile_sections(
            self,
            user_id: str,
            thresh: int,
            isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        if isolations is None:
            isolations = {}
        async with self._lock:
            sections: dict[str, list[_ProfileEntry]] = {}
            for entry in self._profiles_by_user.get(user_id, []):
                if self._matches(entry.isolations, isolations):
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

    async def semantic_search(
            self,
            user_id: str,
            qemb: np.ndarray,
            k: int,
            min_cos: float,
            isolations: dict[str, bool | int | float | str] | None = None,
            include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        if isolations is None:
            isolations = {}
        async with self._lock:
            entries = [
                entry
                for entry in self._profiles_by_user.get(user_id, [])
                if self._matches(entry.isolations, isolations)
            ]
            results: list[tuple[float, _ProfileEntry]] = []
            qnorm = np.linalg.norm(qemb)
            for entry in entries:
                denom = np.linalg.norm(entry.embedding) * qnorm
                if denom == 0:
                    continue
                sim = float(np.dot(entry.embedding, qemb) / denom)
                if sim > min_cos:
                    results.append((sim, entry))
            results.sort(key=lambda item: item[0], reverse=True)
            limited = results if k <= 0 else results[:k]
            payloads: list[dict[str, Any]] = []
            for sim, entry in limited:
                data = {
                    "tag": entry.tag,
                    "feature": entry.feature,
                    "value": entry.value,
                    "metadata": {
                        "id": entry.id,
                        "similarity_score": sim,
                    },
                }
                if include_citations:
                    data["metadata"]["citations"] = [
                        self._history_by_id[cid].content
                        for cid in entry.citations
                        if cid in self._history_by_id
                    ]
                payloads.append(data)
            return payloads

    async def add_history(
            self,
            user_id: str,
            content: str,
            metadata: dict[str, Any] | None = None,
            isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        if isolations is None:
            isolations = {}
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
            return {
                "id": entry.id,
                "user_id": entry.user_id,
                "content": entry.content,
                "metadata": json.dumps(entry.metadata),
                "isolations": json.dumps(entry.isolations),
            }

    async def delete_history(
            self,
            user_id: str,
            start_time: float = 0,
            end_time: float = 0,
            isolations: dict[str, bool | int | float | str] | None = None,
    ):
        if isolations is None:
            isolations = {}
        async with self._lock:
            start = start_time if start_time else float("-inf")
            end = end_time if end_time else float("inf")
            keep: list[_HistoryEntry] = []
            for entry in self._history_by_user.get(user_id, []):
                if not self._matches(entry.isolations, isolations):
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

    async def get_last_history_messages(
            self,
            user_id: str,
            k: int = 0,
            isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[dict[str, Any]]:
        if isolations is None:
            isolations = {}
        async with self._lock:
            entries = [
                entry
                for entry in self._history_by_user.get(user_id, [])
                if self._matches(entry.isolations, isolations)
            ]
            entries.sort(key=lambda item: item.timestamp, reverse=True)
            if k > 0:
                entries = entries[:k]
            return [
                {
                    "id": entry.id,
                    "user_id": entry.user_id,
                    "content": entry.content,
                    "metadata": json.dumps(entry.metadata),
                    "isolations": json.dumps(entry.isolations),
                }
                for entry in entries
            ]

    async def get_history_message(
            self,
            user_id: str,
            start_time: float = 0,
            end_time: float = 0,
            isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[str]:
        if isolations is None:
            isolations = {}
        async with self._lock:
            start = start_time if start_time else float("-inf")
            end = end_time if end_time else float("inf")
            return [
                entry.content
                for entry in sorted(
                    self._history_by_user.get(user_id, []),
                    key=lambda item: item.timestamp,
                )
                if self._matches(entry.isolations, isolations)
                   and start <= entry.timestamp <= end
            ]

    async def purge_history(
            self,
            user_id: str,
            start_time: float = 0,
            isolations: dict[str, bool | int | float | str] | None = None,
    ):
        if isolations is None:
            isolations = {}
        async with self._lock:
            threshold = start_time if start_time else float("-inf")
            keep: list[_HistoryEntry] = []
            for entry in self._history_by_user.get(user_id, []):
                if not self._matches(entry.isolations, isolations):
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

    def _matches(
            self,
            source: dict[str, bool | int | float | str],
            expected: dict[str, bool | int | float | str],
    ) -> bool:
        for key, value in expected.items():
            if source.get(key) != value:
                return False
        return True
