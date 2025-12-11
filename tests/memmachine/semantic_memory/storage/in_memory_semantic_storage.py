from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import InstanceOf

from memmachine.common.episode_store import EpisodeIdT
from memmachine.common.errors import InvalidArgumentError
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.semantic_memory.semantic_model import (
    FeatureIdT,
    SemanticFeature,
    SetIdT,
)
from memmachine.semantic_memory.storage.storage_base import (
    SemanticStorage,
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
    citations: list[EpisodeIdT] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


class InMemorySemanticStorage(SemanticStorage):
    """In-memory implementation of :class:`SemanticStorageBase` used for testing."""

    def __init__(self):
        self._features_by_id: dict[FeatureIdT, _FeatureEntry] = {}
        self._feature_ids_by_set: dict[str, list[FeatureIdT]] = {}
        # History tracking mirrors the SetIngestedHistory table
        self._set_history_map: dict[str, dict[EpisodeIdT, bool]] = {}
        self._history_to_sets: dict[EpisodeIdT, dict[str, bool]] = {}
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
        set_id: SetIdT,
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
        set_id: SetIdT | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        embedding: InstanceOf[np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            self._handle_set_change(entry, feature_id, set_id)
            self._update_entry_fields(
                entry,
                category_name=category_name,
                feature=feature,
                value=value,
                tag=tag,
                embedding=embedding,
                metadata=metadata,
            )

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
                self._remove_feature_from_index(entry)

    async def get_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        if page_num is not None:
            if page_size is None:
                raise InvalidArgumentError("Cannot specify offset without limit")
            if page_num < 0:
                raise InvalidArgumentError("Offset must be non-negative")

        fetch_limit = page_size
        if page_size is not None and page_num:
            fetch_limit = page_size * (page_num + 1)

        async with self._lock:
            entries = self._filter_features(
                set_ids=None,
                type_names=None,
                feature_names=None,
                tags=None,
                filter_expr=filter_expr,
                k=fetch_limit,
                vector_search_opts=vector_search_opts,
                tag_threshold=tag_threshold,
            )
            if page_size is not None:
                start_index = page_size * (page_num or 0)
                entries = entries[start_index : start_index + page_size]
            return [
                self._feature_to_model(entry, load_citations=load_citations)
                for entry in entries
            ]

    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        async with self._lock:
            to_remove = self._filter_features(
                set_ids=None,
                type_names=None,
                feature_names=None,
                tags=None,
                filter_expr=filter_expr,
                k=None,
                vector_search_opts=None,
                tag_threshold=None,
            )
            for entry in to_remove:
                self._features_by_id.pop(entry.id, None)
                self._remove_feature_from_index(entry)

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: list[EpisodeIdT],
    ) -> None:
        if not history_ids:
            return

        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            existing: set[EpisodeIdT] = set(entry.citations)
            for history_id in history_ids:
                history_id = EpisodeIdT(history_id)
                if history_id not in existing:
                    entry.citations.append(history_id)
                    existing.add(history_id)
            entry.updated_at = _utcnow()

    async def get_history_messages(
        self,
        *,
        set_ids: list[SetIdT] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> list[EpisodeIdT]:
        async with self._lock:
            rows = self._history_rows_for_sets(set_ids)
            rows = self._filter_history_rows(rows, is_ingested)
            history_ids = [history_id for history_id, _ in rows]

            if limit is not None:
                history_ids = history_ids[:limit]

            return history_ids

    async def get_history_messages_count(
        self,
        *,
        set_ids: list[SetIdT] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        async with self._lock:
            rows = self._history_rows_for_sets(set_ids)
            filtered_rows = self._filter_history_rows(rows, is_ingested)
            return len(filtered_rows)

    async def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
    ) -> list[SetIdT]:
        async with self._lock:
            if min_uningested_messages is None or min_uningested_messages <= 0:
                return list(self._set_history_map.keys())

            set_ids: list[str] = []
            for set_id, history_map in self._set_history_map.items():
                uningested_count = sum(
                    1 for ingested in history_map.values() if not ingested
                )
                if uningested_count >= min_uningested_messages:
                    set_ids.append(set_id)

            return set_ids

    def _handle_set_change(
        self,
        entry: _FeatureEntry,
        feature_id: FeatureIdT,
        set_id: SetIdT | None,
    ) -> None:
        if set_id is None:
            return
        self._move_feature_to_set(entry, feature_id, set_id)

    def _update_entry_fields(
        self,
        entry: _FeatureEntry,
        *,
        category_name: str | None,
        feature: str | None,
        value: str | None,
        tag: str | None,
        embedding: InstanceOf[np.ndarray] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        simple_updates = {
            "semantic_type_id": category_name,
            "feature": feature,
            "value": value,
            "tag": tag,
        }

        for attr, new_value in simple_updates.items():
            if new_value is None:
                continue
            setattr(entry, attr, new_value)

        if embedding is not None:
            entry.embedding = np.array(embedding, dtype=float, copy=True)

        if metadata is not None:
            entry.metadata = dict(metadata) if metadata else None

    def _move_feature_to_set(
        self,
        entry: _FeatureEntry,
        feature_id: FeatureIdT,
        new_set_id: SetIdT,
    ) -> None:
        if new_set_id == entry.set_id:
            return

        current_ids = self._feature_ids_by_set.get(entry.set_id)
        if current_ids and feature_id in current_ids:
            current_ids.remove(feature_id)
            if not current_ids:
                self._feature_ids_by_set.pop(entry.set_id, None)

        self._feature_ids_by_set.setdefault(new_set_id, []).append(feature_id)
        entry.set_id = new_set_id

    def _remove_feature_from_index(self, entry: _FeatureEntry) -> None:
        ids = self._feature_ids_by_set.get(entry.set_id)
        if ids and entry.id in ids:
            ids.remove(entry.id)
            if not ids:
                self._feature_ids_by_set.pop(entry.set_id, None)

    def _history_rows_for_sets(
        self,
        set_ids: list[SetIdT] | None,
    ) -> list[tuple[EpisodeIdT, bool]]:
        rows = [
            (history_id, ingested)
            for set_id, history_map in self._set_history_map.items()
            if set_ids is None or set_id in set_ids
            for history_id, ingested in history_map.items()
        ]
        rows.sort(key=lambda pair: pair[0])
        return rows

    @staticmethod
    def _filter_history_rows(
        rows: list[tuple[EpisodeIdT, bool]],
        is_ingested: bool | None,
    ) -> list[tuple[EpisodeIdT, bool]]:
        if is_ingested is None:
            return rows
        return [pair for pair in rows if pair[1] == is_ingested]

    async def add_history_to_set(
        self,
        set_id: SetIdT,
        history_id: EpisodeIdT,
    ) -> None:
        async with self._lock:
            history_map = self._set_history_map.setdefault(set_id, {})
            history_id = EpisodeIdT(history_id)
            history_map[history_id] = history_map.get(history_id, False)
            self._history_to_sets.setdefault(history_id, {})[set_id] = history_map[
                history_id
            ]

    async def delete_history(self, history_ids: list[EpisodeIdT]) -> None:
        if not history_ids:
            return

        async with self._lock:
            for history_id in history_ids:
                normalized_id = EpisodeIdT(history_id)
                referencing_sets = self._history_to_sets.pop(normalized_id, None)
                if referencing_sets is None:
                    continue

                for set_id in list(referencing_sets.keys()):
                    history_map = self._set_history_map.get(set_id)
                    if history_map is None:
                        continue

                    history_map.pop(normalized_id, None)
                    if not history_map:
                        self._set_history_map.pop(set_id, None)

    async def delete_history_set(self, set_ids: list[SetIdT]) -> None:
        if not set_ids:
            return

        async with self._lock:
            for set_id in set_ids:
                history_map = self._set_history_map.pop(set_id, None)
                if history_map is None:
                    continue

                for history_id in list(history_map.keys()):
                    sets_map = self._history_to_sets.get(history_id)
                    if sets_map is None:
                        continue

                    sets_map.pop(set_id, None)
                    if not sets_map:
                        self._history_to_sets.pop(history_id, None)

    async def mark_messages_ingested(
        self,
        *,
        set_id: SetIdT,
        history_ids: list[EpisodeIdT],
    ) -> None:
        if not history_ids:
            raise ValueError("No ids provided")

        async with self._lock:
            set_map = self._set_history_map.get(set_id)
            if set_map is None:
                return

            for history_id in history_ids:
                history_id = EpisodeIdT(history_id)
                if history_id in set_map:
                    set_map[history_id] = True
                    self._history_to_sets.setdefault(history_id, {})[set_id] = True

    def _feature_to_model(
        self,
        entry: _FeatureEntry,
        *,
        load_citations: bool,
    ) -> SemanticFeature:
        citations: list[EpisodeIdT] | None = None
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
        filter_expr: FilterExpr | None,
        k: int | None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None,
        tag_threshold: int | None,
    ) -> list[_FeatureEntry]:
        entries = list(self._features_by_id.values())
        entries = self._apply_basic_filters(
            entries,
            set_ids=set_ids,
            type_names=type_names,
            feature_names=feature_names,
            tags=tags,
        )
        entries = self._apply_filter_expression(entries, filter_expr)
        entries = self._apply_vector_filter(entries, vector_search_opts)

        if k is not None:
            entries = entries[:k]

        if tag_threshold is not None:
            return self._apply_tag_threshold(entries, tag_threshold)

        return entries

    @staticmethod
    def _apply_basic_filters(
        entries: list[_FeatureEntry],
        *,
        set_ids: list[str] | None,
        type_names: list[str] | None,
        feature_names: list[str] | None,
        tags: list[str] | None,
    ) -> list[_FeatureEntry]:
        filters: list[tuple[list[str] | None, Callable[[_FeatureEntry], str]]] = [
            (set_ids, lambda entry: entry.set_id),
            (type_names, lambda entry: entry.semantic_type_id),
            (feature_names, lambda entry: entry.feature),
            (tags, lambda entry: entry.tag),
        ]

        filtered_entries = entries
        for allowed_values, getter in filters:
            if not allowed_values:
                continue
            allowed = set(allowed_values)
            filtered_entries = [
                entry for entry in filtered_entries if getter(entry) in allowed
            ]

        return filtered_entries

    def _apply_filter_expression(
        self,
        entries: list[_FeatureEntry],
        filter_expr: FilterExpr | None,
    ) -> list[_FeatureEntry]:
        if filter_expr is None:
            return entries
        return [
            entry for entry in entries if self._evaluate_filter_expr(entry, filter_expr)
        ]

    def _evaluate_filter_expr(
        self,
        entry: _FeatureEntry,
        expr: FilterExpr,
    ) -> bool:
        if isinstance(expr, FilterComparison):
            return self._evaluate_comparison(entry, expr)
        if isinstance(expr, FilterAnd):
            return self._evaluate_filter_expr(
                entry, expr.left
            ) and self._evaluate_filter_expr(
                entry,
                expr.right,
            )
        if isinstance(expr, FilterOr):
            return self._evaluate_filter_expr(
                entry, expr.left
            ) or self._evaluate_filter_expr(
                entry,
                expr.right,
            )
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    def _evaluate_comparison(
        self,
        entry: _FeatureEntry,
        comparison: FilterComparison,
    ) -> bool:
        value, is_metadata = self._resolve_entry_field(entry, comparison.field)
        match comparison.op:
            case "=":
                return self._compare_equals(value, comparison.value, is_metadata)
            case "in":
                return self._compare_in(value, comparison.value, is_metadata)
            case ">" | "<" | ">=" | "<=":
                return self._compare_order(
                    value,
                    comparison.value,
                    is_metadata,
                    comparison.op,
                )
            case "is_null":
                return value is None
            case "is_not_null":
                return value is not None
            case _:
                raise ValueError(f"Unsupported operator: {comparison.op}")

    def _compare_equals(self, value: Any, expected: Any, is_metadata: bool) -> bool:
        if isinstance(expected, list):
            raise TypeError("'=' comparison cannot accept list values")
        if is_metadata and expected is not None:
            expected = self._normalize_metadata_value(expected)
        if is_metadata and value is not None:
            value = self._normalize_metadata_value(value)
        return value == expected

    def _compare_in(self, value: Any, candidates: Any, is_metadata: bool) -> bool:
        if not isinstance(candidates, list):
            raise TypeError("IN comparison requires a list of values")
        if is_metadata:
            candidates = [
                self._normalize_metadata_value(v) if v is not None else None
                for v in candidates
            ]
            if value is not None:
                value = self._normalize_metadata_value(value)
        return value in candidates

    def _compare_order(
        self,
        value: Any,
        expected: Any,
        is_metadata: bool,
        op: str,
    ) -> bool:
        if isinstance(expected, list):
            raise TypeError(f"'{op}' comparison cannot accept list values")
        if value is None:
            return False
        if is_metadata:
            expected = self._normalize_metadata_value(expected)
            value = self._normalize_metadata_value(value)
        operations: dict[str, Callable[[Any, Any], bool]] = {
            ">": lambda lhs, rhs: lhs > rhs,
            "<": lambda lhs, rhs: lhs < rhs,
            ">=": lambda lhs, rhs: lhs >= rhs,
            "<=": lambda lhs, rhs: lhs <= rhs,
        }
        return operations[op](value, expected)

    def _resolve_entry_field(
        self,
        entry: _FeatureEntry,
        field: str,
    ) -> tuple[Any, bool]:
        field_mapping: dict[str, Any] = {
            "set_id": entry.set_id,
            "set": entry.set_id,
            "semantic_category_id": entry.semantic_type_id,
            "category_name": entry.semantic_type_id,
            "category": entry.semantic_type_id,
            "tag_id": entry.tag,
            "tag": entry.tag,
            "feature": entry.feature,
            "feature_name": entry.feature,
            "value": entry.value,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }
        if field in field_mapping:
            return field_mapping[field], False

        if field.startswith(("m.", "metadata.")):
            key = field.split(".", 1)[1]
            metadata = entry.metadata or {}
            return metadata.get(key), True

        return None, False

    @staticmethod
    def _normalize_metadata_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        return str(value)

    def _apply_vector_filter(
        self,
        entries: list[_FeatureEntry],
        vector_search_opts: SemanticStorage.VectorSearchOpts | None,
    ) -> list[_FeatureEntry]:
        if vector_search_opts is None:
            return sorted(entries, key=lambda e: (e.created_at, e.id))

        scored_entries: list[tuple[float, _FeatureEntry]] = []
        for entry in entries:
            similarity = self._resolve_similarity(
                entry.embedding,
                vector_search_opts.query_embedding,
            )
            if not self._passes_min_distance(
                similarity,
                vector_search_opts.min_distance,
            ):
                continue
            scored_entries.append((similarity, entry))

        scored_entries.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored_entries]

    @staticmethod
    def _apply_tag_threshold(
        entries: list[_FeatureEntry],
        tag_threshold: int,
    ) -> list[_FeatureEntry]:
        counts = Counter(entry.tag for entry in entries)
        return [entry for entry in entries if counts[entry.tag] >= tag_threshold]

    @staticmethod
    def _resolve_similarity(
        embedding: np.ndarray,
        query_embedding: np.ndarray,
    ) -> float:
        similarity = _cosine_similarity(embedding, query_embedding)
        return similarity if similarity is not None else float("-inf")

    @staticmethod
    def _passes_min_distance(similarity: float, min_distance: float | None) -> bool:
        if min_distance is None:
            return True
        return similarity >= min_distance
