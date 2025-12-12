"""Neo4j-backed implementation of :class:`SemanticStorageBase`."""

from __future__ import annotations

import json
import re
from asyncio import Lock
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from neo4j import AsyncDriver
from neo4j.graph import Node as Neo4jNode
from pydantic import InstanceOf

from memmachine.common.data_types import FilterablePropertyValue
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
from memmachine.semantic_memory.semantic_model import SemanticFeature, SetIdT
from memmachine.semantic_memory.storage.storage_base import (
    FeatureIdT,
    SemanticStorage,
)


def _utc_timestamp() -> float:
    return datetime.now(UTC).timestamp()


@dataclass
class _FeatureEntry:
    feature_id: FeatureIdT
    set_id: str
    category_name: str
    tag: str
    feature_name: str
    value: str
    embedding: np.ndarray
    metadata: dict[str, Any] | None
    citations: list[EpisodeIdT]
    created_at_ts: float
    updated_at_ts: float


def _required_str_prop(props: Mapping[str, Any], key: str) -> str:
    value = props.get(key)
    if value is None:
        raise ValueError(f"Feature node missing '{key}' property")
    return str(value)


def _sanitize_identifier(value: str) -> str:
    """Sanitize user-provided ids for Neo4j labels/index names."""
    if not value:
        return "_u0_"
    sanitized = []
    for char in value:
        if char.isalnum():
            sanitized.append(char)
        else:
            sanitized.append(f"_u{ord(char):x}_")
    return "".join(sanitized)


def _desanitize_identifier(value: str) -> str:
    """Inverse of :func:`_sanitize_identifier`."""
    if not value:
        return ""

    def _replace(match: re.Match[str]) -> str:
        hex_part = match.group(1)
        try:
            return chr(int(hex_part, 16))
        except ValueError:
            return match.group(0)

    return re.sub(r"_u([0-9A-Fa-f]+)_", _replace, value)


class Neo4jSemanticStorage(SemanticStorage):
    """Concrete :class:`SemanticStorageBase` backed by Neo4j."""

    _VECTOR_INDEX_PREFIX = "feature_embedding_index"
    _DEFAULT_VECTOR_QUERY_CANDIDATES = 100
    _SET_LABEL_PREFIX = "FeatureSet_"
    _METADATA_PROP_PREFIX = "metadata__"

    def __init__(
        self,
        driver: InstanceOf[AsyncDriver],
        owns_driver: bool = False,
    ) -> None:
        """Initialize the storage with a Neo4j driver."""
        self._driver = driver
        self._owns_driver = owns_driver
        # Exposed for fixtures to know which backend is in use
        self.backend_name = "neo4j"
        self._vector_index_by_set: dict[str, int] = {}
        self._set_embedding_dimensions: dict[str, int] = {}
        self._filter_param_counter = 0

        self._vector_global_lock = Lock()
        self._vector_index_creation_lock: dict[str, Lock] = {}

    async def startup(self) -> None:
        await self._driver.execute_query(
            """
            CREATE CONSTRAINT set_history_unique IF NOT EXISTS
            FOR (h:SetHistory)
            REQUIRE (h.set_id, h.history_id) IS UNIQUE
            """,
        )
        await self._driver.execute_query(
            """
            CREATE CONSTRAINT set_embedding_unique IF NOT EXISTS
            FOR (s:SetEmbedding)
            REQUIRE s.set_id IS UNIQUE
            """,
        )
        await self._backfill_embedding_dimensions()
        await self._load_set_embedding_dimensions()
        await self._ensure_existing_set_labels()
        await self._hydrate_vector_index_state()

    async def cleanup(self) -> None:
        if self._owns_driver:
            await self._driver.close()

    async def delete_all(self) -> None:
        await self._driver.execute_query("MATCH (f:Feature) DETACH DELETE f")
        await self._driver.execute_query("MATCH (h:SetHistory) DELETE h")
        await self._driver.execute_query("MATCH (s:SetEmbedding) DELETE s")
        records, _, _ = await self._driver.execute_query(
            """
            SHOW VECTOR INDEXES
            YIELD name
            WHERE name STARTS WITH $prefix
            RETURN name
            """,
            prefix=self._VECTOR_INDEX_PREFIX,
        )
        for record in records:
            record_data = dict(record)
            index_name = record_data.get("name")
            if not index_name:
                continue
            await self._driver.execute_query(f"DROP INDEX {index_name} IF EXISTS")
        self._vector_index_by_set.clear()
        self._set_embedding_dimensions.clear()

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
        timestamp = _utc_timestamp()
        dimensions = len(np.array(embedding, dtype=float))

        await self._ensure_set_embedding_dimensions(set_id, dimensions)

        set_label = self._set_label_for_set(set_id)
        metadata_json, metadata_props = self._prepare_metadata_storage(metadata)

        records, _, _ = await self._driver.execute_query(
            f"""
            CREATE (f:Feature:{set_label} {{
                set_id: $set_id,
                category_name: $category_name,
                feature: $feature,
                value: $value,
                tag: $tag,
                embedding: $embedding,
                embedding_dimensions: $dimensions,
                metadata_json: $metadata_json,
                citations: [],
                created_at_ts: $ts,
                updated_at_ts: $ts
            }})
            SET f += $metadata_props
            RETURN elementId(f) AS feature_id
            """,
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            embedding=[float(x) for x in np.array(embedding, dtype=float).tolist()],
            dimensions=dimensions,
            metadata_json=metadata_json,
            metadata_props=metadata_props,
            ts=timestamp,
        )
        if not records:
            raise RuntimeError("Failed to create feature node")
        feature_id = records[0].get("feature_id")
        if feature_id is None:
            raise RuntimeError("Neo4j did not return a feature id")
        return FeatureIdT(str(feature_id))

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
    ) -> None:
        record = await self._get_feature_dimensions(feature_id)
        if record is None:
            return

        existing_set_id = record["set_id"]
        target_set_id = set_id or existing_set_id
        target_dimensions = self._target_dimensions(
            record.get("embedding_dimensions"),
            embedding,
        )
        if target_set_id is None or target_dimensions is None:
            raise ValueError("Unable to resolve embedding dimensions for feature")

        await self._ensure_set_embedding_dimensions(target_set_id, target_dimensions)

        assignments, params, metadata_props = self._build_update_assignments(
            feature_id=feature_id,
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            embedding=embedding,
            metadata=metadata,
            target_dimensions=target_dimensions,
        )
        label_updates = self._build_label_updates(existing_set_id, set_id)

        set_clause = ", ".join(assignments)
        query_parts = ["MATCH (f:Feature)", f"WHERE {self._feature_id_condition()}"]
        query_parts.extend(label_updates)
        query_parts.append(f"SET {set_clause}")
        if metadata_props is not None:
            query_parts.append(
                "WITH f, [key IN keys(f) WHERE key STARTS WITH $metadata_prefix] AS metadata_keys\n"
                "FOREACH (key IN metadata_keys | SET f[key] = NULL)"
            )
            query_parts.append("SET f += $metadata_props")
            params["metadata_props"] = metadata_props
            params["metadata_prefix"] = self._METADATA_PROP_PREFIX
        await self._driver.execute_query("\n".join(query_parts), **params)

    def _target_dimensions(
        self,
        existing_dimensions: int | None,
        embedding: InstanceOf[np.ndarray] | None,
    ) -> int | None:
        if embedding is not None:
            return len(np.array(embedding, dtype=float))
        if existing_dimensions is None:
            return None
        dims = int(existing_dimensions or 0)
        return dims or None

    def _build_update_assignments(
        self,
        *,
        feature_id: FeatureIdT,
        set_id: str | None,
        category_name: str | None,
        feature: str | None,
        value: str | None,
        tag: str | None,
        embedding: InstanceOf[np.ndarray] | None,
        metadata: dict[str, Any] | None,
        target_dimensions: int,
    ) -> tuple[list[str], dict[str, Any], dict[str, Any] | None]:
        assignments = ["f.updated_at_ts = $updated_at_ts"]
        params: dict[str, Any] = {
            "feature_id": str(feature_id),
            "updated_at_ts": _utc_timestamp(),
            "embedding_dimensions": target_dimensions,
        }

        simple_fields = {
            "set_id": set_id,
            "category_name": category_name,
            "feature": feature,
            "value": value,
            "tag": tag,
        }

        for field, value_to_set in simple_fields.items():
            if value_to_set is None:
                continue
            assignments.append(f"f.{field} = ${field}")
            params[field] = value_to_set

        embedding_param = self._embedding_param(embedding)
        if embedding_param is not None:
            assignments.append("f.embedding = $embedding")
            params["embedding"] = embedding_param

        metadata_props: dict[str, Any] | None = None
        if metadata is not None:
            metadata_json, metadata_props = self._prepare_metadata_storage(metadata)
            assignments.append("f.metadata_json = $metadata_json")
            params["metadata_json"] = metadata_json
            params["metadata_props"] = metadata_props

        if set_id is not None or embedding is not None:
            assignments.append("f.embedding_dimensions = $embedding_dimensions")

        return assignments, params, metadata_props

    @staticmethod
    def _embedding_param(
        embedding: InstanceOf[np.ndarray] | None,
    ) -> list[float] | None:
        if embedding is None:
            return None
        return [float(x) for x in np.array(embedding, dtype=float).tolist()]

    def _build_label_updates(
        self,
        existing_set_id: str | None,
        new_set_id: str | None,
    ) -> list[str]:
        if new_set_id is None or new_set_id == existing_set_id:
            return []

        updates: list[str] = []
        if existing_set_id:
            old_label = self._set_label_for_set(existing_set_id)
            updates.append(f"REMOVE f:{old_label}")
        new_label = self._set_label_for_set(new_set_id)
        updates.append(f"SET f:{new_label}")
        return updates

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        records, _, _ = await self._driver.execute_query(
            f"""
            MATCH (f:Feature)
            WHERE {self._feature_id_condition()}
            RETURN f
            """,
            feature_id=str(feature_id),
        )
        if not records:
            return None
        entry = self._node_to_entry(records[0]["f"])
        return self._entry_to_model(entry, load_citations=load_citations)

    async def delete_features(self, feature_ids: list[FeatureIdT]) -> None:
        if not feature_ids:
            return

        await self._driver.execute_query(
            f"""
            MATCH (f:Feature)
            WHERE {self._feature_ids_condition(param="ids")}
            DETACH DELETE f
            """,
            ids=[str(fid) for fid in feature_ids],
        )

    async def get_feature_set(
        self,
        *,
        page_size: int | None = None,
        page_num: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
        filter_expr: FilterExpr | None = None,
    ) -> list[SemanticFeature]:
        if page_num is not None:
            if page_size is None:
                raise InvalidArgumentError("Cannot specify offset without limit")
            if page_num < 0:
                raise InvalidArgumentError("Offset must be non-negative")

        page_offset = page_num or 0
        if vector_search_opts is not None:
            entries = await self._vector_search_entries(
                limit=page_size,
                offset=page_offset,
                vector_search_opts=vector_search_opts,
                filter_expr=filter_expr,
            )
        else:
            entries = await self._load_feature_entries(
                filter_expr=filter_expr,
            )
            entries.sort(key=lambda e: (e.created_at_ts, str(e.feature_id)))
            if page_size is not None:
                start = page_size * page_offset
                entries = entries[start : start + page_size]

        if tag_threshold is not None and entries:
            from collections import Counter

            counts = Counter(entry.tag for entry in entries)
            entries = [entry for entry in entries if counts[entry.tag] >= tag_threshold]

        return [
            self._entry_to_model(entry, load_citations=load_citations)
            for entry in entries
        ]

    async def delete_feature_set(
        self,
        *,
        thresh: int | None = None,
        limit: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        entries = await self.get_feature_set(
            page_size=limit,
            vector_search_opts=vector_search_opts,
            tag_threshold=thresh,
            filter_expr=filter_expr,
            load_citations=False,
        )
        await self.delete_features(
            [FeatureIdT(entry.metadata.id) for entry in entries if entry.metadata.id],
        )

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: list[EpisodeIdT],
    ) -> None:
        if not history_ids:
            return
        records, _, _ = await self._driver.execute_query(
            f"""
            MATCH (f:Feature)
            WHERE {self._feature_id_condition()}
            RETURN f.citations AS citations
            """,
            feature_id=str(feature_id),
        )
        if not records:
            return
        existing: set[str] = set(records[0]["citations"] or [])
        for history_id in history_ids:
            existing.add(str(history_id))
        await self._driver.execute_query(
            f"""
            MATCH (f:Feature)
            WHERE {self._feature_id_condition()}
            SET f.citations = $citations,
                f.updated_at_ts = $ts
            """,
            feature_id=str(feature_id),
            citations=sorted(existing),
            ts=_utc_timestamp(),
        )

    async def get_history_messages(
        self,
        *,
        set_ids: list[str] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> list[EpisodeIdT]:
        query = ["MATCH (h:SetHistory)"]
        conditions = []
        params: dict[str, Any] = {}
        if set_ids is not None:
            conditions.append("h.set_id IN $set_ids")
            params["set_ids"] = set_ids
        if is_ingested is not None:
            conditions.append("h.is_ingested = $is_ingested")
            params["is_ingested"] = is_ingested
        if conditions:
            query.append("WHERE " + " AND ".join(conditions))
        query.append("RETURN h.history_id AS history_id ORDER BY h.history_id")
        if limit is not None:
            query.append("LIMIT $limit")
            params["limit"] = limit
        records, _, _ = await self._driver.execute_query("\n".join(query), **params)
        return [EpisodeIdT(record["history_id"]) for record in records]

    async def get_history_messages_count(
        self,
        *,
        set_ids: list[str] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        query = ["MATCH (h:SetHistory)"]
        conditions = []
        params: dict[str, Any] = {}
        if set_ids is not None:
            conditions.append("h.set_id IN $set_ids")
            params["set_ids"] = set_ids
        if is_ingested is not None:
            conditions.append("h.is_ingested = $is_ingested")
            params["is_ingested"] = is_ingested
        if conditions:
            query.append("WHERE " + " AND ".join(conditions))
        query.append("RETURN count(*) AS cnt")
        records, _, _ = await self._driver.execute_query("\n".join(query), **params)
        return int(records[0]["cnt"]) if records else 0

    async def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
        older_than: datetime | None = None,
    ) -> list[str]:
        set_ids: set[str] = set()
        filters_applied = False

        if min_uningested_messages is not None and min_uningested_messages > 0:
            filters_applied = True
            records, _, _ = await self._driver.execute_query(
                """
                MATCH (h:SetHistory)
                WITH h.set_id AS set_id,
                     sum(CASE WHEN coalesce(h.is_ingested, false) = false THEN 1 ELSE 0 END) AS uningested_count
                WHERE uningested_count >= $min_uningested_messages
                RETURN set_id
                """,
                min_uningested_messages=min_uningested_messages,
            )
            set_ids.update(
                str(record.get("set_id"))
                for record in records
                if record.get("set_id") is not None
            )

        if older_than is not None:
            filters_applied = True
            records, _, _ = await self._driver.execute_query(
                """
                MATCH (h:SetHistory)
                WHERE coalesce(h.is_ingested, false) = false
                  AND h.created_at <= $older_than
                RETURN DISTINCT h.set_id AS set_id
                """,
                older_than=older_than,
            )
            set_ids.update(
                str(record.get("set_id"))
                for record in records
                if record.get("set_id") is not None
            )

        if not filters_applied:
            records, _, _ = await self._driver.execute_query(
                """
                MATCH (h:SetHistory)
                RETURN DISTINCT h.set_id AS set_id
                """,
            )
            set_ids.update(
                str(record.get("set_id"))
                for record in records
                if record.get("set_id") is not None
            )

        return list(set_ids)

    async def add_history_to_set(self, set_id: str, history_id: EpisodeIdT) -> None:
        await self._driver.execute_query(
            """
            MERGE (h:SetHistory {set_id: $set_id, history_id: $history_id})
            ON CREATE SET h.is_ingested = false,
                          h.created_at = $created_at
            """,
            set_id=set_id,
            history_id=str(history_id),
            created_at=datetime.now(UTC),
        )

    async def delete_history(self, history_ids: list[EpisodeIdT]) -> None:
        if not history_ids:
            return

        await self._driver.execute_query(
            """
            MATCH (h:SetHistory)
            WHERE h.history_id IN $history_ids
            DELETE h
            """,
            history_ids=[str(history_id) for history_id in history_ids],
        )

    async def delete_history_set(self, set_ids: list[SetIdT]) -> None:
        if not set_ids:
            return

        await self._driver.execute_query(
            """
            MATCH (h:SetHistory)
            WHERE h.set_id IN $set_ids
            DELETE h
            """,
            set_ids=[str(set_id) for set_id in set_ids],
        )

    async def mark_messages_ingested(
        self,
        *,
        set_id: str,
        history_ids: list[EpisodeIdT],
    ) -> None:
        if not history_ids:
            raise ValueError("No ids provided")
        await self._driver.execute_query(
            """
            MATCH (h:SetHistory)
            WHERE h.set_id = $set_id AND h.history_id IN $history_ids
            SET h.is_ingested = true
            """,
            set_id=set_id,
            history_ids=[str(hid) for hid in history_ids],
        )

    async def _load_feature_entries(
        self,
        *,
        filter_expr: FilterExpr | None,
    ) -> list[_FeatureEntry]:
        query = ["MATCH (f:Feature)"]
        conditions, params = self._build_filter_conditions(
            alias="f",
            filter_expr=filter_expr,
        )
        if conditions:
            query.append("WHERE " + " AND ".join(conditions))
        query.append("RETURN f")
        records, _, _ = await self._driver.execute_query("\n".join(query), **params)
        return [self._node_to_entry(record["f"]) for record in records]

    def _node_to_entry(self, node: Neo4jNode) -> _FeatureEntry:
        props = dict(node)
        node_id = getattr(node, "element_id", None)
        if node_id is None:
            node_id = props.get("id")
        if node_id is None:
            raise ValueError("Feature node missing identifier")
        feature_id = FeatureIdT(str(node_id))
        embedding = np.array(props.get("embedding", []), dtype=float)
        citations = [EpisodeIdT(cid) for cid in props.get("citations", [])]
        metadata = self._parse_metadata(props)
        return _FeatureEntry(
            feature_id=feature_id,
            set_id=_required_str_prop(props, "set_id"),
            category_name=_required_str_prop(props, "category_name"),
            tag=_required_str_prop(props, "tag"),
            feature_name=_required_str_prop(props, "feature"),
            value=_required_str_prop(props, "value"),
            embedding=embedding,
            metadata=metadata,
            citations=citations,
            created_at_ts=float(props.get("created_at_ts", 0.0)),
            updated_at_ts=float(props.get("updated_at_ts", 0.0)),
        )

    @staticmethod
    def _parse_metadata(props: Mapping[str, Any]) -> dict[str, Any] | None:
        metadata_json = props.get("metadata_json")
        if isinstance(metadata_json, str) and metadata_json:
            try:
                return json.loads(metadata_json)
            except json.JSONDecodeError:
                return None
        legacy = props.get("metadata")
        if isinstance(legacy, Mapping):
            return dict(legacy)
        return None

    def _metadata_property_name(self, key: str) -> str:
        sanitized = _sanitize_identifier(key)
        return f"{self._METADATA_PROP_PREFIX}{sanitized}"

    def _prepare_metadata_storage(
        self,
        metadata: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any]]:
        if metadata is None:
            return None, {}
        metadata_json = json.dumps(metadata)
        metadata_props: dict[str, Any] = {}
        for raw_key, raw_value in metadata.items():
            if raw_value is None:
                continue
            prop_name = self._metadata_property_name(raw_key)
            metadata_props[prop_name] = self._normalize_metadata_property_value(
                raw_value,
            )
        return metadata_json, metadata_props

    @staticmethod
    def _normalize_metadata_property_value(
        value: FilterablePropertyValue,
    ) -> bool | int | float | str:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        raise TypeError(f"Unsupported metadata value type: {type(value)!r}")

    def _entry_to_model(
        self,
        entry: _FeatureEntry,
        *,
        load_citations: bool,
    ) -> SemanticFeature:
        citations: list[EpisodeIdT] | None = None
        if load_citations:
            citations = list(entry.citations)
        return SemanticFeature(
            set_id=entry.set_id,
            category=entry.category_name,
            tag=entry.tag,
            feature_name=entry.feature_name,
            value=entry.value,
            metadata=SemanticFeature.Metadata(
                id=entry.feature_id,
                citations=citations,
                other=dict(entry.metadata) if entry.metadata else None,
            ),
        )

    async def _vector_search_entries(
        self,
        *,
        limit: int | None,
        offset: int,
        vector_search_opts: SemanticStorage.VectorSearchOpts,
        filter_expr: FilterExpr | None,
    ) -> list[_FeatureEntry]:
        embedding_array = np.array(vector_search_opts.query_embedding, dtype=float)
        embedding = [float(x) for x in embedding_array.tolist()]
        embedding_dims = len(embedding)

        effective_limit = limit
        if limit is not None:
            effective_limit = limit * (offset + 1)

        conditions, filter_params = self._build_filter_conditions(
            alias="f",
            filter_expr=filter_expr,
        )

        params_base = self._vector_query_params(
            embedding=embedding,
            filter_params=filter_params,
            candidate_limit=max(
                effective_limit or 0,
                self._DEFAULT_VECTOR_QUERY_CANDIDATES,
            ),
            min_distance=vector_search_opts.min_distance,
            conditions=conditions,
        )
        query_text = self._vector_query_text(conditions)

        combined: list[tuple[float, _FeatureEntry]] = []
        for set_id in self._matching_set_ids(filter_expr, embedding_dims):
            index_name = await self._ensure_vector_index(set_id, embedding_dims)
            combined.extend(
                await self._query_vector_index(query_text, index_name, params_base),
            )

        combined.sort(key=lambda item: item[0], reverse=True)
        entries = [entry for _, entry in combined]
        if limit is not None:
            start = limit * offset
            entries = entries[start : start + limit]
        return entries

    def _vector_query_params(
        self,
        *,
        embedding: list[float],
        filter_params: dict[str, Any],
        candidate_limit: int,
        min_distance: float | None,
        conditions: list[str],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "candidate_limit": candidate_limit,
            "embedding": embedding,
            **filter_params,
        }
        if min_distance is not None and min_distance > 0.0:
            conditions.append("score >= $min_distance")
            params["min_distance"] = min_distance
        return params

    @staticmethod
    def _vector_query_text(conditions: list[str]) -> str:
        query_parts = [
            "CALL db.index.vector.queryNodes($index_name, $candidate_limit, $embedding)",
            "YIELD node, score",
            "WITH node AS f, score",
        ]
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        query_parts.append("RETURN f AS node, score ORDER BY score DESC")
        return "\n".join(query_parts)

    def _matching_set_ids(
        self,
        filter_expr: FilterExpr | None,
        expected_dims: int,
    ) -> list[str]:
        requested_set_ids = self._extract_set_ids(filter_expr)
        candidate_ids = self._deduplicated_set_ids(requested_set_ids)
        return [
            set_id
            for set_id in candidate_ids
            if self._set_embedding_dimensions.get(set_id) == expected_dims
        ]

    def _deduplicated_set_ids(self, requested_set_ids: list[str] | None) -> list[str]:
        if requested_set_ids is None:
            return list(self._set_embedding_dimensions.keys())

        seen: set[str] = set()
        ordered_set_ids: list[str] = []
        for sid in requested_set_ids:
            if sid in seen:
                continue
            seen.add(sid)
            ordered_set_ids.append(sid)
        return ordered_set_ids

    async def _query_vector_index(
        self,
        query_text: str,
        index_name: str,
        params_base: dict[str, Any],
    ) -> list[tuple[float, _FeatureEntry]]:
        params = dict(params_base)
        params["index_name"] = index_name
        records, _, _ = await self._driver.execute_query(query_text, **params)
        return [
            (float(record.get("score") or 0.0), self._node_to_entry(record["node"]))
            for record in records
        ]

    def _build_filter_conditions(
        self,
        *,
        alias: str,
        filter_expr: FilterExpr | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if filter_expr is not None:
            expr_condition, expr_params = self._render_filter_expr(alias, filter_expr)
            if expr_condition:
                conditions.append(expr_condition)
                params.update(expr_params)
        return conditions, params

    def _extract_set_ids(self, expr: FilterExpr | None) -> list[str] | None:
        if expr is None:
            return None

        values = self._collect_field_values(expr, target_field="set_id")
        if values is None:
            return None
        return [str(v) for v in values]

    def _collect_field_values(
        self,
        expr: FilterExpr,
        *,
        target_field: str,
    ) -> set[str] | None:
        if isinstance(expr, FilterComparison):
            return self._collect_comparison_values(expr, target_field)
        if isinstance(expr, FilterAnd):
            return self._merge_and_values(expr, target_field)
        if isinstance(expr, FilterOr):
            # Ambiguous which branch applies; fall back to no restriction
            return None
        return None

    def _collect_comparison_values(
        self, expr: FilterComparison, target_field: str
    ) -> set[str] | None:
        if expr.field != target_field:
            return None
        if expr.op == "=":
            if isinstance(expr.value, list):
                raise ValueError("'=' comparison cannot accept list values")
            return {str(expr.value)}
        if expr.op == "in":
            if not isinstance(expr.value, list):
                raise ValueError("IN comparison requires list values")
            return {str(v) for v in expr.value}
        return None

    def _merge_and_values(self, expr: FilterAnd, target_field: str) -> set[str] | None:
        left_vals = self._collect_field_values(expr.left, target_field=target_field)
        right_vals = self._collect_field_values(expr.right, target_field=target_field)
        if left_vals is None:
            return right_vals
        if right_vals is None:
            return left_vals
        return left_vals & right_vals

    def _render_filter_expr(
        self,
        alias: str,
        expr: FilterExpr,
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(expr, FilterComparison):
            field_ref = self._resolve_field_reference(alias, expr.field)
            return self._render_comparison_condition(field_ref, expr)
        if isinstance(expr, FilterAnd):
            left_cond, left_params = self._render_filter_expr(alias, expr.left)
            right_cond, right_params = self._render_filter_expr(alias, expr.right)
            condition = f"({left_cond}) AND ({right_cond})"
            left_params.update(right_params)
            return condition, left_params
        if isinstance(expr, FilterOr):
            left_cond, left_params = self._render_filter_expr(alias, expr.left)
            right_cond, right_params = self._render_filter_expr(alias, expr.right)
            condition = f"({left_cond}) OR ({right_cond})"
            left_params.update(right_params)
            return condition, left_params
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    def _render_comparison_condition(
        self, field_ref: str, expr: FilterComparison
    ) -> tuple[str, dict[str, Any]]:
        op = expr.op
        params: dict[str, Any] = {}

        if op == "in":
            if not isinstance(expr.value, list):
                raise ValueError("IN comparison requires a list of values")
            param = self._next_filter_param()
            return f"{field_ref} IN ${param}", {param: expr.value}

        if op in (">", "<", ">=", "<=", "="):
            if isinstance(expr.value, list):
                raise ValueError(f"'{op}' comparison cannot accept list values")
            param = self._next_filter_param()
            return f"{field_ref} {op} ${param}", {param: expr.value}

        if op == "is_null":
            return f"{field_ref} IS NULL", params

        if op == "is_not_null":
            return f"{field_ref} IS NOT NULL", params

        raise ValueError(f"Unsupported operator: {op}")

    def _resolve_field_reference(self, alias: str, field: str) -> str:
        if field.startswith(("m.", "metadata.")):
            key = field.split(".", 1)[1]
            prop_name = self._metadata_property_name(key)
            return f"{alias}.{prop_name}"
        return f"{alias}.{field}"

    def _next_filter_param(self) -> str:
        self._filter_param_counter += 1
        return f"filter_param_{self._filter_param_counter}"

    async def _hydrate_vector_index_state(self) -> None:
        self._vector_index_by_set.clear()
        records, _, _ = await self._driver.execute_query(
            """
            SHOW VECTOR INDEXES
            YIELD name, options, labelsOrTypes
            WHERE name STARTS WITH $prefix
            RETURN name, options, labelsOrTypes
            """,
            prefix=self._VECTOR_INDEX_PREFIX,
        )
        for record in records:
            name = record.get("name")
            set_id = self._set_id_from_record(record)
            dimensions = self._dimensions_from_record(record)

            if set_id is None:
                await self._drop_index_if_named(name)
                continue

            if dimensions is not None:
                self._vector_index_by_set[set_id] = dimensions

    async def _drop_index_if_named(self, name: str | None) -> None:
        if not name:
            return
        await self._driver.execute_query(f"DROP INDEX {name} IF EXISTS")

    def _set_id_from_record(self, record: Mapping[str, Any]) -> str | None:
        labels = record.get("labelsOrTypes") or []
        for label in labels or []:
            set_id = self._set_id_from_label(label)
            if set_id is not None:
                return set_id
        return None

    @staticmethod
    def _dimensions_from_record(record: Mapping[str, Any]) -> int | None:
        options = record.get("options") or {}
        config = options.get("indexConfig") or {}
        dimensions = config.get("vector.dimensions")
        if isinstance(dimensions, (int, float)):
            return int(dimensions)
        return None

    async def _create_vector_index(
        self, index_name: str, set_id: str, dimensions: int
    ) -> int:
        if index_name not in self._vector_index_creation_lock:
            async with self._vector_global_lock:
                self._vector_index_creation_lock.setdefault(index_name, Lock())

        async with self._vector_index_creation_lock[index_name]:
            val = self._vector_index_by_set.get(set_id)
            if val is not None:
                return val

            label = self._set_label_for_set(set_id)
            await self._driver.execute_query(
                f"""
                CREATE VECTOR INDEX {index_name}
                IF NOT EXISTS
                FOR (f:{label})
                ON (f.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dimensions=dimensions,
            )
            await self._driver.execute_query("CALL db.awaitIndexes()")

            records, _, _ = await self._driver.execute_query(
                """
                SHOW VECTOR INDEXES
                YIELD name, type, labelsOrTypes, properties, options
                WHERE name = $index_name
                  AND properties = ['embedding']
                RETURN options
                """,
                index_name=index_name,
            )

            if not records:
                raise ValueError("Index not found after creation")

            index_config = records[0]["options"].get("indexConfig", {})
            actual_dim = index_config.get("vector.dimensions")

            if not actual_dim:
                raise ValueError("Index dim not found after creation")

            self._vector_index_by_set.setdefault(set_id, actual_dim)
            return self._vector_index_by_set[set_id]

    async def _ensure_vector_index(self, set_id: str, dimensions: int) -> str:
        cached = self._vector_index_by_set.get(set_id)
        index_name = self._vector_index_name(set_id)

        if cached is None:
            cached = await self._create_vector_index(index_name, set_id, dimensions)

        if cached != dimensions:
            raise ValueError(
                "Embedding dimension mismatch for set_id "
                f"{set_id}: expected {cached}, got {dimensions}",
            )
        return index_name

    def _vector_index_name(self, set_id: str) -> str:
        sanitized = _sanitize_identifier(set_id)
        return f"{self._VECTOR_INDEX_PREFIX}_{sanitized}"

    async def _backfill_embedding_dimensions(self) -> None:
        await self._driver.execute_query(
            """
            MATCH (f:Feature)
            WHERE f.embedding IS NOT NULL AND f.embedding_dimensions IS NULL
            WITH f, size(f.embedding) AS dims
            SET f.embedding_dimensions = dims
            """,
        )
        await self._driver.execute_query(
            """
            MATCH (f:Feature)
            WHERE f.embedding_dimensions IS NOT NULL AND f.set_id IS NOT NULL
            MERGE (s:SetEmbedding {set_id: f.set_id})
            ON CREATE SET s.dimensions = f.embedding_dimensions
            """,
        )

    async def _load_set_embedding_dimensions(self) -> None:
        self._set_embedding_dimensions.clear()
        records, _, _ = await self._driver.execute_query(
            "MATCH (s:SetEmbedding) RETURN s.set_id AS set_id, s.dimensions AS dims",
        )
        for record in records:
            set_id = record.get("set_id")
            dims = record.get("dims")
            if set_id is None or dims is None:
                continue
            self._set_embedding_dimensions[str(set_id)] = int(dims)

    async def _ensure_existing_set_labels(self) -> None:
        if not self._set_embedding_dimensions:
            return
        for set_id in list(self._set_embedding_dimensions.keys()):
            await self._ensure_set_label_applied(set_id)

    async def _ensure_set_embedding_dimensions(
        self,
        set_id: str,
        dimensions: int,
    ) -> None:
        cached = self._set_embedding_dimensions.get(set_id)
        if cached is not None:
            if cached != dimensions:
                raise ValueError(
                    "Embedding dimension mismatch for set_id "
                    f"{set_id}: expected {cached}, got {dimensions}",
                )
            await self._ensure_vector_index(set_id, dimensions)
            return

        records, _, _ = await self._driver.execute_query(
            """
            MERGE (s:SetEmbedding {set_id: $set_id})
            ON CREATE SET s.dimensions = $dimensions
            RETURN s.dimensions AS dims
            """,
            set_id=set_id,
            dimensions=dimensions,
        )
        db_dims = records[0]["dims"] if records else None
        if db_dims is None:
            db_dims = dimensions
        db_dims = int(db_dims)
        if db_dims != dimensions:
            raise ValueError(
                "Embedding dimension mismatch for set_id "
                f"{set_id}: expected {db_dims}, got {dimensions}",
            )
        self._set_embedding_dimensions[set_id] = db_dims
        await self._ensure_set_label_applied(set_id)
        await self._ensure_vector_index(set_id, db_dims)

    async def _ensure_set_label_applied(self, set_id: str) -> None:
        label = self._set_label_for_set(set_id)
        await self._driver.execute_query(
            f"""
            MATCH (f:Feature {{ set_id: $set_id }})
            WHERE NOT f:{label}
            SET f:{label}
            """,
            set_id=set_id,
        )

    def _set_label_for_set(self, set_id: str) -> str:
        return f"{self._SET_LABEL_PREFIX}{_sanitize_identifier(set_id)}"

    def _set_id_from_label(self, label: str) -> str | None:
        if not label or not label.startswith(self._SET_LABEL_PREFIX):
            return None
        suffix = label[len(self._SET_LABEL_PREFIX) :]
        return _desanitize_identifier(suffix)

    @staticmethod
    def _feature_id_condition(alias: str = "f", param: str = "feature_id") -> str:
        return (
            f"(elementId({alias}) = ${param} "
            f"OR ({alias}.id IS NOT NULL AND toString({alias}.id) = ${param}))"
        )

    @staticmethod
    def _feature_ids_condition(alias: str = "f", param: str = "feature_ids") -> str:
        return (
            f"(elementId({alias}) IN ${param} "
            f"OR ({alias}.id IS NOT NULL AND toString({alias}.id) IN ${param}))"
        )

    async def _get_feature_dimensions(
        self,
        feature_id: FeatureIdT,
    ) -> dict[str, Any] | None:
        records, _, _ = await self._driver.execute_query(
            f"""
            MATCH (f:Feature)
            WHERE {self._feature_id_condition()}
            RETURN f.set_id AS set_id, f.embedding_dimensions AS embedding_dimensions,
                   CASE WHEN f.embedding_dimensions IS NULL AND f.embedding IS NOT NULL
                        THEN size(f.embedding)
                        ELSE f.embedding_dimensions END AS resolved_dimensions
            """,
            feature_id=str(feature_id),
        )
        if not records:
            return None
        record = dict(records[0])
        if (
            record.get("embedding_dimensions") is None
            and record.get("resolved_dimensions") is not None
        ):
            record["embedding_dimensions"] = record["resolved_dimensions"]
        return record
