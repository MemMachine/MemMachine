"""Neo4j-backed implementation of :class:`SemanticStorageBase`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from neo4j import AsyncDriver
from neo4j.graph import Node as Neo4jNode
from pydantic import InstanceOf, validate_call

from memmachine.episode_store.episode_model import EpisodeIdT
from memmachine.semantic_memory.semantic_model import SemanticFeature
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
                citations: [],
                created_at_ts: $ts,
                updated_at_ts: $ts
            }})
            RETURN elementId(f) AS feature_id
            """,
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            embedding=[float(x) for x in np.array(embedding, dtype=float).tolist()],
            dimensions=dimensions,
            metadata=dict(metadata or {}) or None,
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

        assignments, params = self._build_update_assignments(
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
    ) -> tuple[list[str], dict[str, Any]]:
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

        if metadata is not None:
            assignments.append("f.metadata = $metadata")
            params["metadata"] = dict(metadata) or None

        if set_id is not None or embedding is not None:
            assignments.append("f.embedding_dimensions = $embedding_dimensions")

        return assignments, params

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

    @validate_call
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

    @validate_call
    async def get_feature_set(
        self,
        *,
        set_ids: list[str] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        if vector_search_opts is not None:
            entries = await self._vector_search_entries(
                set_ids=set_ids,
                category_names=category_names,
                feature_names=feature_names,
                tags=tags,
                limit=limit,
                vector_search_opts=vector_search_opts,
            )
        else:
            entries = await self._load_feature_entries(
                set_ids=set_ids,
                category_names=category_names,
                feature_names=feature_names,
                tags=tags,
            )
            entries.sort(key=lambda e: (e.created_at_ts, str(e.feature_id)))
            if limit is not None:
                entries = entries[:limit]

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
        set_ids: list[str] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        thresh: int | None = None,
        limit: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
    ) -> None:
        entries = await self.get_feature_set(
            set_ids=set_ids,
            category_names=category_names,
            feature_names=feature_names,
            tags=tags,
            limit=limit,
            vector_search_opts=vector_search_opts,
            tag_threshold=thresh,
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
    ) -> list[str]:
        if min_uningested_messages is None or min_uningested_messages <= 0:
            records, _, _ = await self._driver.execute_query(
                """
                MATCH (h:SetHistory)
                RETURN DISTINCT h.set_id AS set_id
                """,
            )
        else:
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

        return [
            str(record.get("set_id"))
            for record in records
            if record.get("set_id") is not None
        ]

    async def add_history_to_set(self, set_id: str, history_id: EpisodeIdT) -> None:
        await self._driver.execute_query(
            """
            MERGE (h:SetHistory {set_id: $set_id, history_id: $history_id})
            ON CREATE SET h.is_ingested = false
            """,
            set_id=set_id,
            history_id=str(history_id),
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
        set_ids: list[str] | None,
        category_names: list[str] | None,
        feature_names: list[str] | None,
        tags: list[str] | None,
    ) -> list[_FeatureEntry]:
        query = ["MATCH (f:Feature)"]
        conditions, params = self._build_filter_conditions(
            alias="f",
            set_ids=set_ids,
            category_names=category_names,
            feature_names=feature_names,
            tags=tags,
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
        return _FeatureEntry(
            feature_id=feature_id,
            set_id=props.get("set_id"),
            category_name=props.get("category_name"),
            tag=props.get("tag"),
            feature_name=props.get("feature"),
            value=props.get("value"),
            embedding=embedding,
            metadata=props.get("metadata") or None,
            citations=citations,
            created_at_ts=float(props.get("created_at_ts", 0.0)),
            updated_at_ts=float(props.get("updated_at_ts", 0.0)),
        )

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
        set_ids: list[str] | None,
        category_names: list[str] | None,
        feature_names: list[str] | None,
        tags: list[str] | None,
        limit: int | None,
        vector_search_opts: SemanticStorage.VectorSearchOpts,
    ) -> list[_FeatureEntry]:
        embedding_array = np.array(vector_search_opts.query_embedding, dtype=float)
        embedding = [float(x) for x in embedding_array.tolist()]
        embedding_dims = len(embedding)

        conditions, filter_params = self._build_filter_conditions(
            alias="f",
            set_ids=set_ids,
            category_names=category_names,
            feature_names=feature_names,
            tags=tags,
        )

        params_base = self._vector_query_params(
            embedding=embedding,
            filter_params=filter_params,
            candidate_limit=max(limit or 0, self._DEFAULT_VECTOR_QUERY_CANDIDATES),
            min_distance=vector_search_opts.min_distance,
            conditions=conditions,
        )
        query_text = self._vector_query_text(conditions)

        combined: list[tuple[float, _FeatureEntry]] = []
        for set_id in self._matching_set_ids(set_ids, embedding_dims):
            index_name = await self._ensure_vector_index(set_id, embedding_dims)
            combined.extend(
                await self._query_vector_index(query_text, index_name, params_base),
            )

        combined.sort(key=lambda item: item[0], reverse=True)
        entries = [entry for _, entry in combined]
        if limit is not None:
            entries = entries[:limit]
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
        requested_set_ids: list[str] | None,
        expected_dims: int,
    ) -> list[str]:
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
        set_ids: list[str] | None,
        category_names: list[str] | None,
        feature_names: list[str] | None,
        tags: list[str] | None,
    ) -> tuple[list[str], dict[str, Any]]:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if set_ids is not None:
            conditions.append(f"{alias}.set_id IN $set_ids")
            params["set_ids"] = set_ids
        if category_names is not None:
            conditions.append(f"{alias}.category_name IN $category_names")
            params["category_names"] = category_names
        if feature_names is not None:
            conditions.append(f"{alias}.feature IN $feature_names")
            params["feature_names"] = feature_names
        if tags is not None:
            conditions.append(f"{alias}.tag IN $tags")
            params["tags"] = tags
        return conditions, params

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

    def _set_id_from_record(self, record: dict[str, Any]) -> str | None:
        labels = record.get("labelsOrTypes") or []
        for label in labels or []:
            set_id = self._set_id_from_label(label)
            if set_id is not None:
                return set_id
        return None

    @staticmethod
    def _dimensions_from_record(record: dict[str, Any]) -> int | None:
        options = record.get("options") or {}
        config = options.get("indexConfig") or {}
        dimensions = config.get("vector.dimensions")
        if isinstance(dimensions, (int, float)):
            return int(dimensions)
        return None

    async def _ensure_vector_index(self, set_id: str, dimensions: int) -> str:
        cached = self._vector_index_by_set.get(set_id)
        index_name = self._vector_index_name(set_id)
        if cached is not None:
            if cached != dimensions:
                raise ValueError(
                    "Embedding dimension mismatch for set_id "
                    f"{set_id}: expected {cached}, got {dimensions}",
                )
            return index_name

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
        self._vector_index_by_set[set_id] = dimensions
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
