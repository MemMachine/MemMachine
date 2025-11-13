"""Neo4j-backed implementation of :class:`SemanticStorageBase`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from neo4j import AsyncDriver
from pydantic import InstanceOf, validate_call

from memmachine.history_store.history_model import HistoryIdT
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.semantic_memory.storage.storage_base import (
    FeatureIdT,
    SemanticStorageBase,
)


def _utc_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


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
    citations: list[HistoryIdT]
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
    result: list[str] = []
    i = 0
    length = len(value)
    while i < length:
        if value[i : i + 2] == "_u":
            end = value.find("_", i + 2)
            if end == -1:
                result.append(value[i])
                i += 1
                continue
            hex_part = value[i + 2 : end]
            try:
                result.append(chr(int(hex_part, 16)))
                i = end + 1
                continue
            except ValueError:
                result.append(value[i])
                i += 1
                continue
        result.append(value[i])
        i += 1
    return "".join(result)


class Neo4jSemanticStorage(SemanticStorageBase):
    """Concrete :class:`SemanticStorageBase` backed by Neo4j."""

    _FEATURE_COUNTER = "feature"
    _VECTOR_INDEX_PREFIX = "feature_embedding_index"
    _DEFAULT_VECTOR_QUERY_CANDIDATES = 100
    _SET_LABEL_PREFIX = "FeatureSet_"

    def __init__(self, driver: InstanceOf[AsyncDriver], owns_driver: bool = False):
        self._driver = driver
        self._owns_driver = owns_driver
        # Exposed for fixtures to know which backend is in use
        self.backend_name = "neo4j"
        self._vector_index_by_set: dict[str, int] = {}
        self._set_embedding_dimensions: dict[str, int] = {}

    async def startup(self):
        await self._driver.execute_query(
            """
            CREATE CONSTRAINT feature_id_unique IF NOT EXISTS
            FOR (f:Feature)
            REQUIRE f.id IS UNIQUE
            """
        )
        await self._driver.execute_query(
            """
            CREATE CONSTRAINT set_history_unique IF NOT EXISTS
            FOR (h:SetHistory)
            REQUIRE (h.set_id, h.history_id) IS UNIQUE
            """
        )
        await self._driver.execute_query(
            """
            CREATE CONSTRAINT set_embedding_unique IF NOT EXISTS
            FOR (s:SetEmbedding)
            REQUIRE s.set_id IS UNIQUE
            """
        )
        await self._backfill_embedding_dimensions()
        await self._load_set_embedding_dimensions()
        await self._ensure_existing_set_labels()
        await self._hydrate_vector_index_state()

    async def cleanup(self):
        if self._owns_driver:
            await self._driver.close()

    async def delete_all(self):
        await self._driver.execute_query("MATCH (f:Feature) DETACH DELETE f")
        await self._driver.execute_query("MATCH (h:SetHistory) DELETE h")
        await self._driver.execute_query("MATCH (s:SetEmbedding) DELETE s")
        await self._driver.execute_query(
            "MATCH (c:SemanticCounter) WHERE c.name = $name DELETE c",
            name=self._FEATURE_COUNTER,
        )
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
            await self._driver.execute_query(
                f"DROP INDEX {index_name} IF EXISTS"
            )
        self._vector_index_by_set.clear()
        self._set_embedding_dimensions.clear()

    async def _next_feature_id(self) -> FeatureIdT:
        records, _, _ = await self._driver.execute_query(
            """
            MERGE (counter:SemanticCounter {name: $name})
            ON CREATE SET counter.value = 0
            SET counter.value = counter.value + 1
            RETURN counter.value AS next_id
            """,
            name=self._FEATURE_COUNTER,
        )
        next_id = int(records[0]["next_id"])
        return FeatureIdT(str(next_id))

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
        feature_id = await self._next_feature_id()

        timestamp = _utc_timestamp()
        dimensions = int(len(np.array(embedding, dtype=float)))

        await self._ensure_set_embedding_dimensions(set_id, dimensions)

        set_label = self._set_label_for_set(set_id)

        await self._driver.execute_query(
            f"""
            CREATE (f:Feature:{set_label} {{
                id: toInteger($feature_id),
                set_id: $set_id,
                category_name: $category_name,
                feature: $feature,
                value: $value,
                tag: $tag,
                embedding: $embedding,
                embedding_dimensions: $dimensions,
                metadata: $metadata,
                citations: [],
                created_at_ts: $ts,
                updated_at_ts: $ts
            }})
            """,
            feature_id=int(feature_id),
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
        record = await self._get_feature_dimensions(feature_id)
        if record is None:
            return

        existing_set_id = record["set_id"]
        existing_dimensions = int(record.get("embedding_dimensions") or 0)
        target_set_id = set_id or existing_set_id
        target_dimensions = existing_dimensions
        if embedding is not None:
            target_dimensions = int(len(np.array(embedding, dtype=float)))
        if target_set_id is None or target_dimensions == 0:
            raise ValueError("Unable to resolve embedding dimensions for feature")

        await self._ensure_set_embedding_dimensions(target_set_id, target_dimensions)

        assignments: list[str] = ["f.updated_at_ts = $updated_at_ts"]
        params: dict[str, Any] = {
            "feature_id": int(feature_id),
            "updated_at_ts": _utc_timestamp(),
            "embedding_dimensions": target_dimensions,
        }

        if set_id is not None:
            assignments.append("f.set_id = $set_id")
            params["set_id"] = set_id

        if category_name is not None:
            assignments.append("f.category_name = $category_name")
            params["category_name"] = category_name

        if feature is not None:
            assignments.append("f.feature = $feature")
            params["feature"] = feature

        if value is not None:
            assignments.append("f.value = $value")
            params["value"] = value

        if tag is not None:
            assignments.append("f.tag = $tag")
            params["tag"] = tag

        if embedding is not None:
            assignments.append("f.embedding = $embedding")
            params["embedding"] = [
                float(x) for x in np.array(embedding, dtype=float).tolist()
            ]

        if set_id is not None or embedding is not None:
            assignments.append("f.embedding_dimensions = $embedding_dimensions")

        if metadata is not None:
            assignments.append("f.metadata = $metadata")
            params["metadata"] = dict(metadata) or None

        label_updates: list[str] = []

        if set_id is not None and set_id != existing_set_id:
            if existing_set_id:
                old_label = self._set_label_for_set(existing_set_id)
                label_updates.append(f"REMOVE f:{old_label}")
            new_label = self._set_label_for_set(set_id)
            label_updates.append(f"SET f:{new_label}")

        set_clause = ", ".join(assignments)
        query_parts = ["MATCH (f:Feature {id: $feature_id})"]
        query_parts.extend(label_updates)
        query_parts.append(f"SET {set_clause}")
        await self._driver.execute_query("\n".join(query_parts), **params)

    @validate_call
    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        records, _, _ = await self._driver.execute_query(
            "MATCH (f:Feature {id: $feature_id}) RETURN f",
            feature_id=int(feature_id),
        )
        if not records:
            return None
        entry = self._node_to_entry(records[0]["f"])
        return self._entry_to_model(entry, load_citations=load_citations)

    async def delete_features(self, feature_ids: list[FeatureIdT]):
        if not feature_ids:
            return

        await self._driver.execute_query(
            "MATCH (f:Feature) WHERE f.id IN $ids DETACH DELETE f",
            ids=[int(fid) for fid in feature_ids],
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
        vector_search_opts: SemanticStorageBase.VectorSearchOpts | None = None,
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
            entries.sort(key=lambda e: (e.created_at_ts, int(e.feature_id)))
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
        vector_search_opts: SemanticStorageBase.VectorSearchOpts | None = None,
    ):
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
            [FeatureIdT(entry.metadata.id) for entry in entries if entry.metadata.id]
        )

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: list[HistoryIdT],
    ):
        if not history_ids:
            return
        records, _, _ = await self._driver.execute_query(
            "MATCH (f:Feature {id: $feature_id}) RETURN f.citations AS citations",
            feature_id=int(feature_id),
        )
        if not records:
            return
        existing: set[str] = set(records[0]["citations"] or [])
        for history_id in history_ids:
            existing.add(str(history_id))
        await self._driver.execute_query(
            """
            MATCH (f:Feature {id: $feature_id})
            SET f.citations = $citations,
                f.updated_at_ts = $ts
            """,
            feature_id=int(feature_id),
            citations=sorted(existing),
            ts=_utc_timestamp(),
        )

    async def get_history_messages(
        self,
        *,
        set_ids: list[str] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> list[HistoryIdT]:
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
        return [HistoryIdT(record["history_id"]) for record in records]

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

    async def add_history_to_set(self, set_id: str, history_id: HistoryIdT) -> None:
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
        history_ids: list[HistoryIdT],
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

    def _node_to_entry(self, node) -> _FeatureEntry:
        props = dict(node)
        feature_id = FeatureIdT(str(int(props["id"])))
        embedding = np.array(props.get("embedding", []), dtype=float)
        citations = [HistoryIdT(cid) for cid in props.get("citations", [])]
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
        citations: list[HistoryIdT] | None = None
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
        vector_search_opts: SemanticStorageBase.VectorSearchOpts,
    ) -> list[_FeatureEntry]:
        embedding_array = np.array(vector_search_opts.query_embedding, dtype=float)
        embedding = [float(x) for x in embedding_array.tolist()]
        embedding_dims = len(embedding)

        candidate_limit = max(limit or 0, self._DEFAULT_VECTOR_QUERY_CANDIDATES)

        conditions, filter_params = self._build_filter_conditions(
            alias="f",
            set_ids=set_ids,
            category_names=category_names,
            feature_names=feature_names,
            tags=tags,
        )

        params_base: dict[str, Any] = {
            "candidate_limit": candidate_limit,
            "embedding": embedding,
            **filter_params,
        }

        if vector_search_opts.min_distance is not None and vector_search_opts.min_distance > 0.0:
            conditions.append("score >= $min_distance")
            params_base["min_distance"] = vector_search_opts.min_distance

        query_parts = [
            "CALL db.index.vector.queryNodes($index_name, $candidate_limit, $embedding)",
            "YIELD node, score",
            "WITH node AS f, score",
        ]
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        query_parts.append("RETURN f AS node, score ORDER BY score DESC")
        query_text = "\n".join(query_parts)

        if set_ids is not None:
            ordered_set_ids: list[str] = []
            seen: set[str] = set()
            for sid in set_ids:
                if sid in seen:
                    continue
                seen.add(sid)
                ordered_set_ids.append(sid)
        else:
            ordered_set_ids = list(self._set_embedding_dimensions.keys())

        combined: list[tuple[float, _FeatureEntry]] = []
        for set_id in ordered_set_ids:
            dims = self._set_embedding_dimensions.get(set_id)
            if dims is None or dims != embedding_dims:
                continue
            index_name = await self._ensure_vector_index(set_id, dims)
            params = dict(params_base)
            params["index_name"] = index_name
            records, _, _ = await self._driver.execute_query(query_text, **params)
            for record in records:
                score = float(record.get("score") or 0.0)
                combined.append((score, self._node_to_entry(record["node"])))

        combined.sort(key=lambda item: item[0], reverse=True)
        entries = [entry for _, entry in combined]
        if limit is not None:
            entries = entries[:limit]
        return entries

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

    async def _hydrate_vector_index_state(self):
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
            options = record.get("options") or {}
            config = options.get("indexConfig") or {}
            dimensions = config.get("vector.dimensions")
            name = record.get("name")
            labels = record.get("labelsOrTypes") or []
            set_id = None
            for label in labels or []:
                set_id = self._set_id_from_label(label)
                if set_id is not None:
                    break
            if set_id is None:
                if name:
                    await self._driver.execute_query(
                        f"DROP INDEX {name} IF EXISTS"
                    )
                continue
            if isinstance(dimensions, (int, float)):
                self._vector_index_by_set[set_id] = int(dimensions)

    async def _ensure_vector_index(self, set_id: str, dimensions: int) -> str:
        cached = self._vector_index_by_set.get(set_id)
        index_name = self._vector_index_name(set_id)
        if cached is not None:
            if cached != dimensions:
                raise ValueError(
                    "Embedding dimension mismatch for set_id "
                    f"{set_id}: expected {cached}, got {dimensions}"
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

    async def _backfill_embedding_dimensions(self):
        await self._driver.execute_query(
            """
            MATCH (f:Feature)
            WHERE f.embedding IS NOT NULL AND f.embedding_dimensions IS NULL
            WITH f, size(f.embedding) AS dims
            SET f.embedding_dimensions = dims
            """
        )
        await self._driver.execute_query(
            """
            MATCH (f:Feature)
            WHERE f.embedding_dimensions IS NOT NULL AND f.set_id IS NOT NULL
            MERGE (s:SetEmbedding {set_id: f.set_id})
            ON CREATE SET s.dimensions = f.embedding_dimensions
            """
        )

    async def _load_set_embedding_dimensions(self):
        self._set_embedding_dimensions.clear()
        records, _, _ = await self._driver.execute_query(
            "MATCH (s:SetEmbedding) RETURN s.set_id AS set_id, s.dimensions AS dims"
        )
        for record in records:
            set_id = record.get("set_id")
            dims = record.get("dims")
            if set_id is None or dims is None:
                continue
            self._set_embedding_dimensions[str(set_id)] = int(dims)

    async def _ensure_existing_set_labels(self):
        if not self._set_embedding_dimensions:
            return
        for set_id in list(self._set_embedding_dimensions.keys()):
            await self._ensure_set_label_applied(set_id)

    async def _ensure_set_embedding_dimensions(self, set_id: str, dimensions: int):
        cached = self._set_embedding_dimensions.get(set_id)
        if cached is not None:
            if cached != dimensions:
                raise ValueError(
                    "Embedding dimension mismatch for set_id "
                    f"{set_id}: expected {cached}, got {dimensions}"
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
                f"{set_id}: expected {db_dims}, got {dimensions}"
            )
        self._set_embedding_dimensions[set_id] = db_dims
        await self._ensure_set_label_applied(set_id)
        await self._ensure_vector_index(set_id, db_dims)

    async def _ensure_set_label_applied(self, set_id: str):
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

    async def _get_feature_dimensions(self, feature_id: FeatureIdT):
        records, _, _ = await self._driver.execute_query(
            """
            MATCH (f:Feature {id: $feature_id})
            RETURN f.set_id AS set_id, f.embedding_dimensions AS embedding_dimensions,
                   CASE WHEN f.embedding_dimensions IS NULL AND f.embedding IS NOT NULL
                        THEN size(f.embedding)
                        ELSE f.embedding_dimensions END AS resolved_dimensions
            """,
            feature_id=int(feature_id),
        )
        if not records:
            return None
        record = dict(records[0])
        if record.get("embedding_dimensions") is None and record.get(
            "resolved_dimensions"
        ) is not None:
            record["embedding_dimensions"] = record["resolved_dimensions"]
        return record
