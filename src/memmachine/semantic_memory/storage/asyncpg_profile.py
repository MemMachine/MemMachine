import functools
import json
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Iterator, Optional

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from pydantic import InstanceOf, validate_call

from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase

logger = logging.getLogger(__name__)


class RecordMapping(Mapping):
    def __init__(self, inner: asyncpg.Record):
        # inner is the external Record instance
        self._inner = inner

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner.keys())

    def __len__(self) -> int:
        return len(self._inner)

    def get(self, key, default=None):
        return self._inner.get(key, default)

    def items(self):
        return self._inner.items()

    def keys(self):
        return self._inner.keys()

    def values(self):
        return self._inner.values()


class AsyncPgSemanticStorage(SemanticStorageBase):
    """
    asyncpg implementation for ProfileStorageBase
    """

    @staticmethod
    def build_config(config: dict[str, Any]) -> SemanticStorageBase:
        return AsyncPgSemanticStorage(config)

    def __init__(self, config: dict[str, Any]):
        self._pool = None
        if config["host"] is None:
            raise ValueError("DB host is not in config")
        if config["port"] is None:
            raise ValueError("DB port is not in config")
        if config["user"] is None:
            raise ValueError("DB user is not in config")
        if config["password"] is None:
            raise ValueError("DB password is not in config")
        if config["database"] is None:
            raise ValueError("DB database is not in config")
        self._config = config

        self.main_table = "semantic"
        self.junction_table = "citations"
        self.history_table = "history"
        schema = self._config.get("schema")
        if schema is not None and schema.strip() != "":
            schema = schema.strip()
            self.main_table = f"{schema}.{self.main_table}"
            self.junction_table = f"{schema}.{self.junction_table}"
            self.history_table = f"{schema}.{self.history_table}"

    async def startup(self):
        """
        initializes connection pool
        """
        if self._pool is None:
            kwargs = {}
            # if using supabase transaction pooler, it does not support prepared statements
            if "statement_cache_size" in self._config:
                kwargs["statement_cache_size"] = self._config["statement_cache_size"]
            self._pool = await asyncpg.create_pool(
                host=self._config["host"],
                port=self._config["port"],
                user=self._config["user"],
                password=self._config["password"],
                database=self._config["database"],
                init=functools.partial(
                    register_vector, schema=self._config.get("vector_schema", "public")
                ),
                **kwargs,
            )

    async def cleanup(self):
        await self._pool.close()

    async def delete_all(self):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {self.main_table} CASCADE")
            await conn.execute(f"TRUNCATE TABLE {self.history_table} CASCADE")
            await conn.execute(f"TRUNCATE TABLE {self.junction_table} CASCADE")

    @validate_call
    async def get_set_features(
        self,
        set_id: str,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> dict[str, dict[str, Any | list[Any]]]:
        result: dict[str, dict[str, list[Any]]] = {}
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT feature, value, tag, create_at FROM {self.main_table}
                WHERE set_id = $1
                """,
                set_id,
            )

            for feature, value, tag, create_at in rows:
                payload = {
                    "value": value,
                }
                if tag not in result:
                    result[tag] = {}
                if feature not in result[tag]:
                    result[tag][feature] = []
                result[tag][feature].append(payload)
            for tag, fv in result.items():
                for feature, value in fv.items():
                    if len(value) == 1:
                        fv[feature] = value[0]
            return result

    @validate_call
    async def get_citation_list(
        self,
        set_id: str,
        feature: str,
        value: str,
        tag: str,
    ) -> list[int]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            result = await conn.fetch(
                f"""
                SELECT j.content_id
                FROM {self.main_table} p
                LEFT JOIN {self.junction_table} j ON p.id = j.semantic_id
                WHERE set_id = $1 AND feature = $2
                AND value = $3 AND tag = $4
            """,
                set_id,
                feature,
                value,
                tag,
            )
            return [i[0] for i in result]

    @validate_call
    async def delete_feature_set(
        self,
        set_id: str,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                    DELETE FROM {self.main_table}
                    WHERE set_id = $1
                    """,
                set_id,
            )

    @validate_call
    async def add_feature(
        self,
        set_id: str,
        semantic_type_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
        citations: list[int] | None = None,
    ) -> int:
        if metadata is None:
            metadata = {}
        if citations is None:
            citations = []

        value = str(value)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                idx = await conn.fetchval(
                    f"""
                    INSERT INTO {self.main_table}
                    (set_id, semantic_type, tag, feature, value, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """,
                    set_id,
                    semantic_type_id,
                    tag,
                    feature,
                    value,
                    embedding,
                    json.dumps(metadata),
                )

                if idx is None:
                    return -1
                if len(citations) != 0:
                    await conn.executemany(
                        f"""
                        INSERT INTO {self.junction_table}
                        (semantic_id, content_id)
                        VALUES ($1, $2)
                    """,
                        [(idx, c) for c in citations],
                    )

                return idx

    async def delete_feature_with_filter(
        self,
        *,
        set_id: str,
        semantic_type_id: str,
        feature: str,
        tag: str,
    ):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                DELETE FROM {self.main_table}
                WHERE set_id = $1 AND feature = $2 AND tag = $3
                """,
                set_id,
                feature,
                tag,
            )

    async def delete_features(self, feature_ids: list[int]):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
            DELETE FROM {self.main_table}
            where id = ANY($1)
            """,
                feature_ids,
            )

    async def get_all_citations_for_ids(self, feature_ids: list[int]) -> list[int]:
        if len(feature_ids) == 0:
            return []
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            stm = f"""
                SELECT DISTINCT j.content_id
                FROM {self.junction_table} j
                JOIN {self.history_table} h ON j.content_id = h.id
                WHERE j.semantic_id = ANY($1)
            """
            res = await conn.fetch(stm, feature_ids)
            return [i[0] for i in res]

    async def get_large_feature_sections(
        self,
        set_id: str,
        thresh: int = 20,
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve every section of the user's feature set which has more then 20 entries, formatted as json.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            agg = await conn.fetch(
                f"""
                SELECT JSON_AGG(JSON_BUILD_OBJECT(
                    'tag', tag,
                    'feature', feature,
                    'value', value,
                    'metadata', JSON_BUILD_OBJECT('id', id)
                ))
                FROM {self.main_table}
                WHERE set_id = $1
                AND tag IN (
                    SELECT tag
                    FROM {self.main_table}
                    WHERE set_id = $1
                    GROUP BY tag
                    HAVING COUNT(*) >= $2
                )
                GROUP BY tag
            """,
                set_id,
                thresh,
            )
            out = [json.loads(obj[0]) for obj in agg]
            return out

    def _normalize_value(self, value: Any) -> str:
        if isinstance(value, list):
            msg = ""
            for item in value:
                msg = msg + " " + self._normalize_value(item)
            return msg
        if isinstance(value, dict):
            msg = ""
            for key, item in value.items():
                msg = msg + " " + key + ": " + self._normalize_value(item)
            return msg
        return str(value)

    async def semantic_search(
        self,
        set_id: str,
        qemb: np.ndarray,
        k: int,
        min_cos: float,
        include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            agg = await conn.fetch(
                """
                SELECT JSON_BUILD_OBJECT(
                    'tag', p.tag,
                    'feature', p.feature,
                    'value', p.value,
                    'metadata', JSON_BUILD_OBJECT(
                        'id', p.id,
                        'similarity_score', (-(p.embedding <#> $1::vector))
                """
                + (
                    f"""
                        , 'citations', COALESCE(
                            (
                                SELECT JSON_AGG(h.content)
                                FROM {self.junction_table} j
                                JOIN {self.history_table} h ON j.content_id = h.id
                                WHERE p.id = j.semantic_id
                            ),
                            '[]'::json
                        )
                    """
                    if include_citations
                    else ""
                )
                + f"""
                    )
                )
                FROM {self.main_table} p
                WHERE p.set_id = $2
                AND -(p.embedding <#> $1::vector) > $3
                GROUP BY p.tag, p.feature, p.value, p.id, p.embedding
                ORDER BY -(p.embedding <#> $1::vector) DESC
                LIMIT $4
                """,
                qemb,
                set_id,
                min_cos,
                k,
            )
            res = [json.loads(a[0]) for a in agg]
            return res

    async def add_history(
        self,
        set_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> Mapping[str, Any]:
        if metadata is None:
            metadata = {}

        stm = f"""
            INSERT INTO {self.history_table} (set_id, content, metadata)
            VALUES($1, $2, $3)
            RETURNING id, set_id, content, metadata
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                stm,
                set_id,
                content,
                json.dumps(metadata),
            )
        return RecordMapping(row)

    async def delete_history(
        self,
        set_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        if start_time is None:
            start_time = datetime.fromtimestamp(0)
        if end_time is None:
            end_time = datetime.now()

        stm = f"""
            DELETE FROM {self.history_table}
            WHERE set_id=$1
            AND create_at >= $2
            AND create_at <= $3
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(stm, set_id, start_time, end_time)

    async def get_history_messages_by_ingestion_status(
        self,
        set_id: str,
        k: int = 10,
        is_ingested: bool = False,
    ) -> list[Mapping[str, Any]]:
        stm = f"""
            SELECT id, set_id, content, metadata FROM {self.history_table}
            WHERE set_id = $1 AND ingested = $2
            ORDER BY create_at ASC
            LIMIT $3
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(stm, set_id, is_ingested, k)
            return [RecordMapping(row) for row in rows]

    async def get_uningested_history_messages_count(self) -> int:
        stm = f"""
            SELECT COUNT(*) FROM {self.history_table}
            WHERE ingested=FALSE
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetchval(stm)
            return rows

    async def mark_messages_ingested(self, ids: list[int]) -> None:
        if not ids:
            return  # nothing to do

        stm = f"""
                UPDATE {self.history_table}
                SET ingested = TRUE
                WHERE id = ANY($1::bigint[])
            """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(stm, ids)

    async def get_history_message(
        self,
        set_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[str]:
        if start_time is None:
            start_time = datetime.fromtimestamp(0)
        if end_time is None:
            end_time = datetime.now()

        stm = f"""
            SELECT content FROM {self.history_table}
            WHERE create_at >= $1 AND create_at <= $2 AND set_id=$3
            ORDER BY create_at ASC
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(stm, start_time, end_time, set_id)
            return [row["content"] for row in rows]
