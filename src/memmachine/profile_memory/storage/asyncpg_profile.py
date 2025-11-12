import functools
import json
import logging
from collections.abc import Iterator, Mapping
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from memmachine.profile_memory.storage.storage_base import ProfileStorageBase

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


class AsyncPgProfileStorage(ProfileStorageBase):
    """
    asyncpg implementation for ProfileStorageBase
    """

    @staticmethod
    def build_config(config: dict[str, Any]) -> ProfileStorageBase:
        return AsyncPgProfileStorage(config)

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

        self.main_table = "prof"
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

    async def delete_all(self):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {self.main_table} CASCADE")
            await conn.execute(f"TRUNCATE TABLE {self.history_table} CASCADE")
            await conn.execute(f"TRUNCATE TABLE {self.junction_table} CASCADE")

    async def get_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, dict[str, Any | list[Any]]]:
        result: dict[str, dict[str, list[Any]]] = {}
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT feature, value, tag, create_at FROM {self.main_table}
                WHERE user_id = $1
                AND isolations @> $2
                """,
                user_id,
                json.dumps(isolations),
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

    async def get_citation_list(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[int]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            result = await conn.fetch(
                f"""
                SELECT j.content_id
                FROM {self.main_table} p
                LEFT JOIN {self.junction_table} j ON p.id = j.profile_id
                WHERE user_id = $1 AND feature = $2
                AND value = $3 AND tag = $4
                AND isolations @> $5
            """,
                user_id,
                feature,
                value,
                tag,
                json.dumps(isolations),
            )
            return [i[0] for i in result]

    async def delete_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                    DELETE FROM {self.main_table}
                    WHERE user_id = $1
                    AND isolations @> $2
                    """,
                user_id,
            )

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

        value = str(value)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                pid = await conn.fetchval(
                    f"""
                    INSERT INTO {self.main_table}
                    (user_id, tag, feature, value, embedding, metadata, isolations)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """,
                    user_id,
                    tag,
                    feature,
                    value,
                    embedding,
                    json.dumps(metadata),
                    json.dumps(isolations),
                )

                if pid is None:
                    return
                if len(citations) == 0:
                    return
                await conn.executemany(
                    f"""
                    INSERT INTO {self.junction_table}
                    (profile_id, content_id)
                    VALUES ($1, $2)
                """,
                    [(pid, c) for c in citations],
                )

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

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            if value is None:
                await conn.execute(
                    f"""
                    DELETE FROM {self.main_table}
                    WHERE user_id = $1 AND feature = $2 AND tag = $3
                    AND isolations @> $4
                    """,
                    user_id,
                    feature,
                    tag,
                    json.dumps(isolations),
                )
            else:
                await conn.execute(
                    f"""
                    DELETE FROM {self.main_table}
                    WHERE user_id = $1 AND feature = $2 AND tag = $3 AND value = $4
                    AND isolations @> $5
                    """,
                    user_id,
                    feature,
                    tag,
                    value,
                    json.dumps(isolations),
                )

    async def delete_profile_feature_by_id(self, pid: int):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
            DELETE FROM {self.main_table}
            where id = $1
            """,
                pid,
            )

    async def get_all_citations_for_ids(
        self, pids: list[int]
    ) -> list[tuple[int, dict[str, bool | int | float | str]]]:
        if len(pids) == 0:
            return []
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            stm = f"""
                SELECT DISTINCT j.content_id, h.isolations
                FROM {self.junction_table} j
                JOIN {self.history_table} h ON j.content_id = h.id
                WHERE j.profile_id = ANY($1)
            """
            res = await conn.fetch(stm, pids)
            return [(i[0], json.loads(i[1])) for i in res]

    async def get_large_profile_sections(
        self,
        user_id: str,
        thresh: int = 20,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve every section of the user's profile which has more then 20 entries, formatted as json.
        """
        if isolations is None:
            isolations = {}

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
                WHERE user_id = $1
                AND isolations @> $2
                AND tag IN (
                    SELECT tag
                    FROM {self.main_table}
                    WHERE user_id = $1
                    AND isolations @> $2
                    GROUP BY tag
                    HAVING COUNT(*) >= $3
                )
                GROUP BY tag
            """,
                user_id,
                json.dumps(isolations),
                thresh,
            )
            out = [json.loads(obj[0]) for obj in agg]
            # print("large_profile_sections for user_id", out)
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
        user_id: str,
        qemb: np.ndarray,
        k: int,
        min_cos: float,
        isolations: dict[str, bool | int | float | str] | None = None,
        include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        if isolations is None:
            isolations = {}

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
                                WHERE p.id = j.profile_id
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
                WHERE p.user_id = $2
                AND -(p.embedding <#> $1::vector) > $3
                AND p.isolations @> $4
                GROUP BY p.tag, p.feature, p.value, p.id, p.embedding
                ORDER BY -(p.embedding <#> $1::vector) DESC
                LIMIT $5
                """,
                qemb,
                user_id,
                min_cos,
                json.dumps(isolations),
                k,
            )
            res = [json.loads(a[0]) for a in agg]
            return res

    async def add_history(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> Mapping[str, Any]:
        if isolations is None:
            isolations = {}
        if metadata is None:
            metadata = {}

        stm = f"""
            INSERT INTO {self.history_table} (user_id, content, metadata, isolations)
            VALUES($1, $2, $3, $4)
            RETURNING id, user_id, content, metadata, isolations
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                stm,
                user_id,
                content,
                json.dumps(metadata),
                json.dumps(isolations),
            )
        return RecordMapping(row)

    async def delete_history(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        stm = f"""
            DELETE FROM {self.history_table}
            WHERE user_id = $1 AND isolations @> $2
            AND timestamp >= {start_time} AND timestamp <= {end_time}
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(stm, user_id, json.dumps(isolations))

    async def get_history_messages_by_ingestion_status(
        self,
        user_id: str,
        k: int = 10,
        is_ingested: bool = False,
    ) -> list[Mapping[str, Any]]:
        stm = f"""
            SELECT id, user_id, content, metadata, isolations FROM {self.history_table}
            WHERE user_id = $1 AND ingested = $2
            ORDER BY create_at DESC
            LIMIT $3
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(stm, user_id, is_ingested, k)
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
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[str]:
        if isolations is None:
            isolations = {}

        stm = f"""
            SELECT content FROM {self.history_table}
            WHERE timestamp >= $1 AND timestamp <= $2 AND user_id=$3
            AND isolations @> $4
            ORDER BY timestamp ASC
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                stm, start_time, end_time, user_id, json.dumps(isolations)
            )
            print(rows)
            return rows

    async def purge_history(
        self,
        user_id: str,
        start_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        if isolations is None:
            isolations = {}

        query = f"""
            DELETE FROM {self.history_table}
            WHERE user_id = $1 AND isolations @> $2 AND start_time > $3
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(query, user_id, start_time, json.dumps(isolations))

    async def cleanup(self):
        await self._pool.close()
