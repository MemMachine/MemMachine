import json
import logging
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

# TODO: AsyncPgProfileStorage should inherit from ProfileStorageBase

logger = logging.getLogger(__name__)


class AsyncPgProfileStorage:
    """
    asyncpg implementation for ProfileStorageBase
    """

    main_table = "prof"
    junction_table = "citations"
    history_table = "history"

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

    async def startup(self):
        """
        initializes connection pool
        """
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self._config["host"],
                port=self._config["port"],
                user=self._config["user"],
                password=self._config["password"],
                database=self._config["database"],
                init=register_vector,
            )

    async def delete_all(self):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"TRUNCATE TABLE {AsyncPgProfileStorage.main_table} CASCADE"
            )
            await conn.execute(
                f"TRUNCATE TABLE {AsyncPgProfileStorage.history_table} CASCADE"
            )
            await conn.execute(
                f"TRUNCATE TABLE {AsyncPgProfileStorage.junction_table} CASCADE"
            )

    async def get_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] = {},
    ) -> dict[str, dict[str, Any | list[Any]]]:
        result: dict[str, dict[str, list[Any]]] = {}
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT feature, value, tag, create_at FROM {AsyncPgProfileStorage.main_table}
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
        isolations: dict[str, bool | int | float | str] = {},
    ) -> list[int]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            result = await conn.fetch(
                f"""
                SELECT j.content_id
                FROM {AsyncPgProfileStorage.main_table} p
                LEFT JOIN {AsyncPgProfileStorage.junction_table} j ON p.id = j.profile_id
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
        isolations: dict[str, bool | int | float | str] = {},
    ):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                    DELETE FROM {AsyncPgProfileStorage.main_table}
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
        metadata: dict[str, Any] = {},
        isolations: dict[str, bool | int | float | str] = {},
        citations: list[int] = [],
    ):
        value = str(value)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                pid = await conn.fetchval(
                    f"""
                    INSERT INTO {AsyncPgProfileStorage.main_table}
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
                    INSERT INTO {AsyncPgProfileStorage.junction_table}
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
        isolations: dict[str, bool | int | float | str] = {},
    ):
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            if value is None:
                await conn.execute(
                    f"""
                    DELETE FROM {AsyncPgProfileStorage.main_table}
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
                    DELETE FROM {AsyncPgProfileStorage.main_table}
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
            DELETE FROM {AsyncPgProfileStorage.main_table}
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
                FROM {AsyncPgProfileStorage.junction_table} j
                JOIN {AsyncPgProfileStorage.history_table} h ON j.content_id = h.id
                WHERE j.profile_id = ANY($1)
            """
            res = await conn.fetch(stm, pids)
            return [(i[0], json.loads(i[1])) for i in res]

    async def get_large_profile_sections(
        self,
        user_id: str,
        thresh: int = 20,
        isolations: dict[str, bool | int | float | str] = {},
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve every section of the user's profile which has more then 20 entries, formatted as json.
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
                FROM {AsyncPgProfileStorage.main_table}
                WHERE user_id = $1
                AND isolations @> $2
                AND tag IN (
                    SELECT tag
                    FROM {AsyncPgProfileStorage.main_table}
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
        isolations: dict[str, bool | int | float | str] = {},
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
                                FROM {AsyncPgProfileStorage.junction_table} j
                                JOIN {AsyncPgProfileStorage.history_table} h ON j.content_id = h.id
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
                FROM {AsyncPgProfileStorage.main_table} p
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
        metadata: dict[str, str] = {},
        isolations: dict[str, bool | int | float | str] = {},
    ) -> asyncpg.Record:
        stm = f"""
            INSERT INTO {AsyncPgProfileStorage.history_table} (user_id, content, metadata, isolations)
            VALUES($1, $2, $3, $4)
            RETURNING id, user_id, content, metadata, isolations
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(
                stm,
                user_id,
                content,
                json.dumps(metadata),
                json.dumps(isolations),
            )

    async def delete_history(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] = {},
    ):
        stm = f"""
            DELETE FROM {AsyncPgProfileStorage.history_table}
            WHERE user_id = $1 AND isolations @> $2
            AND timestamp >= {start_time} AND timestamp <= {end_time}
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(stm, user_id, json.dumps(isolations))

    async def get_last_history_messages(
        self,
        user_id: str,
        k: int = 0,
        isolations: dict[str, bool | int | float | str] = {},
    ) -> list[asyncpg.Record]:
        stm = f"""
            SELECT id, user_id, content, metadata, isolations FROM {AsyncPgProfileStorage.history_table}
            WHERE user_id=$1 AND isolations @> 2
            ORDER BY timestamp DESC
            LIMIT {k}
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(stm, user_id, json.dumps(isolations))
            return rows

    async def get_history_message(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] = {},
    ) -> list[str]:
        stm = f"""
            SELECT content FROM {AsyncPgProfileStorage.history_table}
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
        isolations: dict[str, bool | int | float | str] = {},
    ):
        query = f"""
            DELETE FROM {AsyncPgProfileStorage.history_table}
            WHERE user_id = $1 AND isolations @> $2 AND start_time > $3
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(query, user_id, start_time, json.dumps(isolations))

    async def cleanup(self):
        await self._pool.close()
