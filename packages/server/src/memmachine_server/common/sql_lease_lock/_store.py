"""Atomic SQL storage for leased read/write locks."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Index,
    MetaData,
    String,
    Table,
    cast,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

LockMode = Literal["read", "write"]

_metadata = MetaData()
_resource = Table(
    "lease_lock_resource",
    _metadata,
    Column("key", String, primary_key=True),
    Column("generation", BigInteger, nullable=False),
)
_holder = Table(
    "lease_lock_holder",
    _metadata,
    Column("lease_id", String, primary_key=True),
    Column("key", String, ForeignKey("lease_lock_resource.key"), nullable=False),
    Column("mode", String, nullable=False),
    Column("fencing_token", BigInteger, nullable=False),
    Column("expires_at_ms", BigInteger, nullable=False),
    Index("ix_lease_lock_holder_key_expiry", "key", "expires_at_ms"),
)


@dataclass(frozen=True)
class StoredLease:
    """A granted lease as persisted in SQL."""

    key: str
    mode: LockMode
    lease_id: str
    expires_at_ms: int
    fencing_token: int


class SqlLeaseStore:
    """Serialize lease changes through a database row per resource."""

    def __init__(self, engine: AsyncEngine) -> None:
        if engine.dialect.name not in {"postgresql", "sqlite"}:
            raise ValueError("SQL lease locks require PostgreSQL or SQLite")
        self._engine = engine

    async def startup(self) -> None:
        """Create lock tables if they do not already exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncConnection]:
        if self._engine.dialect.name == "sqlite":
            async with self._engine.connect() as conn:
                # SQLite's deferred BEGIN would let competing readers inspect
                # holder rows before either had reserved the write lock.
                await conn.exec_driver_sql("BEGIN IMMEDIATE")
                try:
                    yield conn
                    await conn.commit()
                except BaseException:
                    await conn.rollback()
                    raise
        else:
            async with self._engine.begin() as conn:
                yield conn

    async def _lock_resource(self, conn: AsyncConnection, key: str) -> int:
        values = {"key": key, "generation": 0}
        if self._engine.dialect.name == "sqlite":
            statement = sqlite_insert(_resource).values(**values)
        else:
            statement = pg_insert(_resource).values(**values)
        await conn.execute(statement.on_conflict_do_nothing(index_elements=["key"]))
        query = select(_resource.c.generation).where(_resource.c.key == key)
        if self._engine.dialect.name == "postgresql":
            query = query.with_for_update()
        generation = (await conn.execute(query)).scalar_one()
        return int(generation)

    async def _now_ms(self, conn: AsyncConnection) -> int:
        if self._engine.dialect.name == "sqlite":
            expression = (func.julianday("now") - 2440587.5) * 86_400_000
        else:
            expression = func.extract("epoch", func.clock_timestamp()) * 1_000
        return int(
            (await conn.execute(select(cast(expression, BigInteger)))).scalar_one()
        )

    async def try_acquire(
        self, key: str, mode: LockMode, duration_ms: int
    ) -> StoredLease | None:
        """Grant a lease if no live holder conflicts with its mode."""
        if not key or duration_ms <= 0 or mode not in {"read", "write"}:
            raise ValueError("key, mode, and duration must be valid")
        async with self._transaction() as conn:
            generation = await self._lock_resource(conn, key)
            now_ms = await self._now_ms(conn)
            await conn.execute(
                delete(_holder).where(
                    _holder.c.key == key, _holder.c.expires_at_ms <= now_ms
                )
            )
            conflict = select(_holder.c.lease_id).where(_holder.c.key == key)
            if mode == "read":
                conflict = conflict.where(_holder.c.mode == "write")
            if (await conn.execute(conflict.limit(1))).first() is not None:
                return None

            token = generation + 1
            lease_id = uuid4().hex
            expires_at_ms = now_ms + duration_ms
            await conn.execute(
                update(_resource).where(_resource.c.key == key).values(generation=token)
            )
            await conn.execute(
                _holder.insert().values(
                    lease_id=lease_id,
                    key=key,
                    mode=mode,
                    fencing_token=token,
                    expires_at_ms=expires_at_ms,
                )
            )
            return StoredLease(key, mode, lease_id, expires_at_ms, token)

    async def renew(self, key: str, lease_id: str, duration_ms: int) -> int | None:
        """Extend one live lease and return its new expiry."""
        if duration_ms <= 0:
            raise ValueError("duration must be positive")
        async with self._transaction() as conn:
            await self._lock_resource(conn, key)
            now_ms = await self._now_ms(conn)
            row = (
                await conn.execute(
                    select(_holder.c.lease_id).where(
                        _holder.c.key == key,
                        _holder.c.lease_id == lease_id,
                        _holder.c.expires_at_ms > now_ms,
                    )
                )
            ).first()
            if row is None:
                return None
            expires_at_ms = now_ms + duration_ms
            await conn.execute(
                update(_holder)
                .where(_holder.c.lease_id == lease_id)
                .values(expires_at_ms=expires_at_ms)
            )
            return expires_at_ms

    async def release(self, key: str, lease_id: str) -> bool:
        """Remove one live lease, leaving other holders untouched."""
        async with self._transaction() as conn:
            await self._lock_resource(conn, key)
            now_ms = await self._now_ms(conn)
            row = (
                await conn.execute(
                    select(_holder.c.lease_id).where(
                        _holder.c.key == key,
                        _holder.c.lease_id == lease_id,
                        _holder.c.expires_at_ms > now_ms,
                    )
                )
            ).first()
            if row is None:
                return False
            await conn.execute(delete(_holder).where(_holder.c.lease_id == lease_id))
            return True
