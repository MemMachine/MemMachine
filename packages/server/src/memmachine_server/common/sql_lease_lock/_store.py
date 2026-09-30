"""Atomic SQL storage for exclusive leases."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Column,
    Integer,
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

_metadata = MetaData()
_lease_lock = Table(
    "lease_lock",
    _metadata,
    Column("key", String, primary_key=True),
    Column("fencing_token", BigInteger, nullable=False),
    Column("lease_id", String, nullable=True),
    Column("expires_at_ms", BigInteger, nullable=True),
)
_counter = Table(
    "lease_lock_counter",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("value", BigInteger, nullable=False),
)


@dataclass(frozen=True)
class StoredLease:
    """A granted exclusive lease as persisted in SQL."""

    key: str
    lease_id: str
    expires_at_ms: int
    fencing_token: int


class SQLLeaseStore:
    """Serialize exclusive lease changes through one row per resource."""

    def __init__(self, engine: AsyncEngine) -> None:
        if engine.dialect.name not in {"postgresql", "sqlite"}:
            raise ValueError("SQL lease locks require PostgreSQL or SQLite")
        self._engine = engine

    async def startup(self) -> None:
        """Create the lock and durable fencing counter tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)
            values = {"id": 1, "value": 0}
            if self._engine.dialect.name == "sqlite":
                statement = sqlite_insert(_counter).values(**values)
            else:
                statement = pg_insert(_counter).values(**values)
            await conn.execute(statement.on_conflict_do_nothing(index_elements=["id"]))

    @asynccontextmanager
    async def _transaction(self) -> AsyncGenerator[AsyncConnection, None]:
        if self._engine.dialect.name == "sqlite":
            async with self._engine.connect() as conn:
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

    async def _lock_resource(
        self, conn: AsyncConnection, key: str, *, create: bool = False
    ) -> tuple[str | None, int | None] | None:
        if create:
            values = {"key": key, "fencing_token": 0}
            if self._engine.dialect.name == "sqlite":
                statement = sqlite_insert(_lease_lock).values(**values)
            else:
                statement = pg_insert(_lease_lock).values(**values)
            await conn.execute(statement.on_conflict_do_nothing(index_elements=["key"]))
        query = select(
            _lease_lock.c.lease_id,
            _lease_lock.c.expires_at_ms,
        ).where(_lease_lock.c.key == key)
        if self._engine.dialect.name == "postgresql":
            query = query.with_for_update()
        row = (await conn.execute(query)).one_or_none()
        if row is None:
            return None
        return row.lease_id, row.expires_at_ms

    async def _next_token(self, conn: AsyncConnection) -> int:
        token = (
            await conn.execute(
                update(_counter)
                .where(_counter.c.id == 1)
                .values(value=_counter.c.value + 1)
                .returning(_counter.c.value)
            )
        ).scalar_one()
        return int(token)

    async def _now_ms(self, conn: AsyncConnection) -> int:
        if self._engine.dialect.name == "sqlite":
            expression = (func.julianday("now") - 2440587.5) * 86_400_000
        else:
            expression = func.extract("epoch", func.clock_timestamp()) * 1_000
        return int(
            (await conn.execute(select(cast(expression, BigInteger)))).scalar_one()
        )

    async def try_acquire(self, key: str, duration_ms: int) -> StoredLease | None:
        """Grant an exclusive lease if the key has no live holder."""
        if not key or duration_ms <= 0:
            raise ValueError("key and duration must be valid")
        async with self._transaction() as conn:
            resource = await self._lock_resource(conn, key, create=True)
            assert resource is not None
            holder_id, expiry = resource
            now_ms = await self._now_ms(conn)
            if holder_id is not None and expiry is not None and expiry > now_ms:
                return None
            token = await self._next_token(conn)
            lease_id = uuid4().hex
            expires_at_ms = now_ms + duration_ms
            await conn.execute(
                update(_lease_lock)
                .where(_lease_lock.c.key == key)
                .values(
                    fencing_token=token,
                    lease_id=lease_id,
                    expires_at_ms=expires_at_ms,
                )
            )
            return StoredLease(
                key=key,
                lease_id=lease_id,
                expires_at_ms=expires_at_ms,
                fencing_token=token,
            )

    async def renew(self, key: str, lease_id: str, duration_ms: int) -> int | None:
        """Extend a live lease and return its new expiry."""
        if duration_ms <= 0:
            raise ValueError("duration must be positive")
        async with self._transaction() as conn:
            resource = await self._lock_resource(conn, key)
            if resource is None:
                return None
            holder_id, expiry = resource
            now_ms = await self._now_ms(conn)
            if holder_id != lease_id or expiry is None or expiry <= now_ms:
                return None
            expires_at_ms = now_ms + duration_ms
            await conn.execute(
                update(_lease_lock)
                .where(_lease_lock.c.key == key)
                .values(expires_at_ms=expires_at_ms)
            )
            return expires_at_ms

    async def release(self, key: str, lease_id: str) -> bool:
        """Delete a live lease row while retaining the global token counter."""
        async with self._transaction() as conn:
            resource = await self._lock_resource(conn, key)
            if resource is None:
                return False
            holder_id, expiry = resource
            now_ms = await self._now_ms(conn)
            if holder_id != lease_id or expiry is None or expiry <= now_ms:
                return False
            await conn.execute(
                delete(_lease_lock).where(
                    _lease_lock.c.key == key, _lease_lock.c.lease_id == lease_id
                )
            )
            return True
