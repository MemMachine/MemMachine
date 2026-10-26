"""A SQLite-backed store's recorded schema."""

from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import MetaData, select

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.errors import SchemaMismatchError
from memmachine_server.common.sql import write_connection
from memmachine_server.common.vector_store import VectorStoreSchema
from memmachine_server.common.vector_store.sql_store_schema import (
    check_store_schema,
    record_store_schema,
    store_schema_tables,
)

_SCHEMA = VectorStoreSchema(
    vector_dimensions=3,
    similarity_metric=SimilarityMetric.DOT,
    indexed_properties={"name": str, "created_at": datetime},
)


async def _recorded(engine, tables) -> tuple[list[tuple], list[tuple]]:
    store_table, declared_property_table = tables
    async with engine.connect() as connection:
        stores = (await connection.execute(select(store_table))).all()
        properties = (
            await connection.execute(
                select(declared_property_table).order_by(
                    declared_property_table.c.property_key
                )
            )
        ).all()
    return [tuple(row) for row in stores], [tuple(row) for row in properties]


@pytest_asyncio.fixture
async def tables(sqlalchemy_sqlite_engine):
    metadata = MetaData()
    tables = store_schema_tables(metadata, "test_store")
    async with sqlalchemy_sqlite_engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    return tables


@pytest.mark.asyncio
async def test_records_the_dimensions_metric_and_declared_properties(
    sqlalchemy_sqlite_engine, tables
):
    async with write_connection(sqlalchemy_sqlite_engine) as connection:
        await record_store_schema(connection, tables, "events", _SCHEMA)

    assert await _recorded(sqlalchemy_sqlite_engine, tables) == (
        [("events", 3, "dot")],
        [("events", "created_at", "datetime"), ("events", "name", "str")],
    )


@pytest.mark.asyncio
async def test_a_recorded_schema_is_kept(sqlalchemy_sqlite_engine, tables):
    async with write_connection(sqlalchemy_sqlite_engine) as connection:
        await record_store_schema(connection, tables, "events", _SCHEMA)
    before = await _recorded(sqlalchemy_sqlite_engine, tables)

    async with write_connection(sqlalchemy_sqlite_engine) as connection:
        await record_store_schema(
            connection,
            tables,
            "events",
            VectorStoreSchema(vector_dimensions=4, indexed_properties={}),
        )

    assert await _recorded(sqlalchemy_sqlite_engine, tables) == before


@pytest.mark.asyncio
async def test_the_store_s_own_recorded_schema_passes_the_check(
    sqlalchemy_sqlite_engine, tables
):
    async with write_connection(sqlalchemy_sqlite_engine) as connection:
        await record_store_schema(connection, tables, "events", _SCHEMA)

    await check_store_schema(sqlalchemy_sqlite_engine, tables, "events", _SCHEMA)


@pytest.mark.asyncio
async def test_the_check_refuses_a_store_with_no_recorded_schema(
    sqlalchemy_sqlite_engine, tables
):
    with pytest.raises(SchemaMismatchError, match="memmachine-server migrate"):
        await check_store_schema(sqlalchemy_sqlite_engine, tables, "events", _SCHEMA)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"vector_dimensions": 4}, "holds vectors other than"),
        ({"similarity_metric": SimilarityMetric.COSINE}, "holds vectors other than"),
        ({"indexed_properties": {"name": str}}, "declares other properties"),
        (
            {"indexed_properties": {"name": int, "created_at": datetime}},
            "declares other properties",
        ),
    ],
)
async def test_the_check_refuses_a_schema_recorded_otherwise(
    sqlalchemy_sqlite_engine, tables, changes, match
):
    async with write_connection(sqlalchemy_sqlite_engine) as connection:
        await record_store_schema(connection, tables, "events", _SCHEMA)

    with pytest.raises(SchemaMismatchError, match=match):
        await check_store_schema(
            sqlalchemy_sqlite_engine,
            tables,
            "events",
            _SCHEMA.model_copy(update=changes),
        )
