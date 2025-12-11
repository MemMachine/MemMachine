from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.engine import Inspector
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    BaseSemanticStorage,
    apply_alembic_migrations,
)
from tests.memmachine.semantic_memory.storage.alembic_pg.helpers import apply_sql_file

pytestmark = pytest.mark.integration

LEGACY_SCHEMA_FILE = Path(__file__).with_name("original_schema.sql")
RESET_STATEMENTS: tuple[str, ...] = (
    "DROP TABLE IF EXISTS citations CASCADE",
    "DROP TABLE IF EXISTS set_ingested_history CASCADE",
    "DROP TABLE IF EXISTS feature CASCADE",
    "DROP TABLE IF EXISTS history CASCADE",
    "DROP TABLE IF EXISTS prof CASCADE",
    "DROP TABLE IF EXISTS metadata.migration_tracker CASCADE",
    "DROP TABLE IF EXISTS alembic_version CASCADE",
    "DROP SCHEMA IF EXISTS metadata CASCADE",
)


@dataclass(frozen=True)
class SchemaSnapshot:
    tables: set[str]
    columns: Mapping[str, Mapping[str, dict[str, object]]]
    indexes: Mapping[str, set[tuple[str, ...]]]


@pytest.mark.asyncio
async def test_alembic_schema_matches_sqlalchemy_metadata(
    sqlalchemy_pg_engine: AsyncEngine,
):
    await _reset_database(sqlalchemy_pg_engine)

    await apply_alembic_migrations(sqlalchemy_pg_engine)
    migrated_schema = await _collect_schema(sqlalchemy_pg_engine)

    await _reset_database(sqlalchemy_pg_engine)

    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseSemanticStorage.metadata.create_all)
    sqlalchemy_schema = await _collect_schema(sqlalchemy_pg_engine)

    _assert_schema_equivalence(migrated_schema, sqlalchemy_schema)

    await _reset_database(sqlalchemy_pg_engine)


@pytest.mark.asyncio
async def test_legacy_schema_upgrades_to_modern_layout(
    sqlalchemy_pg_engine: AsyncEngine,
):
    await _reset_database(sqlalchemy_pg_engine)

    await apply_sql_file(sqlalchemy_pg_engine, LEGACY_SCHEMA_FILE)

    await apply_alembic_migrations(sqlalchemy_pg_engine)

    migrated_schema = await _collect_schema(sqlalchemy_pg_engine)

    await _reset_database(sqlalchemy_pg_engine)

    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseSemanticStorage.metadata.create_all)

    sqlalchemy_schema = await _collect_schema(sqlalchemy_pg_engine)

    _assert_schema_equivalence(migrated_schema, sqlalchemy_schema)

    await _reset_database(sqlalchemy_pg_engine)


async def _reset_database(async_engine: AsyncEngine) -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: BaseSemanticStorage.metadata.drop_all(bind=sync_conn),
        )
        for statement in RESET_STATEMENTS:
            await conn.execute(text(statement))


async def _collect_schema(engine: AsyncEngine) -> SchemaSnapshot:
    async with engine.connect() as conn:

        def _collect(sync_conn):
            inspector = sqlalchemy.inspect(sync_conn)
            _tables = set(inspector.get_table_names())
            _columns = {table: get_table_columns(inspector, table) for table in _tables}
            _indexes = {
                table: get_index_column_sets(inspector, table) for table in _tables
            }
            return SchemaSnapshot(tables=_tables, columns=_columns, indexes=_indexes)

        snapshot = await conn.run_sync(_collect)

    return snapshot


def _assert_schema_equivalence(
    migrated: SchemaSnapshot,
    metadata: SchemaSnapshot,
) -> None:
    assert "alembic_version" in migrated.tables, (
        "alembic_version table is missing after migrations"
    )

    migrated_tables = migrated.tables - {"alembic_version"}
    assert migrated_tables == metadata.tables, (
        "Mismatch between Alembic and SQLAlchemy tables. "
        f"Migrated-only: {sorted(migrated_tables - metadata.tables)}; "
        f"SQLAlchemy-only: {sorted(metadata.tables - migrated_tables)}"
    )

    for table_name in sorted(metadata.tables):
        migrated_columns = migrated.columns[table_name]
        sqlalchemy_columns = metadata.columns[table_name]

        assert set(migrated_columns) == set(sqlalchemy_columns), (
            f"Column mismatch in {table_name}. "
            f"Migrated-only: {sorted(set(migrated_columns) - set(sqlalchemy_columns))}; "
            f"SQLAlchemy-only: {sorted(set(sqlalchemy_columns) - set(migrated_columns))}"
        )

        for column, column_meta in sqlalchemy_columns.items():
            migrated_meta = migrated_columns[column]

            assert normalize_type(migrated_meta["type"]) == normalize_type(
                column_meta["type"],
            ), f"Type mismatch for {table_name}.{column}"

            if not column_meta["nullable"]:
                assert not migrated_meta["nullable"], (
                    f"{table_name}.{column} should be NOT NULL "
                    "to match the SQLAlchemy schema"
                )

        migrated_index_columns = migrated.indexes[table_name]
        sqlalchemy_index_columns = metadata.indexes[table_name]
        assert migrated_index_columns == sqlalchemy_index_columns, (
            f"Index column mismatch in {table_name}. "
            f"Migrated-only: {sorted(migrated_index_columns - sqlalchemy_index_columns)}; "
            f"SQLAlchemy-only: {sorted(sqlalchemy_index_columns - migrated_index_columns)}"
        )


def get_table_columns(
    inspector: Inspector,
    table_name: str,
) -> Mapping[str, dict[str, object]]:
    columns = inspector.get_columns(table_name)
    return {
        column["name"]: {
            "type": str(column["type"]),
            "nullable": column["nullable"],
        }
        for column in columns
    }


def get_index_column_sets(
    inspector: Inspector,
    table_name: str,
) -> set[tuple[str, ...]]:
    indexes = inspector.get_indexes(table_name)
    return {
        tuple(index["column_names"]) for index in indexes if index.get("column_names")
    }


def normalize_type(type_name: str) -> str:
    normalized = type_name.upper()
    normalized = normalized.replace("CHARACTER VARYING", "VARCHAR")
    normalized = normalized.replace("TIMESTAMP WITHOUT TIME ZONE", "TIMESTAMP")
    if normalized.startswith("VARCHAR") or normalized == "TEXT":
        return "TEXT"
    return normalized
