"""Upgrading and checking a lineage's tables."""

from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, inspect
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.errors import SchemaMismatchError
from memmachine_server.common.schema_migration import (
    Lineage,
    check_lineage,
    upgrade_lineage,
)

_REVISION = """
import sqlalchemy as sa
from alembic import op

revision = "{revision}"
down_revision = {down_revision!r}
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("{table}", sa.Column("id", sa.Integer, primary_key=True))
    {failure}


def downgrade() -> None:
    op.drop_table("{table}")
"""


def _lineage(
    directory: Path, tables: list[str], *, failing_last: bool = False
) -> Lineage:
    """A lineage whose revisions each create one of the tables, in order."""
    directory.mkdir()
    down_revision = None
    for position, table in enumerate(tables):
        revision = f"rev{position}"
        failure = (
            'raise RuntimeError("revision failed")'
            if failing_last and position == len(tables) - 1
            else ""
        )
        (directory / f"{revision}.py").write_text(
            _REVISION.format(
                revision=revision,
                down_revision=down_revision,
                table=table,
                failure=failure,
            )
        )
        down_revision = revision
    metadata = MetaData()
    for table in tables:
        Table(table, metadata, Column("id", Integer, primary_key=True))
    return Lineage(name="toy", versions=directory, metadata=metadata)


async def _table_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        return set(
            await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_table_names()
            )
        )


@pytest.mark.asyncio
async def test_upgrade_creates_the_tables_and_a_rerun_changes_nothing(
    sqlalchemy_engine, tmp_path
):
    lineage = _lineage(tmp_path / "versions", ["first", "second"])

    await upgrade_lineage(sqlalchemy_engine, lineage)
    await upgrade_lineage(sqlalchemy_engine, lineage)

    assert await _table_names(sqlalchemy_engine) == {
        "first",
        "second",
        "alembic_version_toy",
    }
    await check_lineage(sqlalchemy_engine, lineage)


@pytest.mark.asyncio
async def test_check_refuses_missing_tables(sqlalchemy_engine, tmp_path):
    lineage = _lineage(tmp_path / "versions", ["first"])

    with pytest.raises(
        SchemaMismatchError, match=r"missing.*memmachine-server migrate"
    ):
        await check_lineage(sqlalchemy_engine, lineage)


@pytest.mark.asyncio
async def test_check_refuses_an_older_revision(sqlalchemy_engine, tmp_path):
    older = _lineage(tmp_path / "older", ["first"])
    newer = _lineage(tmp_path / "newer", ["first", "second"])
    await upgrade_lineage(sqlalchemy_engine, older)

    with pytest.raises(
        SchemaMismatchError, match="at revision rev0, not at revision rev1"
    ):
        await check_lineage(sqlalchemy_engine, newer)


@pytest.mark.asyncio
async def test_check_refuses_a_revision_it_does_not_know(sqlalchemy_engine, tmp_path):
    newer = _lineage(tmp_path / "newer", ["first", "second"])
    older = _lineage(tmp_path / "older", ["first"])
    await upgrade_lineage(sqlalchemy_engine, newer)

    with pytest.raises(SchemaMismatchError, match="newer than this server"):
        await check_lineage(sqlalchemy_engine, older)


@pytest.mark.asyncio
async def test_a_failed_revision_leaves_no_tables(sqlalchemy_engine, tmp_path):
    lineage = _lineage(tmp_path / "versions", ["first", "second"], failing_last=True)

    with pytest.raises(RuntimeError, match="revision failed"):
        await upgrade_lineage(sqlalchemy_engine, lineage)

    assert await _table_names(sqlalchemy_engine) == set()
