"""Every store's lineage creates the tables its store's code declares."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from memmachine_server.common.episode_store.episode_sqlalchemy_store import (
    EPISODE_STORE_LINEAGE,
)
from memmachine_server.common.schema_migration import Lineage, upgrade_lineage
from memmachine_server.common.schema_migration.runner import _config
from memmachine_server.common.session_manager.session_data_manager_sql_impl import (
    SESSION_LINEAGE,
)
from memmachine_server.common.sql_lease_lock._store import LEASE_LOCK_LINEAGE
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    PARTITION_REGISTRY_LINEAGE,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLITE_VEC_VECTOR_STORE_LINEAGE,
)
from memmachine_server.common.vector_store.sqlite_vector_store import (
    SQLITE_VECTOR_STORE_LINEAGE,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SEGMENT_STORE_LINEAGE,
)

LINEAGES = [
    SESSION_LINEAGE,
    EPISODE_STORE_LINEAGE,
    LEASE_LOCK_LINEAGE,
    SEGMENT_STORE_LINEAGE,
    PARTITION_REGISTRY_LINEAGE,
    SQLITE_VECTOR_STORE_LINEAGE,
    SQLITE_VEC_VECTOR_STORE_LINEAGE,
]


def _lineage_name(lineage: Lineage) -> str:
    return lineage.name


def test_lineage_names_are_unique():
    names = [lineage.name for lineage in LINEAGES]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("lineage", LINEAGES, ids=_lineage_name)
def test_lineage_has_one_head(lineage):
    assert len(ScriptDirectory.from_config(_config(lineage)).get_heads()) == 1


@pytest.mark.parametrize("lineage", LINEAGES, ids=_lineage_name)
def test_lineage_revisions_are_packaged(lineage):
    # setuptools packages a directory only when it has an __init__.py.
    assert (lineage.versions / "__init__.py").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("lineage", LINEAGES, ids=_lineage_name)
async def test_newest_revision_creates_the_declared_tables(sqlalchemy_engine, lineage):
    await upgrade_lineage(sqlalchemy_engine, lineage)

    def differences(connection):
        context = MigrationContext.configure(
            connection,
            opts={
                "include_name": lambda name, type_, _parents: (
                    type_ != "table" or name in lineage.metadata.tables
                )
            },
        )
        return compare_metadata(context, lineage.metadata)

    async with sqlalchemy_engine.connect() as connection:
        assert await connection.run_sync(differences) == []
