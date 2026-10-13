"""
Alembic environment shared by every lineage: runs one lineage's revisions.

The runner passes the lineage and an open connection. From the alembic
command line, the ini section names the lineage as `module:attribute`, and
`-x url=<database URL>` names the database.
"""

import asyncio
import importlib

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.schema_migration.lineage import Lineage

config = context.config


def _lineage() -> Lineage:
    lineage = config.attributes.get("lineage")
    if lineage is None:
        module_name, _, attribute = config.get_main_option("lineage", "").partition(":")
        lineage = getattr(importlib.import_module(module_name), attribute)
    return lineage


lineage = _lineage()


def _include_name(name: str | None, type_: str, _parent_names: object) -> bool:
    # Other stores' tables can share the database; autogenerate compares
    # only this lineage's.
    return type_ != "table" or name in lineage.metadata.tables


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=lineage.metadata,
        version_table=lineage.version_table,
        include_name=_include_name,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations_on_url(url: str) -> None:
    engine = create_async_engine(url, poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


connection = config.attributes.get("connection")
if connection is not None:
    _run_migrations(connection)
else:
    asyncio.run(
        _run_migrations_on_url(context.get_x_argument(as_dictionary=True)["url"])
    )
