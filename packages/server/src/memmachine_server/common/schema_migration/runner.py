"""Upgrade a lineage's tables to its newest revision."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.sql import write_connection

from .lineage import Lineage

_ENVIRONMENT = Path(__file__).parent / "environment"


async def upgrade_lineage(engine: AsyncEngine, lineage: Lineage) -> None:
    """
    Upgrade the lineage's tables in the engine's database to its newest revision.

    The upgrade runs in one transaction, so a revision that fails leaves the
    tables as they were. Requires, on SQLite, an engine set up by
    `enable_sqlite_explicit_begin`: the driver otherwise begins no
    transaction before the schema changes. Alembic's runtime state is
    process-global, so a process runs one upgrade at a time.
    """
    async with write_connection(engine) as connection:
        await connection.run_sync(_upgrade, lineage)


def _upgrade(connection: Connection, lineage: Lineage) -> None:
    config = _config(lineage)
    config.attributes["connection"] = connection
    command.upgrade(config, "head")


def _config(lineage: Lineage) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ENVIRONMENT))
    config.set_main_option("version_locations", str(lineage.versions))
    config.set_main_option("path_separator", "os")
    config.attributes["lineage"] = lineage
    return config
