"""Upgrade a lineage's tables to its newest revision, and check that they are there."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.errors import SchemaMismatchError

from .lineage import Lineage

_ENVIRONMENT = Path(__file__).parent / "environment"


async def upgrade_lineage(engine: AsyncEngine, lineage: Lineage) -> None:
    """
    Upgrade the lineage's tables in the engine's database to its newest revision.

    The upgrade runs in one transaction, so a revision that fails leaves the
    tables as they were. Alembic's runtime state is process-global, so a
    process runs one upgrade at a time.
    """
    async with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            # pysqlite begins a transaction only before a data change, so
            # without this each schema change would commit on its own.
            await connection.exec_driver_sql("BEGIN")
        await connection.run_sync(_upgrade, lineage)


async def check_lineage(engine: AsyncEngine, lineage: Lineage) -> None:
    """
    Check that the lineage's tables in the engine's database are at its newest revision.

    Raises:
        SchemaMismatchError:
            If the tables are missing, at an older revision, or at a
            revision this code does not know.
    """
    async with engine.connect() as connection:
        current = await connection.run_sync(_current_revisions, lineage)
    script = ScriptDirectory.from_config(_config(lineage))
    newest = set(script.get_heads())
    if current == newest:
        return

    location = engine.url.render_as_string(hide_password=True)
    known = {revision.revision for revision in script.walk_revisions()}
    if current - known:
        raise SchemaMismatchError(
            f"The {lineage.name} tables in {location} are at revision "
            f"{', '.join(sorted(current))}, which this server does not know; "
            "the database is newer than this server."
        )
    found = f"at revision {', '.join(sorted(current))}" if current else "missing"
    raise SchemaMismatchError(
        f"The {lineage.name} tables in {location} are {found}, not at revision "
        f"{', '.join(sorted(newest))}. Run `memmachine-server migrate`."
    )


def _upgrade(connection: Connection, lineage: Lineage) -> None:
    config = _config(lineage)
    config.attributes["connection"] = connection
    command.upgrade(config, "head")


def _current_revisions(connection: Connection, lineage: Lineage) -> set[str]:
    context = MigrationContext.configure(
        connection, opts={"version_table": lineage.version_table}
    )
    return set(context.get_current_heads())


def _config(lineage: Lineage) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ENVIRONMENT))
    config.set_main_option("version_locations", str(lineage.versions))
    config.set_main_option("path_separator", "os")
    config.attributes["lineage"] = lineage
    return config
