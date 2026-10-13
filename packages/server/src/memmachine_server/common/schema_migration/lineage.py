"""The revision history of one store's tables."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import MetaData


@dataclass(frozen=True, kw_only=True)
class Lineage:
    """
    The Alembic revisions that create and evolve one store's tables.

    A database holds the tables of every store configured on it, and each
    store's lineage records its revision in a version table of its own, so
    upgrading one store's tables leaves every other store's untouched.
    """

    name: str
    """Name of the lineage, unique among lineages."""

    versions: Path
    """Directory of the lineage's revision scripts."""

    metadata: MetaData
    """The tables at the newest revision, as the store's code declares them."""

    @property
    def version_table(self) -> str:
        """Name of the table that records the lineage's revision."""
        return f"alembic_version_{self.name}"
