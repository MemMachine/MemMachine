"""Revision histories of stores' tables, upgraded by the setup step."""

from .lineage import Lineage
from .runner import upgrade_lineage

__all__ = ["Lineage", "upgrade_lineage"]
