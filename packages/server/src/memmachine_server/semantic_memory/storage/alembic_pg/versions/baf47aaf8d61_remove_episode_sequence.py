"""Order semantic history by episode time and UUID.

Revision ID: baf47aaf8d61
Revises: f241a7936b8e
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "baf47aaf8d61"
down_revision: str | Sequence[str] | None = "f241a7936b8e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Remove the obsolete ordering column."""
    op.drop_column("set_ingested_history", "sequence_num")


def downgrade() -> None:
    """Restore the column for older application versions."""
    op.add_column(
        "set_ingested_history",
        sa.Column("sequence_num", sa.BigInteger(), server_default="0", nullable=False),
    )
