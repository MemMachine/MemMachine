"""
Add composite index on (set_id, ingested) to set_ingested_history.

Revision ID: c7a2f8e31b90
Revises: b65f7f4a9d2c
Create Date: 2026-03-23 16:22:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7a2f8e31b90"
down_revision: str | Sequence[str] | None = "b65f7f4a9d2c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "ix_set_ingested_history_set_id_ingested",
        "set_ingested_history",
        ["set_id", "ingested"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_set_ingested_history_set_id_ingested",
        table_name="set_ingested_history",
    )
