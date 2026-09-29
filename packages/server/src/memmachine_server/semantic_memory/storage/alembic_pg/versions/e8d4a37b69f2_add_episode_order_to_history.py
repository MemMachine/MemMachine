"""Store episode time and batch position separately from registration time.

Revision ID: e8d4a37b69f2
Revises: c7a2f8e31b90
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8d4a37b69f2"
down_revision: str | Sequence[str] | None = "c7a2f8e31b90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add episode chronology without changing the ingestion debounce clock."""
    op.add_column(
        "set_ingested_history",
        sa.Column("episode_created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "set_ingested_history",
        sa.Column("batch_position", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    """Drop episode chronology fields."""
    op.drop_column("set_ingested_history", "batch_position")
    op.drop_column("set_ingested_history", "episode_created_at")
