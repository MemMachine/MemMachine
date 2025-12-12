"""
Add created at to history add.

Revision ID: 62dff1150a46
Revises: 79f00a9f2409
Create Date: 2025-12-10 08:30:58.962290

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "62dff1150a46"
down_revision: str | Sequence[str] | None = "79f00a9f2409"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "set_ingested_history",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("set_ingested_history", "created_at")
