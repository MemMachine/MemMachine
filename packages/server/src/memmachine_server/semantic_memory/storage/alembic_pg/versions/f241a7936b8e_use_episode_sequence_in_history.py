"""Replace semantic batch position with the global episode sequence.

Revision ID: f241a7936b8e
Revises: e8d4a37b69f2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f241a7936b8e"
down_revision: str | Sequence[str] | None = "e8d4a37b69f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Preserve existing order values and allow global sequence growth."""
    op.alter_column(
        "set_ingested_history",
        "batch_position",
        new_column_name="sequence_num",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
        existing_server_default="0",
    )


def downgrade() -> None:
    """Restore the legacy position column."""
    op.alter_column(
        "set_ingested_history",
        "sequence_num",
        new_column_name="batch_position",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
        existing_server_default="0",
    )
