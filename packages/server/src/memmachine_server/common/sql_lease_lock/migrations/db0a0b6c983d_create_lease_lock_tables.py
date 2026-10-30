"""Create the SQL lease lock's tables and its fencing counter."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "db0a0b6c983d"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables, and the counter's one row."""
    op.create_table(
        "lease_lock",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("lease_id", sa.String(), nullable=True),
        sa.Column("expires_at_ms", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )
    counter = op.create_table(
        "lease_lock_counter",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("value", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.bulk_insert(counter, [{"id": 1, "value": 0}])


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("lease_lock_counter")
    op.drop_table("lease_lock")
