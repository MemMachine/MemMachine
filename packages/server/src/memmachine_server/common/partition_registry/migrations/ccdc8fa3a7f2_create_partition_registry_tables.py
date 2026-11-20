"""Create the partition registry's tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "ccdc8fa3a7f2"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "partition_registry_gc",
        sa.Column("partition_uuid", sa.Uuid(), nullable=False),
        sa.Column("partition_namespace", sa.String(length=32), nullable=False),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts_without_progress", sa.Integer(), nullable=False),
        sa.Column("claimable_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claim_generation", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("partition_uuid"),
    )
    op.create_index(
        "partition_registry_gc__ns_ea",
        "partition_registry_gc",
        ["partition_namespace", "enqueued_at"],
    )

    op.create_table(
        "partition_registry_pt",
        sa.Column("partition_namespace", sa.String(length=32), nullable=False),
        sa.Column("partition_name", sa.String(length=255), nullable=False),
        sa.Column("partition_uuid", sa.Uuid(), nullable=False),
        sa.Column("live", sa.Boolean(), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("partition_namespace", "partition_name"),
        sa.UniqueConstraint("partition_uuid"),
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("partition_registry_pt")
    op.drop_table("partition_registry_gc")
