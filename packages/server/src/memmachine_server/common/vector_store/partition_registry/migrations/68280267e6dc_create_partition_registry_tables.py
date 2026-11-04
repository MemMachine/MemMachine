"""Create the vector store partition registry's tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "68280267e6dc"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "partition_registry_gc",
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts_without_progress", sa.Integer(), nullable=False),
        sa.Column("last_failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claim_generation", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("incarnation"),
    )
    op.create_index(
        "partition_registry_gc__vs_ea",
        "partition_registry_gc",
        ["vector_store_name", "enqueued_at"],
    )
    op.create_table(
        "partition_registry_pt",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("live", sa.Boolean(), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("vector_store_name", "partition_key"),
        sa.UniqueConstraint("incarnation"),
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("partition_registry_pt")
    op.drop_table("partition_registry_gc")
