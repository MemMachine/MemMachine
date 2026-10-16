"""Create the SQLite vector store's catalog tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "bf37d0750a77"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "vector_store_sqlite_pt",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.Column("index_saved", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("vector_store_name", "partition_key"),
    )
    op.create_table(
        "vector_store_sqlite_pd_op",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.Column("record_row_id", sa.Integer(), nullable=False),
        sa.Column("operation_type", sa.String(length=8), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=True),
        sa.Column("applied", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["vector_store_name", "partition_key"],
            [
                "vector_store_sqlite_pt.vector_store_name",
                "vector_store_sqlite_pt.partition_key",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("vector_store_name", "partition_key", "record_row_id"),
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("vector_store_sqlite_pd_op")
    op.drop_table("vector_store_sqlite_pt")
