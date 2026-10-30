"""Create the SQLite vector store's catalog tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f3def40e31cc"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "vector_store_sqlite_pt",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("index_saved", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("vector_store_name", "partition_key"),
        sa.UniqueConstraint("incarnation"),
    )
    op.create_table(
        "vector_store_sqlite_st",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("vector_dimensions", sa.Integer(), nullable=False),
        sa.Column("similarity_metric", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("vector_store_name"),
    )
    op.create_table(
        "vector_store_sqlite_dp",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("property_key", sa.String(length=32), nullable=False),
        sa.Column("property_type", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["vector_store_name"],
            ["vector_store_sqlite_st.vector_store_name"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("vector_store_name", "property_key"),
    )
    op.create_table(
        "vector_store_sqlite_pd_op",
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("record_row_id", sa.Integer(), nullable=False),
        sa.Column("operation_type", sa.String(length=8), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=True),
        sa.Column("applied", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["incarnation"],
            ["vector_store_sqlite_pt.incarnation"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("incarnation", "record_row_id"),
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("vector_store_sqlite_pd_op")
    op.drop_table("vector_store_sqlite_dp")
    op.drop_table("vector_store_sqlite_st")
    op.drop_table("vector_store_sqlite_pt")
