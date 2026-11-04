"""Create the sqlite-vec vector store's catalog tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1e49cb8d4c8"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "vector_store_sqlite_vec_pt",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.PrimaryKeyConstraint("vector_store_name", "partition_key"),
        sa.UniqueConstraint("incarnation"),
    )
    op.create_table(
        "vector_store_sqlite_vec_st",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("vector_dimensions", sa.Integer(), nullable=False),
        sa.Column("similarity_metric", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("vector_store_name"),
    )
    op.create_table(
        "vector_store_sqlite_vec_dp",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("property_key", sa.String(length=32), nullable=False),
        sa.Column("property_type", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["vector_store_name"],
            ["vector_store_sqlite_vec_st.vector_store_name"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("vector_store_name", "property_key"),
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("vector_store_sqlite_vec_dp")
    op.drop_table("vector_store_sqlite_vec_st")
    op.drop_table("vector_store_sqlite_vec_pt")
