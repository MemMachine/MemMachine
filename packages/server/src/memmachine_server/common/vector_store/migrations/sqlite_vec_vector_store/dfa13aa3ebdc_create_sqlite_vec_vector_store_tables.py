"""Create the sqlite-vec vector store's catalog table."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "dfa13aa3ebdc"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the table."""
    op.create_table(
        "vector_store_sqlite_vec_pt",
        sa.Column("vector_store_name", sa.String(length=32), nullable=False),
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint("vector_store_name", "partition_key"),
    )


def downgrade() -> None:
    """Drop the table."""
    op.drop_table("vector_store_sqlite_vec_pt")
