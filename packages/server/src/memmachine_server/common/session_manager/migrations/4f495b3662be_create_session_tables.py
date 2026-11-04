"""Create the session store's tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "4f495b3662be"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "sessions",
        sa.Column("session_key", sa.String(), nullable=False),
        sa.Column("timestamp", sa.Integer(), nullable=False),
        sa.Column("configuration", _JSON, nullable=False),
        sa.Column("param_data", _JSON, nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("user_metadata", _JSON, nullable=False),
        sa.Column(
            "status", sa.String(), server_default=sa.text("'active'"), nullable=False
        ),
        sa.PrimaryKeyConstraint("session_key"),
    )
    op.create_table(
        "short_term_memory_data",
        sa.Column("session_key", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("last_seq", sa.Integer(), nullable=False),
        sa.Column("episode_num", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["session_key"], ["sessions.session_key"]),
        sa.PrimaryKeyConstraint("session_key"),
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("short_term_memory_data")
    op.drop_table("sessions")
