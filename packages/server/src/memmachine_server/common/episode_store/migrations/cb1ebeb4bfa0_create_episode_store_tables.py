"""Create the episode store's tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "cb1ebeb4bfa0"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
_EPISODE_TYPE = sa.Enum("MESSAGE", name="episode_type")


def upgrade() -> None:
    """Create the tables, and on PostgreSQL the episode type."""
    op.create_table(
        "episodestore",
        sa.Column("uid", sa.Uuid(), autoincrement=False, nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("session_key", sa.String(), nullable=False),
        sa.Column("producer_id", sa.String(), nullable=False),
        sa.Column("producer_role", sa.String(), nullable=False),
        sa.Column("produced_for_id", sa.String(), nullable=True),
        sa.Column("episode_type", _EPISODE_TYPE, nullable=True),
        sa.Column("metadata", _JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("uid"),
    )
    op.create_index("idx_producer_id", "episodestore", ["producer_id"])
    op.create_index("idx_producer_role", "episodestore", ["producer_role"])
    op.create_index("idx_session_key", "episodestore", ["session_key"])
    op.create_index(
        "idx_session_key_producer_id", "episodestore", ["session_key", "producer_id"]
    )
    op.create_index(
        "idx_session_key_producer_id_producer_role_produced_for_id",
        "episodestore",
        ["session_key", "producer_id", "producer_role", "produced_for_id"],
    )


def downgrade() -> None:
    """Drop the tables, and on PostgreSQL the episode type."""
    op.drop_table("episodestore")
    _EPISODE_TYPE.drop(op.get_bind(), checkfirst=True)
