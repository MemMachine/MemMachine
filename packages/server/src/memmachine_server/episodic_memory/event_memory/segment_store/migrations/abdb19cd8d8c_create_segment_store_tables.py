"""Create the segment store's tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "abdb19cd8d8c"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "segment_store_gc",
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("purged_through", sa.Uuid(), nullable=True),
        sa.PrimaryKeyConstraint("incarnation"),
    )
    op.create_index("segment_store_gc__ea", "segment_store_gc", ["enqueued_at"])
    op.create_table(
        "segment_store_pt",
        sa.Column("partition_key", sa.String(length=255), nullable=False),
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("payload_codec_config", _JSON, nullable=False),
        sa.PrimaryKeyConstraint("partition_key"),
        sa.UniqueConstraint("incarnation"),
    )
    op.create_table(
        "segment_store_sg",
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("uuid", sa.Uuid(), nullable=False),
        sa.Column("event_uuid", sa.Uuid(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("offset", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timestamp_timezone_offset", sa.Integer(), nullable=False),
        sa.Column("context", sa.LargeBinary(), nullable=False),
        sa.Column("block", sa.LargeBinary(), nullable=False),
        sa.Column("properties", _JSON, nullable=False),
        sa.PrimaryKeyConstraint("incarnation", "uuid"),
    )
    op.create_index(
        "segment_store_sg__in_ev", "segment_store_sg", ["incarnation", "event_uuid"]
    )
    op.create_index(
        "segment_store_sg__in_ts_ev_ix_of",
        "segment_store_sg",
        ["incarnation", "timestamp", "event_uuid", "index", "offset"],
    )
    op.create_table(
        "segment_store_dv_ln",
        sa.Column("incarnation", sa.Uuid(), nullable=False),
        sa.Column("uuid", sa.Uuid(), nullable=False),
        sa.Column("segment_uuid", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["incarnation", "segment_uuid"],
            ["segment_store_sg.incarnation", "segment_store_sg.uuid"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("incarnation", "uuid"),
    )
    op.create_index(
        "segment_store_dv_ln__in_su",
        "segment_store_dv_ln",
        ["incarnation", "segment_uuid"],
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("segment_store_dv_ln")
    op.drop_table("segment_store_sg")
    op.drop_table("segment_store_pt")
    op.drop_table("segment_store_gc")
