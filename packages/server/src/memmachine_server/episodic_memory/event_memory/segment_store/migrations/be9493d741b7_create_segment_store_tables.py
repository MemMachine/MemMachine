"""Create the segment store's tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "be9493d741b7"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    """Create the tables."""
    op.create_table(
        "segment_store_pt",
        sa.Column("partition_uuid", sa.Uuid(), nullable=False),
        sa.Column(
            "payload_codec_config",
            _JSON,
            nullable=False,
        ),
        sa.Column("purged_through", sa.Uuid(), nullable=True),
        sa.PrimaryKeyConstraint("partition_uuid"),
    )
    op.create_table(
        "segment_store_sg",
        sa.Column("partition_uuid", sa.Uuid(), nullable=False),
        sa.Column("uuid", sa.Uuid(), nullable=False),
        sa.Column("event_uuid", sa.Uuid(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("offset", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timestamp_timezone_offset", sa.Integer(), nullable=False),
        sa.Column("context", sa.LargeBinary(), nullable=False),
        sa.Column("block", sa.LargeBinary(), nullable=False),
        sa.Column(
            "properties",
            _JSON,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("partition_uuid", "uuid"),
    )
    op.create_index(
        "segment_store_sg__pu_ev", "segment_store_sg", ["partition_uuid", "event_uuid"]
    )
    op.create_index(
        "segment_store_sg__pu_ts_ev_ix_of",
        "segment_store_sg",
        ["partition_uuid", "timestamp", "event_uuid", "index", "offset"],
    )

    op.create_table(
        "segment_store_dv_ln",
        sa.Column("partition_uuid", sa.Uuid(), nullable=False),
        sa.Column("uuid", sa.Uuid(), nullable=False),
        sa.Column("segment_uuid", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["partition_uuid", "segment_uuid"],
            ["segment_store_sg.partition_uuid", "segment_store_sg.uuid"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("partition_uuid", "uuid"),
    )
    op.create_index(
        "segment_store_dv_ln__pu_su",
        "segment_store_dv_ln",
        ["partition_uuid", "segment_uuid"],
    )


def downgrade() -> None:
    """Drop the tables."""
    op.drop_table("segment_store_dv_ln")
    op.drop_table("segment_store_sg")
    op.drop_table("segment_store_pt")
