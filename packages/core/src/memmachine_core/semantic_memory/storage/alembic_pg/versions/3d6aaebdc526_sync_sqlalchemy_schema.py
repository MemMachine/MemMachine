"""
sync_sqlalchemy_schema.

Revision ID: 3d6aaebdc526
Revises: 001
Create Date: 2025-11-04 20:32:38.622715

"""

import contextlib
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# revision identifiers, used by Alembic.
revision: str = "3d6aaebdc526"
down_revision: str | Sequence[str] | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Sync the database schema with the current SQLAlchemy models."""
    # Vector extension (no-op if already installed)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1) prof -> feature (rename), all inside TX
    # Guarded rename to avoid errors if already renamed by earlier runs
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name='prof'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name='feature'
            ) THEN
                ALTER TABLE prof RENAME TO feature;
            END IF;
        END$$;
        """,
    )

    # Column renames & transformations on feature
    with op.batch_alter_table("feature", schema=None) as b:
        # If a column is already renamed, PostgreSQL will error. Use IF EXISTS via two-step try blocks.
        # batch_alter_table doesn't support IF EXISTS directly, so rely on presence tests via SQL.
        b.alter_column("user_id", new_column_name="set_id", existing_type=sa.TEXT())
        b.alter_column("tag", new_column_name="tag_id", existing_type=sa.TEXT())
        b.alter_column(
            "create_at",
            new_column_name="created_at",
            existing_type=pg.TIMESTAMP(timezone=True),
        )
        b.alter_column(
            "update_at",
            new_column_name="updated_at",
            existing_type=pg.TIMESTAMP(timezone=True),
        )
        b.add_column(
            sa.Column(
                "semantic_type_id",
                sa.String(),
                server_default=sa.text("'default'"),
            ),
        )
        b.drop_column("isolations")

    # Type fixes / defaults
    op.alter_column(
        "feature",
        "created_at",
        type_=sa.TIMESTAMP(timezone=False),
        postgresql_using="created_at::timestamp",
        existing_type=pg.TIMESTAMP(timezone=True),
    )
    op.alter_column(
        "feature",
        "updated_at",
        type_=sa.TIMESTAMP(timezone=False),
        postgresql_using="updated_at::timestamp",
        existing_type=pg.TIMESTAMP(timezone=True),
    )
    op.alter_column(
        "feature",
        "metadata",
        type_=pg.JSONB(),
        postgresql_using="metadata::jsonb",
        server_default=sa.text("'{}'::jsonb"),
    )

    # Indexes
    op.execute("DROP INDEX IF EXISTS prof_user_idx")
    op.create_index("idx_feature_set_id", "feature", ["set_id"])
    op.create_index(
        "idx_feature_set_id_semantic_type",
        "feature",
        ["set_id", "semantic_type_id"],
    )
    op.create_index(
        "idx_feature_set_semantic_type_tag",
        "feature",
        ["set_id", "semantic_type_id", "tag_id"],
    )
    op.create_index(
        "idx_feature_set_semantic_type_tag_feature",
        "feature",
        ["set_id", "semantic_type_id", "tag_id", "feature"],
    )

    # 2) history changes
    with op.batch_alter_table("history", schema=None) as b:
        b.alter_column(
            "create_at",
            new_column_name="created_at",
            existing_type=pg.TIMESTAMP(timezone=True),
        )
    op.alter_column(
        "history",
        "created_at",
        type_=sa.TIMESTAMP(timezone=False),
        postgresql_using="created_at::timestamp",
        existing_type=pg.TIMESTAMP(timezone=True),
    )
    op.alter_column(
        "history",
        "metadata",
        type_=pg.JSONB(),
        postgresql_using="metadata::jsonb",
        server_default=sa.text("'{}'::jsonb"),
    )

    # 2a) New join table and backfill
    op.create_table(
        "set_ingested_history",
        sa.Column("set_id", sa.String(), nullable=False),
        sa.Column("history_id", sa.Integer(), nullable=False),
        sa.Column(
            "ingested",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["history_id"],
            ["history.id"],
            ondelete="CASCADE",
            onupdate="CASCADE",
        ),
        sa.PrimaryKeyConstraint("set_id", "history_id"),
    )
    op.execute(
        """
        INSERT INTO set_ingested_history (set_id, history_id, ingested)
        SELECT user_id, id, COALESCE(ingested, false)
        FROM history
        WHERE user_id IS NOT NULL
        ON CONFLICT DO NOTHING
        """,
    )

    # 2b) Drop legacy columns / indexes
    with op.batch_alter_table("history", schema=None) as b:
        b.drop_column("user_id")
        b.drop_column("ingested")
        b.drop_column("isolations")
    op.execute("DROP INDEX IF EXISTS history_user_idx")
    op.execute("DROP INDEX IF EXISTS history_user_ingested_idx")
    op.execute("DROP INDEX IF EXISTS history_user_ingested_ts_desc")

    # 3) citations column renames (preserve data) + rebuild FKs
    with op.batch_alter_table("citations", schema=None) as b:
        # rename legacy columns if present
        with contextlib.suppress(Exception):
            b.alter_column(
                "profile_id",
                new_column_name="feature_id",
                existing_type=sa.Integer,
            )
        with contextlib.suppress(Exception):
            b.alter_column(
                "content_id",
                new_column_name="history_id",
                existing_type=sa.Integer,
            )

    # Drop existing FKs (unknown names) and recreate with explicit names
    op.execute(
        """
        DO $$
        DECLARE r record;
        BEGIN
            FOR r IN
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'citations'::regclass
                  AND contype = 'f'
            LOOP
                EXECUTE format('ALTER TABLE citations DROP CONSTRAINT IF EXISTS %I', r.conname);
            END LOOP;
        END$$;
        """,
    )
    with op.batch_alter_table("citations", schema=None) as b:
        b.create_foreign_key(
            "fk_citations_feature",
            "feature",
            local_cols=["feature_id"],
            remote_cols=["id"],
            ondelete="CASCADE",
            onupdate="CASCADE",
        )
        b.create_foreign_key(
            "fk_citations_history",
            "history",
            local_cols=["history_id"],
            remote_cols=["id"],
            ondelete="CASCADE",
            onupdate="CASCADE",
        )


def downgrade() -> None:
    """Revert the schema changes applied in this migration."""
    # citations back
    with op.batch_alter_table("citations", schema=None) as b:
        with contextlib.suppress(Exception):
            b.drop_constraint("fk_citations_history", type_="foreignkey")
        with contextlib.suppress(Exception):
            b.drop_constraint("fk_citations_feature", type_="foreignkey")
    with op.batch_alter_table("citations", schema=None) as b:
        with contextlib.suppress(Exception):
            b.alter_column(
                "history_id",
                new_column_name="content_id",
                existing_type=sa.Integer,
            )
        with contextlib.suppress(Exception):
            b.alter_column(
                "feature_id",
                new_column_name="profile_id",
                existing_type=sa.Integer,
            )

    # history: add back legacy cols & data
    with op.batch_alter_table("history", schema=None) as b:
        b.add_column(sa.Column("user_id", sa.TEXT(), nullable=True))
        b.add_column(
            sa.Column(
                "ingested",
                sa.Boolean(),
                server_default=sa.text("false"),
                nullable=False,
            ),
        )
        b.add_column(
            sa.Column(
                "isolations",
                pg.JSONB(),
                server_default=sa.text("'{}'::jsonb"),
                nullable=False,
            ),
        )
    op.execute(
        """
        UPDATE history h
        SET user_id = s.set_id,
            ingested = s.ingested
        FROM set_ingested_history s
        WHERE s.history_id = h.id
        """,
    )
    op.execute("CREATE INDEX IF NOT EXISTS history_user_idx ON history (user_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS history_user_ingested_idx ON history (user_id, ingested)",
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS history_user_ingested_ts_desc ON history (user_id, ingested, created_at DESC)",
    )
    op.drop_table("set_ingested_history")

    # history type back
    op.alter_column(
        "history",
        "metadata",
        type_=pg.JSONB(),
        postgresql_using="metadata::jsonb",
        server_default=sa.text("'{}'::jsonb"),
    )
    op.alter_column(
        "history",
        "created_at",
        type_=pg.TIMESTAMP(timezone=True),
        postgresql_using="created_at::timestamptz",
    )
    with op.batch_alter_table("history", schema=None) as b:
        b.alter_column(
            "created_at",
            new_column_name="create_at",
            existing_type=pg.TIMESTAMP(timezone=True),
        )

    # feature -> prof back
    op.alter_column(
        "feature",
        "metadata",
        type_=pg.JSONB(),
        postgresql_using="metadata::jsonb",
        server_default=sa.text("'{}'::jsonb"),
    )
    op.alter_column(
        "feature",
        "updated_at",
        type_=pg.TIMESTAMP(timezone=True),
        postgresql_using="updated_at::timestamptz",
    )
    op.alter_column(
        "feature",
        "created_at",
        type_=pg.TIMESTAMP(timezone=True),
        postgresql_using="created_at::timestamptz",
    )
    with op.batch_alter_table("feature", schema=None) as b:
        b.add_column(
            sa.Column(
                "isolations",
                pg.JSONB(),
                server_default=sa.text("'{}'::jsonb"),
                nullable=False,
            ),
        )
        b.drop_column("semantic_type_id")
        b.alter_column(
            "updated_at",
            new_column_name="update_at",
            existing_type=pg.TIMESTAMP(timezone=True),
        )
        b.alter_column(
            "created_at",
            new_column_name="create_at",
            existing_type=pg.TIMESTAMP(timezone=True),
        )
        b.alter_column("tag_id", new_column_name="tag", existing_type=sa.TEXT())
        b.alter_column("set_id", new_column_name="user_id", existing_type=sa.TEXT())

    op.execute("DROP INDEX IF EXISTS idx_feature_set_id")
    op.execute("DROP INDEX IF EXISTS idx_feature_set_id_semantic_type")
    op.execute("DROP INDEX IF EXISTS idx_feature_set_semantic_type_tag")
    op.execute("DROP INDEX IF EXISTS idx_feature_set_semantic_type_tag_feature")
    op.execute("CREATE INDEX IF NOT EXISTS prof_user_idx ON feature (user_id)")

    # rename back to prof (transaction-safe)
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name='feature'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name='prof'
            ) THEN
                ALTER TABLE feature RENAME TO prof;
            END IF;
        END$$;
        """,
    )
