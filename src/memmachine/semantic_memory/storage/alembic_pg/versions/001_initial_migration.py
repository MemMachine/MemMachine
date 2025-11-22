"""
Initial migration - original schema without SQLAlchemy.

Revision ID: 001
Revises:
Create Date: 2025-11-04

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply the initial semantic storage schema."""
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create metadata schema
    op.execute("CREATE SCHEMA IF NOT EXISTS metadata")

    # Create prof table (original version)
    op.execute("""
        CREATE TABLE IF NOT EXISTS prof (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            tag TEXT NOT NULL DEFAULT 'Miscellaneous',
            feature TEXT NOT NULL,
            value TEXT NOT NULL,
            create_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            update_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            embedding vector NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}',
            isolations JSONB NOT NULL DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS prof_user_idx ON prof (user_id)")

    # Create history table (original version)
    op.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            ingested BOOLEAN NOT NULL DEFAULT FALSE,
            content TEXT NOT NULL,
            create_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB NOT NULL DEFAULT '{}',
            isolations JSONB NOT NULL DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS history_user_idx ON history (user_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS history_user_ingested_idx ON history (user_id, ingested)",
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS history_user_ingested_ts_desc ON history (user_id, ingested, create_at DESC)",
    )

    # Create citations table
    op.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            profile_id INTEGER REFERENCES prof(id) ON DELETE CASCADE,
            content_id INTEGER REFERENCES history(id) ON DELETE CASCADE,
            PRIMARY KEY (profile_id, content_id)
        )
    """)

    # Create migration tracker table in metadata schema
    op.execute("""
        CREATE TABLE IF NOT EXISTS metadata.migration_tracker (
            version VARCHAR(255) PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def downgrade() -> None:
    """Revert the initial semantic storage schema."""
    op.execute("DROP TABLE IF EXISTS metadata.migration_tracker")
    op.execute("DROP TABLE IF EXISTS citations")
    op.execute("DROP TABLE IF EXISTS history")
    op.execute("DROP TABLE IF EXISTS prof")
    op.execute("DROP SCHEMA IF EXISTS metadata CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector")
