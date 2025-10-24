CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS metadata;

--TODO: find a better way to model metadata and isolations than jsonb
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
);

CREATE INDEX IF NOT EXISTS prof_user_idx ON prof (user_id);

CREATE TABLE IF NOT EXISTS history (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    ingested BOOLEAN NOT NULL DEFAULT FALSE,
    content TEXT NOT NULL,
    create_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}',
    isolations JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS history_user_idx ON
    history (user_id);
CREATE INDEX IF NOT EXISTS history_user_ingested_idx ON
    history (user_id, ingested);
CREATE INDEX IF NOT EXISTS history_user_ingested_ts_desc ON
    history (user_id, ingested, create_at DESC);


CREATE TABLE IF NOT EXISTS citations (
    profile_id INTEGER REFERENCES prof(id) ON DELETE CASCADE,
    content_id INTEGER REFERENCES history(id) ON DELETE CASCADE,
    PRIMARY KEY (profile_id, content_id)
);

CREATE TABLE IF NOT EXISTS metadata.migration_tracker (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);