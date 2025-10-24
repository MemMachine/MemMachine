CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS metadata;

CREATE TABLE IF NOT EXISTS semantic (
    id SERIAL PRIMARY KEY,
    set_id TEXT NOT NULL,
    semantic_type TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT 'Miscellaneous',
    feature TEXT NOT NULL,
    value TEXT NOT NULL,
    create_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1536) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS feature_set_idx ON semantic (set_id);
CREATE INDEX IF NOT EXISTS feature_set_semantic_type_idx ON semantic (set_id, semantic_type);
CREATE INDEX IF NOT EXISTS feature_set_semantic_type_tag_idx ON semantic (set_id, semantic_type, tag);
CREATE INDEX IF NOT EXISTS feature_set_semantic_type_tag_feature_idx
    ON semantic (set_id, semantic_type, tag, feature);

CREATE TABLE IF NOT EXISTS history (
    id SERIAL PRIMARY KEY,
    set_id TEXT NOT NULL,
    ingested BOOLEAN NOT NULL DEFAULT FALSE,
    content TEXT NOT NULL,
    create_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS history_user_idx ON
    history (set_id);
CREATE INDEX IF NOT EXISTS history_user_ingested_idx ON
    history (set_id, ingested);
CREATE INDEX IF NOT EXISTS history_user_ts_desc_idx ON
    history (set_id, create_at DESC);


CREATE TABLE IF NOT EXISTS citations (
    semantic_id INTEGER REFERENCES semantic(id) ON DELETE CASCADE,
    content_id INTEGER REFERENCES history(id) ON DELETE CASCADE,
    PRIMARY KEY (semantic_id, content_id)
);

CREATE TABLE IF NOT EXISTS metadata.migration_tracker (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);