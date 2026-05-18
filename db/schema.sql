CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    station     VARCHAR(255),
    region      VARCHAR(100),
    doc_date    VARCHAR(50),
    source_file VARCHAR(255),
    section     VARCHAR(255),
    chunk_text  TEXT,
    embedding   vector(384)
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS documents_station_idx ON documents (station);
CREATE INDEX IF NOT EXISTS documents_region_idx ON documents (region);

CREATE TABLE IF NOT EXISTS conversations (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(64) NOT NULL,
    role        VARCHAR(16) NOT NULL,
    content     TEXT        NOT NULL,
    intent      VARCHAR(32),
    extras      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversations_session_idx ON conversations (session_id);
CREATE INDEX IF NOT EXISTS conversations_created_at_idx ON conversations (created_at);
