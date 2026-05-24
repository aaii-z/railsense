CREATE TABLE IF NOT EXISTS staff_users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(64) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
