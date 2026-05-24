"""
Lightweight migration runner.

Tracks applied migrations in a `schema_migrations` table.
Migrations are numbered SQL files in db/migrations/ (e.g. 0001_initial_schema.sql).

Usage:
    python -m db.migrate          # apply all pending migrations
    python -m db.migrate --status # show applied/pending
"""

import argparse
import sys
from pathlib import Path

import psycopg2

from db import DB_URL

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _ensure_tracking_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version  VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    conn.commit()


def _applied_versions(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations ORDER BY version")
        return {row[0] for row in cur.fetchall()}


def _pending_migrations(applied: set[str]) -> list[Path]:
    if not _MIGRATIONS_DIR.is_dir():
        return []
    all_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    return [f for f in all_files if f.name not in applied]


def migrate() -> None:
    conn = psycopg2.connect(DB_URL)
    try:
        _ensure_tracking_table(conn)
        applied = _applied_versions(conn)
        pending = _pending_migrations(applied)

        if not pending:
            print("Database is up to date.")
            return

        for path in pending:
            print(f"Applying {path.name} ... ", end="", flush=True)
            sql = path.read_text()
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (path.name,),
                )
            conn.commit()
            print("done")

        print(f"\n{len(pending)} migration(s) applied.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def status() -> None:
    conn = psycopg2.connect(DB_URL)
    try:
        _ensure_tracking_table(conn)
        applied = _applied_versions(conn)
        pending = _pending_migrations(applied)

        print("Applied:")
        for v in sorted(applied):
            print(f"  [x] {v}")
        print("Pending:")
        if pending:
            for p in pending:
                print(f"  [ ] {p.name}")
        else:
            print("  (none)")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        migrate()
