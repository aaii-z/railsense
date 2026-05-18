from __future__ import annotations

import json
import logging
from typing import Any

from db import get_conn, put_conn

log = logging.getLogger(__name__)


def save_turn(
    session_id: str,
    role: str,
    content: str,
    intent: str | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO conversations (session_id, role, content, intent, extras)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, role, content, intent, json.dumps(extras) if extras else None),
            )
            conn.commit()
        finally:
            cur.close()
    except Exception:
        conn.rollback()
        log.exception("Failed to save conversation turn (session=%s, role=%s)", session_id, role)
    finally:
        put_conn(conn)


def get_sessions() -> list[dict]:
    """All sessions, newest first. Uses the first user message as the label."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute("""
                WITH first_user AS (
                    SELECT DISTINCT ON (session_id)
                        session_id, content
                    FROM conversations
                    WHERE role = 'user'
                    ORDER BY session_id, created_at ASC
                ),
                last_active AS (
                    SELECT session_id, MAX(created_at) AS ts
                    FROM conversations
                    GROUP BY session_id
                )
                SELECT fu.session_id, fu.content, la.ts
                FROM first_user fu
                JOIN last_active la ON fu.session_id = la.session_id
                ORDER BY la.ts DESC
            """)
            rows = cur.fetchall()
        finally:
            cur.close()
    finally:
        put_conn(conn)
    return [{"session_id": r[0], "label": r[1], "last_active": r[2]} for r in rows]


def get_session_messages(session_id: str) -> list[dict]:
    """All messages for one session, oldest first."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT role, content, intent, extras, created_at
                FROM conversations
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (session_id,))
            rows = cur.fetchall()
        finally:
            cur.close()
    finally:
        put_conn(conn)
    return [
        {
            "role": r[0],
            "content": r[1],
            "intent": r[2],
            "extras": r[3] or {},
            "created_at": r[4],
        }
        for r in rows
    ]


def ensure_table() -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id         SERIAL PRIMARY KEY,
                    session_id VARCHAR(64)  NOT NULL,
                    role       VARCHAR(16)  NOT NULL,
                    content    TEXT         NOT NULL,
                    intent     VARCHAR(32),
                    extras     JSONB,
                    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                ALTER TABLE conversations
                ADD COLUMN IF NOT EXISTS extras JSONB
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS conversations_session_idx
                ON conversations (session_id)
            """)
            conn.commit()
        finally:
            cur.close()
    finally:
        put_conn(conn)
