"""
DB-backed staff authentication with PBKDF2-hashed passwords.

Passwords are stored as 'salt_hex:hash_hex' using PBKDF2-HMAC-SHA256
with 100 000 iterations. The default staff user is seeded automatically
when the table is empty.
"""

import hashlib
import logging
import os
import secrets

from db import get_conn, put_conn

log = logging.getLogger(__name__)

_ITERATIONS = 100_000


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    salt_hex, hash_hex = stored.split(":", 1)
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return secrets.compare_digest(dk.hex(), hash_hex)


def authenticate(username: str, password: str) -> bool:
    conn = get_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT password_hash FROM staff_users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    finally:
        put_conn(conn)

    if not row:
        return False
    return _verify_password(password, row[0])


def ensure_default_user() -> None:
    """Seed the default staff user if the table is empty."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1 FROM staff_users LIMIT 1")
            if cur.fetchone() is not None:
                return
            username = os.environ.get("STAFF_USERNAME", "staff")
            password = os.environ.get("STAFF_PASSWORD", "railsense123")
            cur.execute(
                "INSERT INTO staff_users (username, password_hash) VALUES (%s, %s)",
                (username, _hash_password(password)),
            )
            conn.commit()
            log.info("Seeded default staff user '%s'", username)
        finally:
            cur.close()
    finally:
        put_conn(conn)
