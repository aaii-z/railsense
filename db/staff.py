import hashlib
import logging
import os
import secrets

from db import db_cursor

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
    with db_cursor() as cur:
        cur.execute(
            "SELECT password_hash FROM staff_users WHERE username = %s",
            (username,),
        )
        row = cur.fetchone()
    if not row:
        return False
    return _verify_password(password, row[0])


def ensure_default_user() -> None:
    """Seed the default staff user if the table is empty."""
    with db_cursor(commit=True) as cur:
        cur.execute("SELECT 1 FROM staff_users LIMIT 1")
        if cur.fetchone() is not None:
            return
        username = os.environ.get("STAFF_USERNAME", "staff")
        password = os.environ.get("STAFF_PASSWORD", "railsense123")
        cur.execute(
            "INSERT INTO staff_users (username, password_hash) VALUES (%s, %s)",
            (username, _hash_password(password)),
        )
    log.info("Seeded default staff user '%s'", username)
