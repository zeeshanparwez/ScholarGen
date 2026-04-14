"""
SQLite database layer for ScholarGen.
Replaces Excel-based credential and profile storage.

Tables:
  users         — authentication (username, bcrypt password hash)
  user_profiles — interests, skills, last_updated (stored as JSON arrays)
"""

import json
import os
import sqlite3
from contextlib import contextmanager

import bcrypt

# Project root is 2 levels up from this file (backend/core/ → backend/ → ScholarGen/)
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_FILE_DIR))

# Override with SCHOLARGEN_DB_PATH env var to use ":memory:" or a custom path.
DB_PATH = os.environ.get(
    "SCHOLARGEN_DB_PATH",
    os.path.join(BASE_DIR, "Data", "scholargen.db"),
)


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                username     TEXT PRIMARY KEY REFERENCES users(username),
                interests    TEXT NOT NULL DEFAULT '[]',
                skills       TEXT NOT NULL DEFAULT '[]',
                current_role TEXT NOT NULL DEFAULT '',
                target_role  TEXT NOT NULL DEFAULT '',
                last_updated TEXT
            )
            """
        )
        # Migration: add columns to existing databases that predate this schema
        for col, default in [("current_role", "''"), ("target_role", "''")]:
            try:
                conn.execute(f"ALTER TABLE user_profiles ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
            except Exception:
                pass  # column already exists


@contextmanager
def _conn():
    """Per-call connection with WAL mode for better concurrent reads."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Auth ──────────────────────────────────────────────────────────────────────

def create_user(username: str, password: str):
    """Hash password with bcrypt and insert new user. Returns (success, message)."""
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, pw_hash),
            )
        return True, "Signup successful"
    except sqlite3.IntegrityError:
        return False, "Username already exists"


def verify_user(username: str, password: str):
    """Verify username/password. Returns (success, message)."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return False, "Username not found"
    if bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return True, "Login successful"
    return False, "Incorrect password"


# ── Profiles ──────────────────────────────────────────────────────────────────

def upsert_profile(
    username: str,
    interests: list,
    skills: list,
    last_updated: str,
    current_role: str = "",
    target_role: str = "",
):
    """Atomically insert or update a user's profile."""
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (username, interests, skills, current_role, target_role, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                interests    = excluded.interests,
                skills       = excluded.skills,
                current_role = CASE WHEN excluded.current_role != '' THEN excluded.current_role ELSE current_role END,
                target_role  = CASE WHEN excluded.target_role != '' THEN excluded.target_role ELSE target_role END,
                last_updated = excluded.last_updated
            """,
            (username, json.dumps(interests), json.dumps(skills), current_role, target_role, last_updated),
        )


def get_profile(username: str):
    """Return profile dict for a user, or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM user_profiles WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return None
    return {
        "username": row["username"],
        "interests": json.loads(row["interests"]),
        "skills": json.loads(row["skills"]),
        "current_role": row["current_role"] or "",
        "target_role": row["target_role"] or "",
        "last_updated": row["last_updated"],
    }


def get_all_profiles() -> list:
    """Return all user profiles as a list of dicts."""
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM user_profiles").fetchall()
    return [
        {
            "username": row["username"],
            "interests": json.loads(row["interests"]),
            "skills": json.loads(row["skills"]),
            "current_role": row["current_role"] or "",
            "target_role": row["target_role"] or "",
            "last_updated": row["last_updated"],
        }
        for row in rows
    ]


# Auto-initialise when module is first imported.
init_db()
