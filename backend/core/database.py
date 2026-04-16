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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookmarks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT NOT NULL REFERENCES users(username),
                content   TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT NOT NULL REFERENCES users(username),
                item_type TEXT NOT NULL,
                item_url  TEXT NOT NULL,
                title     TEXT NOT NULL,
                status    TEXT NOT NULL DEFAULT 'saved',
                timestamp TEXT NOT NULL,
                UNIQUE(username, item_url)
            )
            """
        )
        # Migration: add columns to existing databases that predate this schema
        for col, default in [
            ("current_role",  "''"),
            ("target_role",   "''"),
            ("streak_count",  "0"),
            ("last_active",   "''"),
        ]:
            try:
                dtype = "INTEGER" if col == "streak_count" else "TEXT"
                conn.execute(f"ALTER TABLE user_profiles ADD COLUMN {col} {dtype} NOT NULL DEFAULT {default}")
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


# ── Streak tracker ────────────────────────────────────────────────────────────

def record_activity(username: str) -> int:
    """Record today's activity for the user and return their current streak count."""
    from datetime import date, timedelta
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    with _conn() as conn:
        row = conn.execute(
            "SELECT streak_count, last_active FROM user_profiles WHERE username = ?", (username,)
        ).fetchone()

        if row is None:
            return 0  # User profile not created yet

        last_active = row["last_active"] or ""
        streak = row["streak_count"] or 0

        if last_active == today:
            return streak  # Already recorded today

        if last_active == yesterday:
            streak += 1
        else:
            streak = 1  # Reset if gap > 1 day

        conn.execute(
            "UPDATE user_profiles SET streak_count = ?, last_active = ? WHERE username = ?",
            (streak, today, username),
        )
        return streak


# ── Bookmarks ─────────────────────────────────────────────────────────────────

def add_bookmark(username: str, content: str, timestamp: str) -> int:
    """Save a bookmarked message. Returns the new bookmark id."""
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO bookmarks (username, content, timestamp) VALUES (?, ?, ?)",
            (username, content, timestamp),
        )
        return cur.lastrowid


def get_bookmarks(username: str) -> list:
    """Return all bookmarks for a user, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, content, timestamp FROM bookmarks WHERE username = ? ORDER BY id DESC",
            (username,),
        ).fetchall()
    return [{"id": r["id"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]


def delete_bookmark(bookmark_id: int, username: str) -> bool:
    """Delete a bookmark. Returns True if a row was deleted."""
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM bookmarks WHERE id = ? AND username = ?",
            (bookmark_id, username),
        )
        return cur.rowcount > 0


# ── Progress tracker ───────────────────────────────────────────────────────────

def upsert_progress(username: str, item_type: str, item_url: str, title: str, status: str, timestamp: str) -> int:
    """Insert or update a progress item. Returns the item id."""
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO progress (username, item_type, item_url, title, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, item_url) DO UPDATE SET
                status    = excluded.status,
                timestamp = excluded.timestamp
            """,
            (username, item_type, item_url, title, status, timestamp),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute(
            "SELECT id FROM progress WHERE username = ? AND item_url = ?", (username, item_url)
        ).fetchone()
        return row["id"] if row else -1


def get_progress(username: str) -> list:
    """Return all tracked items for a user, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM progress WHERE username = ? ORDER BY id DESC",
            (username,),
        ).fetchall()
    return [
        {
            "id": r["id"], "item_type": r["item_type"], "item_url": r["item_url"],
            "title": r["title"], "status": r["status"], "timestamp": r["timestamp"],
        }
        for r in rows
    ]


def get_progress_by_url(username: str, item_url: str) -> dict | None:
    """Return a single progress item by URL, or None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM progress WHERE username = ? AND item_url = ?", (username, item_url)
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"], "item_type": row["item_type"], "item_url": row["item_url"],
        "title": row["title"], "status": row["status"], "timestamp": row["timestamp"],
    }


def update_progress_status(progress_id: int, username: str, status: str) -> bool:
    """Update the status of a progress item. Returns True if updated."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE progress SET status = ? WHERE id = ? AND username = ?",
            (status, progress_id, username),
        )
        return cur.rowcount > 0


def delete_progress(progress_id: int, username: str) -> bool:
    """Delete a progress item. Returns True if deleted."""
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM progress WHERE id = ? AND username = ?",
            (progress_id, username),
        )
        return cur.rowcount > 0


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_analytics_data() -> dict:
    """Aggregate actual org-level metrics."""
    from datetime import date, timedelta

    with _conn() as conn:
        real_profiles = conn.execute("SELECT * FROM user_profiles").fetchall()
        progress_rows = conn.execute("SELECT status FROM progress").fetchall()

    today = date.today()

    learners = []
    all_skills = []
    
    for row in real_profiles:
        skills = json.loads(row["skills"] or "[]")
        all_skills.extend(skills)
        learners.append({
            "username":    row["username"],
            "skills":      skills,
            "streak":      row["streak_count"] or 0,
            "target_role": row["target_role"] or "",
            "current_role": row["current_role"] or "",
            "last_active": row["last_active"] or "",
        })

    active_learners = sum(1 for l in learners if l["streak"] > 0)
    avg_streak      = round(sum(l["streak"] for l in learners) / len(learners), 1) if learners else 0

    # ── Skill gap aggregation ─────────────────────────────────────────────────
    # Since we don't have hardcoded demo skill gaps, we can figure out what skills are present
    # Or rely on an external skill taxonomy. For now, we will return top skills present as placeholders for gaps,
    # or identify which users are missing standard skills. Let's return frequency of existing skills as "strengths".
    # For actual gaps, we'd need to compare against target role requirements.
    # To keep the dashboard working, we return a basic structure.
    skill_counts = {}
    for skill in all_skills:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
    # Example gaps calculation based on least common skills
    # (In a real system, you'd compare against target roles)
    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1])
    skill_gaps = [{"skill": s, "count": len(learners) - c} for s, c in sorted_skills[:8]]

    # ── Progress stats ────────────────────────────────────────────────────────
    status_counts = {"saved": 0, "in_progress": 0, "done": 0}
    for row in progress_rows:
        s = row["status"]
        if s in status_counts:
            status_counts[s] += 1

    # ── Leaderboard (top 6 by streak) ────────────────────────────────────────
    leaderboard = sorted(learners, key=lambda x: x["streak"], reverse=True)[:6]

    # ── Activity heatmap (last 7 days) ────────────────────────────────────────
    week_window = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    activity_counts = {d: 0 for d in week_window}
    for l in learners:
        last_active_str = l.get("last_active", "")
        streak = l.get("streak", 0)
        if not last_active_str or not streak:
            continue
        try:
            last_active_date = date.fromisoformat(last_active_str)
        except ValueError:
            continue
        for offset in range(min(streak, 7)):
            active_day = last_active_date - timedelta(days=offset)
            if active_day in activity_counts:
                activity_counts[active_day] += 1
    activity_week = [activity_counts[d] for d in week_window]

    return {
        "total_learners":  len(learners),
        "active_learners": active_learners,
        "avg_streak":      avg_streak,
        "total_skills":    len(all_skills),
        "skill_gaps":      skill_gaps,
        "progress":        status_counts,
        "leaderboard":     leaderboard,
        "activity_week":   activity_week,
        "completion_rate": round(
            status_counts["done"] / max(1, sum(status_counts.values())) * 100
        ),
    }


# Auto-initialise when module is first imported.
init_db()
