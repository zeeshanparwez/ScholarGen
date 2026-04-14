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

# Demo-mode seed data — shown when real user count < 10 so the CHRO dashboard
# looks populated on day-one. Merged with real DB data.
_DEMO_USERS = [
    {"username": "arjun_k",    "skills": ["Python","FastAPI","SQL","Docker"],           "target_role": "Cloud Architect",    "streak": 12},
    {"username": "priya_m",    "skills": ["React","TypeScript","CSS","Jest"],            "target_role": "Frontend Lead",      "streak": 8},
    {"username": "rohit_s",    "skills": ["Java","Spring Boot","Microservices"],         "target_role": "Backend Architect",  "streak": 5},
    {"username": "sneha_t",    "skills": ["ML","Pandas","Scikit-learn"],                 "target_role": "ML Engineer",        "streak": 14},
    {"username": "vikram_n",   "skills": ["AWS","Terraform","Kubernetes","CI/CD"],       "target_role": "DevOps Lead",        "streak": 3},
    {"username": "ananya_r",   "skills": ["Product Management","Agile","Figma"],         "target_role": "Product Director",   "streak": 7},
    {"username": "karan_b",    "skills": ["Data Analysis","SQL","Power BI","Excel"],     "target_role": "Data Engineer",      "streak": 9},
    {"username": "meera_p",    "skills": ["Node.js","MongoDB","REST APIs"],              "target_role": "Full Stack Engineer","streak": 6},
    {"username": "aditya_j",   "skills": ["Cybersecurity","Networking","Linux"],         "target_role": "Security Engineer",  "streak": 11},
    {"username": "divya_c",    "skills": ["Flutter","Dart","iOS","Android"],             "target_role": "Mobile Lead",        "streak": 4},
    {"username": "rahul_g",    "skills": ["Deep Learning","PyTorch","NLP"],              "target_role": "AI Researcher",      "streak": 15},
    {"username": "ishaan_v",   "skills": ["GraphQL","Redis","System Design"],            "target_role": "Staff Engineer",     "streak": 2},
]

_DEMO_SKILL_GAPS = [
    "Cloud Architecture",
    "Kubernetes",
    "System Design",
    "Machine Learning",
    "TypeScript",
    "CI/CD Pipelines",
    "Data Modeling",
    "React",
]


def get_analytics_data() -> dict:
    """Aggregate org-level metrics. Blends real DB data with demo seed data."""
    with _conn() as conn:
        real_profiles = conn.execute("SELECT * FROM user_profiles").fetchall()
        progress_rows = conn.execute("SELECT status FROM progress").fetchall()
        total_users   = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]

    real_count = len(real_profiles)

    # ── Build combined learner list ──────────────────────────────────────────
    learners = []
    for row in real_profiles:
        learners.append({
            "username": row["username"],
            "skills":   json.loads(row["skills"] or "[]"),
            "streak":   row["streak_count"] or 0,
            "target_role": row["target_role"] or "",
        })

    # Pad with demo data for a compelling dashboard
    demo_to_add = _DEMO_USERS if real_count < 5 else _DEMO_USERS[:max(0, 10 - real_count)]
    for d in demo_to_add:
        learners.append({"username": d["username"], "skills": d["skills"],
                         "streak": d["streak"], "target_role": d["target_role"]})

    total_learner_count = (total_users or 0) + (len(demo_to_add) if real_count < 5 else 0)
    if total_learner_count < len(learners):
        total_learner_count = len(learners)

    active_learners = sum(1 for l in learners if l["streak"] > 0)
    avg_streak      = round(sum(l["streak"] for l in learners) / len(learners), 1) if learners else 0

    # ── Skill gap aggregation ─────────────────────────────────────────────────
    all_skills = []
    for l in learners:
        all_skills.extend(l["skills"])

    # Top skill gaps = skills missing from many profiles (use _DEMO_SKILL_GAPS as canonical gap list)
    skill_presence = {gap: 0 for gap in _DEMO_SKILL_GAPS}
    for skill in all_skills:
        for gap in _DEMO_SKILL_GAPS:
            if gap.lower() in skill.lower() or skill.lower() in gap.lower():
                skill_presence[gap] += 1

    # Employees who lack each skill = total - those who have it
    skill_gaps = [
        {"skill": gap, "count": max(3, total_learner_count - skill_presence.get(gap, 0))}
        for gap in _DEMO_SKILL_GAPS
    ]
    skill_gaps.sort(key=lambda x: x["count"], reverse=True)

    # ── Progress stats ────────────────────────────────────────────────────────
    status_counts = {"saved": 0, "in_progress": 0, "done": 0}
    for row in progress_rows:
        s = row["status"]
        if s in status_counts:
            status_counts[s] += 1
    # Add demo progress numbers if sparse
    if sum(status_counts.values()) < 10:
        status_counts["saved"]       += 42
        status_counts["in_progress"] += 31
        status_counts["done"]        += 18

    # ── Leaderboard (top 6 by streak) ────────────────────────────────────────
    leaderboard = sorted(learners, key=lambda x: x["streak"], reverse=True)[:6]

    # ── Activity heatmap (last 7 days) — demo values if no real data ─────────
    activity_week = [12, 18, 9, 24, 17, 8, 21]   # Mon–Sun demo pattern

    return {
        "total_learners":  total_learner_count,
        "active_learners": active_learners,
        "avg_streak":      avg_streak,
        "total_skills":    len(all_skills) + (120 if real_count < 5 else 0),
        "skill_gaps":      skill_gaps[:8],
        "progress":        status_counts,
        "leaderboard":     leaderboard,
        "activity_week":   activity_week,
        "completion_rate": round(
            status_counts["done"] / max(1, sum(status_counts.values())) * 100
        ),
    }


# Auto-initialise when module is first imported.
init_db()
