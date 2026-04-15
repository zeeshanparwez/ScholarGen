"""
Startup seeder — ensures demo employees exist in the DB on every deploy.
Called automatically from the FastAPI lifespan hook.

Safe to run multiple times: all writes use INSERT OR IGNORE / ON CONFLICT DO UPDATE.
"""

import json
import logging
from datetime import date, timedelta

import bcrypt

from backend.core.database import _conn

logger = logging.getLogger(__name__)

_DEMO_EMPLOYEES = [
    {
        "username":     "arjun_kumar",
        "current_role": "Backend Engineer",
        "target_role":  "Cloud Architect",
        "skills":       ["Python", "FastAPI", "SQL", "Docker", "REST APIs", "PostgreSQL"],
        "interests":    ["Cloud Computing", "Distributed Systems", "DevOps"],
        "streak":       12,
        "days_ago":     0,
    },
    {
        "username":     "priya_mehta",
        "current_role": "Frontend Developer",
        "target_role":  "Frontend Lead",
        "skills":       ["React", "TypeScript", "CSS", "Jest", "Figma", "Next.js"],
        "interests":    ["UI/UX Design", "Web Performance", "Accessibility"],
        "streak":       8,
        "days_ago":     0,
    },
    {
        "username":     "rohit_sharma",
        "current_role": "Senior Java Developer",
        "target_role":  "Backend Architect",
        "skills":       ["Java", "Spring Boot", "Microservices", "Kafka", "MySQL"],
        "interests":    ["System Design", "Distributed Systems", "API Design"],
        "streak":       5,
        "days_ago":     1,
    },
    {
        "username":     "sneha_tiwari",
        "current_role": "Data Analyst",
        "target_role":  "ML Engineer",
        "skills":       ["Python", "Pandas", "Scikit-learn", "SQL", "Tableau"],
        "interests":    ["Machine Learning", "Deep Learning", "Data Visualization"],
        "streak":       14,
        "days_ago":     0,
    },
    {
        "username":     "vikram_nair",
        "current_role": "DevOps Engineer",
        "target_role":  "DevOps Lead",
        "skills":       ["AWS", "Terraform", "Kubernetes", "CI/CD", "Linux", "Docker"],
        "interests":    ["Cloud Infrastructure", "Automation", "Site Reliability"],
        "streak":       3,
        "days_ago":     2,
    },
    {
        "username":     "ananya_reddy",
        "current_role": "Product Manager",
        "target_role":  "Product Director",
        "skills":       ["Product Management", "Agile", "Figma", "SQL", "User Research"],
        "interests":    ["Product Strategy", "Growth", "Analytics"],
        "streak":       7,
        "days_ago":     0,
    },
    {
        "username":     "karan_bhatia",
        "current_role": "Data Analyst",
        "target_role":  "Data Engineer",
        "skills":       ["SQL", "Power BI", "Excel", "Python", "Data Analysis"],
        "interests":    ["Data Engineering", "ETL Pipelines", "Business Intelligence"],
        "streak":       9,
        "days_ago":     1,
    },
    {
        "username":     "meera_pillai",
        "current_role": "Full Stack Developer",
        "target_role":  "Engineering Manager",
        "skills":       ["Node.js", "MongoDB", "React", "REST APIs", "AWS", "Git"],
        "interests":    ["Leadership", "Team Building", "System Architecture"],
        "streak":       6,
        "days_ago":     0,
    },
    {
        "username":     "rahul_gupta",
        "current_role": "ML Engineer",
        "target_role":  "AI Researcher",
        "skills":       ["Deep Learning", "PyTorch", "NLP", "Python", "TensorFlow", "LLMs"],
        "interests":    ["Generative AI", "Computer Vision", "Research Papers"],
        "streak":       15,
        "days_ago":     0,
    },
    {
        "username":     "ishaan_verma",
        "current_role": "Software Engineer",
        "target_role":  "Staff Engineer",
        "skills":       ["GraphQL", "Redis", "System Design", "Python", "TypeScript"],
        "interests":    ["Distributed Systems", "API Design", "Performance Engineering"],
        "streak":       11,
        "days_ago":     1,
    },
]

_PROGRESS_TEMPLATES = [
    {"item_type": "course", "title": "Cloud Architecture Fundamentals",       "item_url": "https://nptel.ac.in/courses/106/105/106105233/", "status": "in_progress"},
    {"item_type": "course", "title": "Machine Learning with Python",           "item_url": "https://nptel.ac.in/courses/106/106/106106139/", "status": "done"},
    {"item_type": "course", "title": "Kubernetes for Developers",              "item_url": "https://nptel.ac.in/courses/106/105/106105244/", "status": "saved"},
    {"item_type": "paper",  "title": "Attention Is All You Need",              "item_url": "https://arxiv.org/abs/1706.03762",               "status": "done"},
    {"item_type": "course", "title": "System Design Masterclass",              "item_url": "https://nptel.ac.in/courses/106/105/106105031/", "status": "in_progress"},
    {"item_type": "course", "title": "Data Engineering with Apache Spark",     "item_url": "https://nptel.ac.in/courses/106/105/106105265/", "status": "saved"},
    {"item_type": "paper",  "title": "MapReduce: Simplified Data Processing",  "item_url": "https://arxiv.org/abs/1912.09803",               "status": "done"},
    {"item_type": "course", "title": "TypeScript Deep Dive",                   "item_url": "https://nptel.ac.in/courses/106/106/106106097/", "status": "in_progress"},
]


def _hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def seed_demo_users() -> None:
    """
    Upsert the 10 demo employees.  Skips any that already exist with correct data.
    Idempotent — safe to call on every startup.
    """
    today = date.today()

    with _conn() as conn:
        existing = {
            r[0] for r in conn.execute("SELECT username FROM users").fetchall()
        }
        demo_usernames = {e["username"] for e in _DEMO_EMPLOYEES}
        missing = demo_usernames - existing

        if not missing:
            logger.info("Seeder: all demo users present — skipping")
            return

        logger.info("Seeder: creating %d missing demo user(s): %s", len(missing), sorted(missing))

        for emp in _DEMO_EMPLOYEES:
            uname       = emp["username"]
            password    = f"{uname}@123"
            pw_hash     = _hash_pw(password)
            last_active = (today - timedelta(days=emp["days_ago"])).isoformat()

            conn.execute(
                "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
                (uname, pw_hash),
            )

            conn.execute(
                """
                INSERT INTO user_profiles
                  (username, interests, skills, current_role, target_role,
                   last_updated, streak_count, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                  interests    = excluded.interests,
                  skills       = excluded.skills,
                  current_role = excluded.current_role,
                  target_role  = excluded.target_role,
                  last_updated = excluded.last_updated,
                  streak_count = excluded.streak_count,
                  last_active  = excluded.last_active
                """,
                (
                    uname,
                    json.dumps(emp["interests"]),
                    json.dumps(emp["skills"]),
                    emp["current_role"],
                    emp["target_role"],
                    today.isoformat(),
                    emp["streak"],
                    last_active,
                ),
            )

            # 3 progress items per employee
            start_idx = _DEMO_EMPLOYEES.index(emp) % len(_PROGRESS_TEMPLATES)
            for i in range(3):
                item       = _PROGRESS_TEMPLATES[(start_idx + i) % len(_PROGRESS_TEMPLATES)]
                unique_url = f"{item['item_url']}?u={uname}"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO progress
                      (username, item_type, item_url, title, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (uname, item["item_type"], unique_url,
                     item["title"], item["status"], today.isoformat()),
                )

    logger.info("Seeder: demo users ready")
