"""
Demo data seeder for UpskillOS.
Run from ScholarGen/ root:  python seed_demo.py

- Removes all test/probe/temp users
- Keeps real users (zeeshanparwez, aman, kumarineha)
- Creates 10 realistic demo employees with full profiles
- Password pattern: username@123  (e.g. arjun_k@123)
"""

import json
import sqlite3
import bcrypt
from datetime import date, timedelta
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "Data", "scholargen.db")

# ── Users to DELETE (test/probe/garbage + old abbreviated names) ──────────────
REMOVE_USERS = {
    "_probe2", "debuguser", "debuguser2", "demouser",
    "qatest99", "testuser", "testuser2", "zee_test_01",
    # old abbreviated demo names
    "arjun_k", "priya_m", "rohit_s", "sneha_t", "vikram_n",
    "ananya_r", "karan_b", "meera_p", "rahul_g", "ishaan_v",
}

# ── Users to KEEP as-is ──────────────────────────────────────────────────────
KEEP_USERS = {"zeeshanparwez", "aman", "kumarineha"}

# ── 10 demo employees — real Indian names ────────────────────────────────────
DEMO_EMPLOYEES = [
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

# Progress items per employee (realistic learning activity)
PROGRESS_TEMPLATES = [
    {"item_type": "course", "title": "Cloud Architecture Fundamentals", "item_url": "https://nptel.ac.in/courses/106/105/106105233/", "status": "in_progress"},
    {"item_type": "course", "title": "Machine Learning with Python",    "item_url": "https://nptel.ac.in/courses/106/106/106106139/", "status": "done"},
    {"item_type": "course", "title": "Kubernetes for Developers",       "item_url": "https://nptel.ac.in/courses/106/105/106105244/", "status": "saved"},
    {"item_type": "paper",  "title": "Attention Is All You Need",       "item_url": "https://arxiv.org/abs/1706.03762",               "status": "done"},
    {"item_type": "course", "title": "System Design Masterclass",       "item_url": "https://nptel.ac.in/courses/106/105/106105031/", "status": "in_progress"},
    {"item_type": "course", "title": "Data Engineering with Apache Spark","item_url": "https://nptel.ac.in/courses/106/105/106105265/","status": "saved"},
    {"item_type": "paper",  "title": "MapReduce: Simplified Data Processing","item_url": "https://arxiv.org/abs/1912.09803",          "status": "done"},
    {"item_type": "course", "title": "TypeScript Deep Dive",            "item_url": "https://nptel.ac.in/courses/106/106/106106097/", "status": "in_progress"},
]


def hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def run():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    # ── Step 1: Remove test users ─────────────────────────────────────────────
    print("Removing test users...")
    for username in REMOVE_USERS:
        conn.execute("DELETE FROM progress     WHERE username = ?", (username,))
        conn.execute("DELETE FROM bookmarks    WHERE username = ?", (username,))
        conn.execute("DELETE FROM user_profiles WHERE username = ?", (username,))
        conn.execute("DELETE FROM users         WHERE username = ?", (username,))
        print(f"  ✗ removed: {username}")

    # ── Step 2: Create demo employees ────────────────────────────────────────
    print("\nCreating demo employees...")
    today = date.today()

    for emp in DEMO_EMPLOYEES:
        uname   = emp["username"]
        password = f"{uname}@123"
        pw_hash  = hash_pw(password)

        last_active = (today - timedelta(days=emp["days_ago"])).isoformat()

        # Upsert user
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
            (uname, pw_hash)
        )
        # Update password in case user already existed
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (pw_hash, uname)
        )

        # Upsert profile
        conn.execute("""
            INSERT INTO user_profiles
              (username, interests, skills, current_role, target_role, last_updated, streak_count, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
              interests    = excluded.interests,
              skills       = excluded.skills,
              current_role = excluded.current_role,
              target_role  = excluded.target_role,
              last_updated = excluded.last_updated,
              streak_count = excluded.streak_count,
              last_active  = excluded.last_active
        """, (
            uname,
            json.dumps(emp["interests"]),
            json.dumps(emp["skills"]),
            emp["current_role"],
            emp["target_role"],
            today.isoformat(),
            emp["streak"],
            last_active,
        ))

        # Add 2-3 progress items per employee (rotate through templates)
        start_idx = DEMO_EMPLOYEES.index(emp) % len(PROGRESS_TEMPLATES)
        for i in range(3):
            item = PROGRESS_TEMPLATES[(start_idx + i) % len(PROGRESS_TEMPLATES)]
            # make URL unique per user so UNIQUE(username, item_url) doesn't conflict
            unique_url = f"{item['item_url']}?u={uname}"
            conn.execute("""
                INSERT OR IGNORE INTO progress (username, item_type, item_url, title, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (uname, item["item_type"], unique_url, item["title"], item["status"], today.isoformat()))

        print(f"  ✓ {uname:15} | {emp['current_role']:30} → {emp['target_role']:25} | streak={emp['streak']}d | pw={password}")

    conn.commit()
    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Demo credentials (for CHRO demo):")
    print(f"{'─'*60}")
    for emp in DEMO_EMPLOYEES:
        print(f"  {emp['username']:15}  /  {emp['username']}@123")
    print(f"{'─'*60}")
    print("\nRun the app and log in with any of these accounts.")
    print("The Org Analytics dashboard will show all 10 employees.\n")


if __name__ == "__main__":
    run()
