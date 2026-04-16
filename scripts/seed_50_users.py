import sqlite3
import bcrypt
import json
import random
import os
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "scholargen.db")

ROLES = [
    ("Backend Engineer", "Cloud Architect"),
    ("Frontend Developer", "Frontend Lead"),
    ("Data Analyst", "Data Engineer"),
    ("ML Engineer", "AI Researcher"),
    ("DevOps Engineer", "Platform Engineer"),
    ("Product Manager", "Product Director"),
    ("Quantitative Analyst", "Quant Lead"),
    ("Risk Analyst", "Risk Manager"),
    ("Security Engineer", "CISO"),
    ("Full Stack Developer", "Engineering Manager")
]

SKILLS_POOL = [
    "Python", "Java", "C++", "Go", "Rust", "JavaScript", "TypeScript", "React", "Node.js",
    "SQL", "PostgreSQL", "MongoDB", "Redis", "Kafka", "Docker", "Kubernetes", "AWS", "Azure",
    "Terraform", "CI/CD", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "NLP",
    "Computer Vision", "LLMs", "Data Analysis", "Pandas", "Spark", "Quantitative Methods",
    "Financial Modeling", "Risk Management", "Cybersecurity", "System Design", "Agile"
]

INTERESTS_POOL = [
    "Generative AI", "Distributed Systems", "Cloud Architecture", "Web Performance",
    "Data Engineering", "Algorithmic Trading", "Blockchain", "DevSecOps", "Leadership",
    "Product Strategy", "UI/UX", "Open Source"
]

COURSES = [
    {"type": "course", "title": "Deep Learning for Computer Vision", "url": "https://nptel.ac.in/courses/106/106/106106224/"},
    {"type": "course", "title": "Reinforcement Learning", "url": "https://nptel.ac.in/courses/106/106/106106143/"},
    {"type": "course", "title": "Blockchain Architecture Design", "url": "https://nptel.ac.in/courses/106/104/106104220/"},
    {"type": "course", "title": "Financial Mathematics", "url": "https://nptel.ac.in/courses/111/104/111104098/"},
    {"type": "course", "title": "Cloud Computing", "url": "https://nptel.ac.in/courses/106/105/106105167/"},
    {"type": "paper",  "title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762"},
    {"type": "paper",  "title": "LoRA: Low-Rank Adaptation", "url": "https://arxiv.org/abs/2106.09685"},
    {"type": "paper",  "title": "Retrieval-Augmented Generation", "url": "https://arxiv.org/abs/2005.11401"}
]

FIRST_NAMES = ["Amit", "Neha", "Rahul", "Priya", "Vikram", "Sneha", "Karan", "Ananya", "Rohan", "Meera", 
               "Arjun", "Pooja", "Siddharth", "Kavya", "Aditya", "Riya", "Varun", "Shruti", "Nikhil", "Aarti",
               "Zeeshan", "Aman", "Tanya", "Sanjay", "Kiran", "Deepak", "Divya", "Gaurav", "Nisha", "Manoj"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Reddy", "Nair", "Rao", "Das",
              "Jain", "Mehta", "Bose", "Chawla", "Iyer", "Pillai", "Shah", "Agarwal", "Bhatia", "Chaudhary"]

def hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def run():
    print("Generating 50 realistic users...")
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    
    today = date.today()
    pw_hash = hash_pw("demo@123")  # All demo users get same password for simplicity
    
    for i in range(50):
        # Generate identity
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        username = f"{first.lower()}_{last.lower()}_{random.randint(10,99)}"
        
        # Generate roles & skills
        current_role, target_role = random.choice(ROLES)
        skills = random.sample(SKILLS_POOL, k=random.randint(4, 8))
        interests = random.sample(INTERESTS_POOL, k=random.randint(2, 4))
        
        # Generate activity
        streak = random.choice([0, 1, 2, 3, 5, 7, 12, 14, 21, 30])
        if streak > 0:
            days_ago = random.choice([0, 1])
        else:
            days_ago = random.randint(2, 10)
            
        last_active = (today - timedelta(days=days_ago)).isoformat()
        
        # 1. Insert User
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)", 
            (username, pw_hash)
        )
        
        # 2. Insert Profile
        conn.execute("""
            INSERT OR IGNORE INTO user_profiles 
            (username, interests, skills, current_role, target_role, last_updated, streak_count, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username, 
            json.dumps(interests), 
            json.dumps(skills), 
            current_role, 
            target_role, 
            today.isoformat(), 
            streak, 
            last_active
        ))
        
        # 3. Add Progress Items
        my_courses = random.sample(COURSES, k=random.randint(1, 4))
        for item in my_courses:
            status = random.choice(["saved", "in_progress", "done"])
            unique_url = f"{item['url']}?u={username}"
            conn.execute("""
                INSERT OR IGNORE INTO progress (username, item_type, item_url, title, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, item["type"], unique_url, item["title"], status, today.isoformat()))

    conn.commit()
    conn.close()
    print("Done! The database now has 50 users for the Org Analytics dashboard.")

if __name__ == "__main__":
    run()
