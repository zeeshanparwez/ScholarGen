import json
import os

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.database import get_all_profiles, get_profile, upsert_profile

# Project root is 2 levels up from this file (backend/core/ → backend/ → ScholarGen/)
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_FILE_DIR))
load_dotenv(os.path.join(BASE_DIR, "Config", ".env"))

embedding_model = SentenceTransformer(
    os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
)

llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    temperature=0.7,
    max_retries=2,
    model_kwargs={
        "generation_config": {
            "thinking_config": {"thinking_budget": 0}
        }
    },
)


# ── Profile extraction ────────────────────────────────────────────────────────

def extract_profile_from_text(text: str) -> dict:
    """Use Gemini to extract {interests, skills} as lists from conversation text."""
    prompt = (
        "Extract user's interests and skills from the following text. "
        "Return ONLY a valid JSON dictionary with keys 'interests' and 'skills'.\n\n"
        f"Text:\n{text}\n\n"
        "Respond with ONLY this format:\n"
        '{"interests": ["AI", "Machine Learning"], "skills": ["Python", "Data Analysis"]}'
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text_out = response.content.strip()
        start = text_out.find("{")
        end = text_out.rfind("}") + 1
        json_str = text_out[start:end] if start != -1 and end != 0 else text_out
        profile = json.loads(json_str)
        profile.setdefault("interests", [])
        profile.setdefault("skills", [])
        return profile
    except Exception:
        return {"interests": [], "skills": []}


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _profile_to_text(profile: dict) -> str:
    interests = profile.get("interests", [])
    skills = profile.get("skills", [])
    return " ".join(interests) + " " + " ".join(skills)


def _compute_embeddings(profiles: list) -> np.ndarray:
    texts = [_profile_to_text(p) for p in profiles]
    return embedding_model.encode(texts, convert_to_numpy=True)


# ── Collaboration logic ───────────────────────────────────────────────────────

def match_similar_users(username: str, top_n: int = 3) -> list:
    """Return usernames of the top_n most similar users based on profile embeddings."""
    profiles = get_all_profiles()
    if not any(p["username"] == username for p in profiles):
        return []

    embeddings = _compute_embeddings(profiles)
    user_idx = next(i for i, p in enumerate(profiles) if p["username"] == username)
    sims = cosine_similarity([embeddings[user_idx]], embeddings)[0]

    ranked = sorted(
        [(i, s) for i, s in enumerate(sims) if i != user_idx],
        key=lambda x: x[1],
        reverse=True,
    )
    return [profiles[i]["username"] for i, _ in ranked[:top_n]]


def suggest_collaboration_topics(usernames: list) -> list:
    """Return simple collaboration topics aggregated from matched users' profiles."""
    all_profiles = get_all_profiles()
    profile_map = {p["username"]: p for p in all_profiles}

    all_interests: set = set()
    all_skills: set = set()

    for username in usernames:
        p = profile_map.get(username)
        if p:
            all_interests.update(p["interests"])
            all_skills.update(p["skills"])

    topics = []
    if all_interests:
        topics.append(f"Interests: {', '.join(list(all_interests)[:3])}")
    if all_skills:
        topics.append(f"Skills: {', '.join(list(all_skills)[:3])}")
    if usernames:
        topics.append(f"Study group: {', '.join(usernames)}")
    return topics


def update_user_profile_with_timestamp(username: str, profile: dict, timestamp: str):
    """Thin wrapper kept for backward compatibility. Delegates to database.upsert_profile."""
    upsert_profile(
        username=username,
        interests=profile.get("interests", []),
        skills=profile.get("skills", []),
        last_updated=timestamp,
    )


if __name__ == "__main__":
    matched = match_similar_users("test_user")
    print(f"Matched Users: {matched}")
    topics = suggest_collaboration_topics(matched)
    print(f"Collaboration Topics: {topics}")
