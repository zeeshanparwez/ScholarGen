import json
import os

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.database import get_all_profiles, get_profile, upsert_profile

# Project root is 2 levels up from this file (backend/core/ → backend/ → ScholarGen/)
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_FILE_DIR))
load_dotenv(os.path.join(BASE_DIR, "Config", ".env"))

# NIM embeddings — saves Gemini embedding quota (was at 84% of free tier)
NIM_EMBED_MODEL = "baai/bge-m3"
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

from backend.core.key_manager import get_api_key
llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    google_api_key=get_api_key(0),
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
    """Use the primary LLM (Azure → Gemini) to extract {interests, skills} from conversation text."""
    prompt = (
        "Extract user's interests and skills from the following text. "
        "Return ONLY a valid JSON dictionary with keys 'interests' and 'skills'.\n\n"
        f"Text:\n{text}\n\n"
        "Respond with ONLY this format:\n"
        '{"interests": ["AI", "Machine Learning"], "skills": ["Python", "Data Analysis"]}'
    )
    try:
        from backend.core.azure_llm import invoke_azure, is_azure_configured
        if is_azure_configured():
            text_out = invoke_azure(prompt, temperature=0.3, max_tokens=300)
        else:
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
    """Embed all user profiles via NVIDIA NIM (baai/bge-m3).
    Falls back to Gemini key rotation if NIM key is unavailable.
    """
    import logging
    texts = [_profile_to_text(p) for p in profiles]

    nim_key = os.environ.get("NIM_API_KEY", "")
    if nim_key:
        try:
            client = OpenAI(base_url=NIM_BASE_URL, api_key=nim_key)
            response = client.embeddings.create(
                input=texts,
                model=NIM_EMBED_MODEL,
                encoding_format="float",
            )
            return np.array([e.embedding for e in response.data], dtype=np.float32)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "NIM embedding failed, falling back to Gemini: %s", exc
            )

    # Fallback: Gemini embeddings with key rotation
    from google import genai as gai
    from google.genai import types as gtypes
    from backend.core.key_manager import get_all_keys
    keys = get_all_keys()
    last_exc = None
    for i, key in enumerate(keys):
        try:
            gclient = gai.Client(api_key=key)
            result = gclient.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=texts,
                config=gtypes.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                ),
            )
            return np.array([e.values for e in result.embeddings], dtype=np.float32)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if ("503" in msg or "429" in msg) and i < len(keys) - 1:
                logging.getLogger(__name__).warning(
                    "Collab embed key %d failed (%s), trying key %d...", i + 1, exc, i + 2
                )
                continue
            raise
    raise last_exc


# ── Collaboration logic ───────────────────────────────────────────────────────

def _jaccard(set_a: set, set_b: set) -> float:
    a = {s.lower() for s in set_a}
    b = {s.lower() for s in set_b}
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_similar_users(username: str, top_n: int = 5) -> list:
    """
    Return the top_n most similar users as enriched dicts.
    Uses Jaccard similarity on skills (60%) + interests (40%).
    No external API calls — always works.
    """
    profiles = get_all_profiles()
    user_profile = next((p for p in profiles if p["username"] == username), None)
    if not user_profile:
        return []

    user_skills    = set(user_profile.get("skills", []))
    user_interests = set(user_profile.get("interests", []))

    scored = []
    for p in profiles:
        if p["username"] == username:
            continue
        p_skills    = set(p.get("skills", []))
        p_interests = set(p.get("interests", []))

        skill_sim    = _jaccard(user_skills, p_skills)
        interest_sim = _jaccard(user_interests, p_interests)
        score        = 0.6 * skill_sim + 0.4 * interest_sim

        # Shared skills preserving original casing from the match's profile
        shared = [s for s in p_skills if any(s.lower() == u.lower() for u in user_skills)]

        scored.append({
            "username":     p["username"],
            "current_role": p.get("current_role", ""),
            "target_role":  p.get("target_role", ""),
            "shared_skills": shared[:5],
            "score":         round(score * 100),   # 0-100 percent
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def suggest_collaboration_topics(matches: list) -> list:
    """Generate collaboration topic suggestions from matched user profiles."""
    all_shared: list = []
    for m in matches:
        all_shared.extend(m.get("shared_skills", []))

    # Deduplicate preserving order
    seen: set = set()
    unique_shared = [s for s in all_shared if not (s.lower() in seen or seen.add(s.lower()))]

    topics = []
    if unique_shared:
        topics.append(f"Deep-dive on shared skills: {', '.join(unique_shared[:4])}")
    roles = [m["target_role"] for m in matches if m.get("target_role")]
    if roles:
        unique_roles = list(dict.fromkeys(roles))[:3]
        topics.append(f"Career paths to explore together: {', '.join(unique_roles)}")
    if len(matches) >= 2:
        names = [m["username"] for m in matches[:3]]
        topics.append(f"Start a study group with {', '.join(names)}")
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
