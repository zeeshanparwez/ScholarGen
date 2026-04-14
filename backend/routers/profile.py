import datetime
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI as _NimSync
from pydantic import BaseModel
from typing import List

from backend.dependencies import get_current_user
from backend.core.database import get_profile, upsert_profile, record_activity
from backend.core.key_manager import get_api_key

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT, "Config", ".env"))

logger = logging.getLogger(__name__)
router = APIRouter()

_llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    google_api_key=get_api_key(0),
    temperature=0.7,
    max_retries=1,
    model_kwargs={"generation_config": {"thinking_config": {"thinking_budget": 0}}},
)


def _invoke_llm(prompt: str) -> str:
    try:
        return _llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        msg = str(e)
        if "429" in msg or "503" in msg or "quota" in msg.lower():
            nim_key = os.environ.get("NIM_API_KEY", "")
            if nim_key:
                client = _NimSync(
                    base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=nim_key,
                )
                r = client.chat.completions.create(
                    model=os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                return r.choices[0].message.content.strip()
        raise


class ProfileUpdate(BaseModel):
    interests: List[str] = []
    skills: List[str] = []
    current_role: str = ""
    target_role: str = ""


@router.get("")
async def get_my_profile(username: str = Depends(get_current_user)):
    profile = get_profile(username)
    if not profile:
        return {"username": username, "interests": [], "skills": [], "current_role": "", "target_role": "", "last_updated": None}
    return profile


@router.put("")
async def update_my_profile(body: ProfileUpdate, username: str = Depends(get_current_user)):
    upsert_profile(
        username=username,
        interests=body.interests,
        skills=body.skills,
        current_role=body.current_role,
        target_role=body.target_role,
        last_updated=datetime.datetime.now().isoformat(),
    )
    return {"message": "Profile updated"}


@router.post("/streak")
async def update_streak(username: str = Depends(get_current_user)):
    """Record today's activity and return current streak."""
    streak = record_activity(username)
    return {"streak": streak}


@router.post("/generate-bio")
async def generate_bio(username: str = Depends(get_current_user)):
    """Generate a LinkedIn bio + headline from the user's profile."""
    profile = get_profile(username)
    if not profile or (not profile.get("skills") and not profile.get("current_role")):
        return {"error": "Add some skills and roles to your profile first."}

    skills_str = ", ".join(profile.get("skills", [])[:15]) or "Not listed"
    interests_str = ", ".join(profile.get("interests", [])[:10]) or "Not listed"

    prompt = f"""Create a professional LinkedIn bio and headline for this person.

Current Role: {profile.get('current_role') or 'Not specified'}
Target Role: {profile.get('target_role') or 'Not specified'}
Skills: {skills_str}
Interests: {interests_str}

Return ONLY valid JSON with no markdown:
{{
  "headline": "LinkedIn headline (120 chars max, e.g. Senior Engineer | Building X | Learning Y)",
  "bio": "2-3 paragraph About section for LinkedIn. Professional but human. Show trajectory.",
  "elevator_pitch": "1 sentence elevator pitch they can use anywhere"
}}"""

    try:
        text = _invoke_llm(prompt)
        start, end = text.find("{"), text.rfind("}") + 1
        result = json.loads(text[start:end])
        return result
    except Exception as e:
        logger.error("Bio generation failed: %s", e)
        return {"error": "Generation failed. Please try again."}
