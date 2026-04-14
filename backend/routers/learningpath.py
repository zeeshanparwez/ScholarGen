from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional
import os
import json
import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from openai import OpenAI as _NimSync

from backend.dependencies import get_current_user
from backend.core.course_retriever import CourseRetriever
from backend.core.database import get_profile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT, "Config", ".env"))

logger = logging.getLogger(__name__)
router = APIRouter()

from backend.core.key_manager import get_api_key  # noqa: E402
_llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    google_api_key=get_api_key(0),
    temperature=0.4,
    max_retries=1,   # fail fast so NIM fallback kicks in sooner
    model_kwargs={"generation_config": {"thinking_config": {"thinking_budget": 0}}},
)
_retriever = CourseRetriever()


def _invoke_llm(prompt: str) -> str:
    """Call Gemini; fall back to NIM Llama 3.3 on quota/overload errors."""
    try:
        resp = _llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        msg = str(e)
        if "429" in msg or "503" in msg or "quota" in msg.lower() or "overload" in msg.lower():
            logger.warning("Gemini unavailable (%s), falling back to NIM for learning path", msg[:80])
            nim_key = os.environ.get("NIM_API_KEY", "")
            nim_base = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
            nim_model = os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct")
            if not nim_key:
                raise
            client = _NimSync(base_url=nim_base, api_key=nim_key)
            r = client.chat.completions.create(
                model=nim_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2048,
            )
            return r.choices[0].message.content.strip()
        raise


class LearningPathRequest(BaseModel):
    current_role: str = Field(..., min_length=2, description="Current job role or skill level")
    target_role: str = Field(..., min_length=2, description="Target role or goal")
    job_description: Optional[str] = Field(None, description="Optional JD to extract required skills from")


@router.post("/generate")
async def generate_learning_path(
    body: LearningPathRequest,
    username: str = Depends(get_current_user),
):
    # Pull user's existing skills from profile to personalise the gap analysis
    profile = get_profile(username) or {}
    current_skills = profile.get("skills", [])
    current_interests = profile.get("interests", [])

    profile_context = ""
    if current_skills or current_interests:
        profile_context = f"\nUser's known skills: {', '.join(current_skills)}" if current_skills else ""
        profile_context += f"\nUser's interests: {', '.join(current_interests)}" if current_interests else ""

    jd_context = f"\n\nJob Description provided:\n{body.job_description[:1500]}" if body.job_description else ""

    prompt = f"""You are a career development expert. Create a precise, actionable learning path.

Current role: {body.current_role}
Target role: {body.target_role}{profile_context}{jd_context}

Return ONLY valid JSON in exactly this structure:
{{
  "skill_gaps": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "phases": [
    {{
      "phase": 1,
      "title": "Phase title",
      "duration": "4-6 weeks",
      "skills": ["skill1", "skill2"],
      "description": "One sentence on what this phase achieves"
    }}
  ],
  "search_queries": ["query1 for course search", "query2", "query3"]
}}

Rules:
- 3-5 skill gaps maximum, most impactful ones only
- 3-4 phases that build on each other logically
- search_queries must be short keyword phrases suitable for NPTEL course search
- Be specific to the actual roles, not generic advice"""

    try:
        text = _invoke_llm(prompt)
        start = text.find("{")
        end = text.rfind("}") + 1
        plan = json.loads(text[start:end])
    except Exception as e:
        logger.error("Learning path generation failed: %s", e)
        return {"error": "Could not generate learning path. Please try again."}

    # Fetch real course recommendations for each search query
    courses_by_phase = []
    for query in plan.get("search_queries", [])[:3]:
        try:
            result = _retriever.retrieve(query, top_k=2, min_similarity=0.2)
            payload_start = result.find("<!-- JSON:") + len("<!-- JSON:")
            payload_end = result.find(" -->", payload_start)
            if payload_start > 9 and payload_end > 0:
                payload = json.loads(result[payload_start:payload_end])
                courses_by_phase.extend(payload.get("results", []))
        except Exception:
            pass

    # Deduplicate courses by URL
    seen_urls = set()
    unique_courses = []
    for c in courses_by_phase:
        if c["url"] not in seen_urls:
            seen_urls.add(c["url"])
            unique_courses.append(c)

    return {
        "current_role": body.current_role,
        "target_role": body.target_role,
        "skill_gaps": plan.get("skill_gaps", []),
        "phases": plan.get("phases", []),
        "recommended_courses": unique_courses[:6],
    }
