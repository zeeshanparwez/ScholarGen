import asyncio

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

from backend.core.key_manager import get_all_keys  # noqa: E402
_retriever = CourseRetriever()


def _gemini_fallback(prompt: str) -> str:
    """Try each Gemini key in order; raises if all are exhausted."""
    keys = get_all_keys()
    if not keys:
        raise RuntimeError("No GOOGLE_API_KEY configured")
    last_exc: Exception = RuntimeError("No keys")
    for key in keys:
        try:
            llm = ChatGoogleGenerativeAI(
                model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
                google_api_key=key,
                temperature=0.4,
                max_retries=0,
                model_kwargs={"generation_config": {"thinking_config": {"thinking_budget": 0}}},
            )
            return llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            last_exc = e
            if "429" in str(e):
                logger.warning("Gemini key exhausted (429), trying next key...")
                continue
            raise
    raise last_exc


def _invoke_llm(prompt: str, max_tokens: int = 2048) -> str:
    """Azure OpenAI (primary) → NIM (secondary) → Gemini fallback."""
    from backend.core.azure_llm import invoke_azure, is_azure_configured
    if is_azure_configured():
        try:
            return invoke_azure(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.warning("Azure OpenAI failed, falling back to NIM: %s", str(e)[:120])

    nim_key = os.environ.get("NIM_API_KEY", "")
    if nim_key:
        try:
            client = _NimSync(
                base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                api_key=nim_key,
                timeout=45.0,
            )
            r = client.chat.completions.create(
                model=os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("NIM failed, falling back to Gemini: %s", str(e)[:100])
    return _gemini_fallback(prompt)


class LearningPathRequest(BaseModel):
    current_role: str = Field(..., min_length=2, description="Current job role or skill level")
    target_role: str = Field(..., min_length=2, description="Target role or goal")
    job_description: Optional[str] = Field(None, description="Optional JD to extract required skills from")


class OnboardingRequest(BaseModel):
    new_hire_role: str = Field(..., min_length=2, description="Role being hired for")
    department: str = Field("", description="Team or department (optional)")


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
        text = await asyncio.to_thread(_invoke_llm, prompt)
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


@router.post("/onboarding")
async def generate_onboarding_plan(
    body: OnboardingRequest,
    username: str = Depends(get_current_user),
):
    """Generate a 30-60-90 day onboarding plan for a new hire."""
    dept = f" in {body.department}" if body.department.strip() else ""

    prompt = f"""You are an expert L&D specialist. Create a structured 30-60-90 day onboarding plan
for a new {body.new_hire_role}{dept}.

Return ONLY valid JSON:
{{
  "role": "{body.new_hire_role}",
  "overview": "2-sentence summary of the onboarding philosophy for this role",
  "periods": [
    {{
      "label": "Days 1-30",
      "theme": "Orientation & Foundation",
      "goals": ["goal1", "goal2", "goal3"],
      "skills_to_learn": ["skill1", "skill2", "skill3"],
      "key_activities": ["activity1", "activity2", "activity3"],
      "success_metric": "How to measure success at end of this period"
    }},
    {{
      "label": "Days 31-60",
      "theme": "Building Proficiency",
      "goals": [],
      "skills_to_learn": [],
      "key_activities": [],
      "success_metric": ""
    }},
    {{
      "label": "Days 61-90",
      "theme": "Independence & Impact",
      "goals": [],
      "skills_to_learn": [],
      "key_activities": [],
      "success_metric": ""
    }}
  ],
  "key_tools": ["tool1", "tool2", "tool3"],
  "recommended_courses": ["course topic 1", "course topic 2", "course topic 3"]
}}

Be specific to the {body.new_hire_role} role. Include technical skills, soft skills, and team integration."""

    try:
        text = await asyncio.to_thread(_invoke_llm, prompt, 2048)
        start, end = text.find("{"), text.rfind("}") + 1
        plan = json.loads(text[start:end])
        return plan
    except Exception as e:
        logger.error("Onboarding plan generation failed: %s", e)
        return {"error": "Could not generate onboarding plan. Please try again."}
