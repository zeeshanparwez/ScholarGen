"""
Analytics router — org-level skill and learning metrics for the CHRO dashboard.

Endpoints:
  GET  /api/analytics          — dashboard data
  POST /api/analytics/brief    — AI-generated executive workforce intelligence brief
  POST /api/analytics/gap-plan — AI training campaign for a specific skill gap
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI as _NimSync
from pydantic import BaseModel

from backend.core.database import get_analytics_data
from backend.core.key_manager import get_all_keys
from backend.dependencies import get_current_user

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT, "Config", ".env"))

logger = logging.getLogger(__name__)
router = APIRouter()


def _gemini_fallback(prompt: str, max_tokens: int = 1024) -> str:
    keys = get_all_keys()
    if not keys:
        raise RuntimeError("No GOOGLE_API_KEY configured")
    last_exc: Exception = RuntimeError("No keys")
    for key in keys:
        try:
            llm = ChatGoogleGenerativeAI(
                model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
                google_api_key=key,
                temperature=0.3,
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


def _invoke_llm(prompt: str, max_tokens: int = 1024) -> str:
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
                timeout=30.0,
            )
            r = client.chat.completions.create(
                model=os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("NIM failed, falling back to Gemini: %s", str(e)[:100])
    return _gemini_fallback(prompt, max_tokens)


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end]) if start != -1 else {}


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("")
def get_dashboard(username: str = Depends(get_current_user)):
    """Return org-level analytics for the CHRO dashboard."""
    return get_analytics_data()


# ── Skill Matrix ──────────────────────────────────────────────────────────────

@router.get("/skill-matrix")
def get_skill_matrix(username: str = Depends(get_current_user)):
    """
    Returns an employee × skill grid.
    Response: { skills: [...], employees: [{username, current_role, has: [bool, ...]}, ...] }
    """
    import json as _json
    from backend.core.database import _conn

    data = get_analytics_data()

    with _conn() as conn:
        real_profiles = conn.execute("SELECT * FROM user_profiles").fetchall()

    all_learners = []
    skill_counter = {}

    for row in real_profiles:
        skills = _json.loads(row["skills"] or "[]")
        for s in skills:
            skill_counter[s] = skill_counter.get(s, 0) + 1
            
        all_learners.append({
            "username":     row["username"],
            "current_role": row["current_role"] or "",
            "skills":       skills,
        })

    # Determine top 8 skills by frequency across all employees
    sorted_skills = sorted(skill_counter.items(), key=lambda x: x[1], reverse=True)
    top_skills = [s for s, _ in sorted_skills[:8]]

    # Build matrix rows
    rows = []
    for emp in all_learners:
        emp_skills_lower = [s.lower() for s in emp["skills"]]
        has = []
        for skill in top_skills:
            matched = any(
                skill.lower() in es or es in skill.lower()
                for es in emp_skills_lower
            )
            has.append(matched)
        rows.append({
            "username":     emp["username"],
            "current_role": emp["current_role"],
            "has":          has,
        })

    # Sort: most skills first
    rows.sort(key=lambda r: sum(r["has"]), reverse=True)

    return {"skills": top_skills, "employees": rows}


# ── Executive Workforce Intelligence Brief ────────────────────────────────────

class GapPlanRequest(BaseModel):
    skill: str


@router.post("/brief")
async def generate_brief(username: str = Depends(get_current_user)):
    """
    AI-generated executive workforce intelligence brief.
    Synthesises all org analytics into a board-ready report with:
    - Org Health Score (0-100) + grade (A-F)
    - Top risk areas with business impact
    - Strategic opportunities
    - Priority actions with 30/60/90-day timelines
    """
    data = get_analytics_data()

    top_gaps   = ", ".join(f"{g['skill']} ({g['count']} employees)" for g in data["skill_gaps"][:5])
    top_people = ", ".join(
        f"{l['username']} ({l['streak']}d streak, targeting {l['target_role'] or 'unset'})"
        for l in data["leaderboard"][:4]
    )

    prompt = f"""You are a Chief People Officer writing an executive workforce intelligence brief for a board meeting.

Org Metrics:
- Total learners: {data['total_learners']}
- Active learners this week: {data['active_learners']} ({round(data['active_learners']/max(1,data['total_learners'])*100)}% engagement rate)
- Average learning streak: {data['avg_streak']} days
- Completion rate: {data['completion_rate']}%
- In-progress learning items: {data['progress']['in_progress']}
- Completed items: {data['progress']['done']}
- Top 5 skill gaps (employees lacking): {top_gaps}
- Top performers: {top_people}

Write a crisp, board-ready executive brief. Return ONLY valid JSON:
{{
  "org_health_score": <integer 0-100 based on engagement, completion, streak>,
  "org_health_grade": "<A+|A|A-|B+|B|B-|C+|C|D>",
  "headline": "<one punchy sentence — the single most important thing the CHRO needs to know>",
  "risk_areas": [
    {{
      "skill": "<skill name>",
      "severity": "<critical|high|medium>",
      "employees_affected": <number>,
      "business_impact": "<one sentence on what business initiative is at risk if this gap isn't closed>"
    }}
  ],
  "opportunities": [
    {{
      "title": "<short opportunity title>",
      "description": "<one sentence on the opportunity and how to capture it>"
    }}
  ],
  "top_performers": [
    {{
      "name": "<username>",
      "highlight": "<one sentence on what makes them stand out and what role they're ready for>"
    }}
  ],
  "priority_actions": [
    {{
      "action": "<specific action — verb-first>",
      "timeline": "<This week|30 days|60 days|90 days>",
      "owner": "<CHRO|L&D Team|Manager|Individual>",
      "expected_outcome": "<measurable outcome>"
    }}
  ],
  "strategic_summary": "<2-3 sentences: where the org stands today, what the biggest lever is, and what success looks like in 90 days>"
}}

Be specific and data-driven. Reference the actual skill gaps and metrics. Do not be generic."""

    try:
        text = await asyncio.to_thread(_invoke_llm, prompt, 1200)
        result = _parse_json(text)
        # Attach raw analytics so frontend can cross-reference
        result["_analytics"] = {
            "total_learners":  data["total_learners"],
            "active_learners": data["active_learners"],
            "completion_rate": data["completion_rate"],
            "avg_streak":      data["avg_streak"],
        }
        return result
    except Exception as e:
        logger.error("Executive brief generation failed: %s", e)
        return {"error": "Could not generate brief. Please try again."}


# ── Skill Gap Training Campaign ───────────────────────────────────────────────

@router.post("/gap-plan")
async def generate_gap_training_plan(
    body: GapPlanRequest,
    username: str = Depends(get_current_user),
):
    """
    For a specific skill gap, generate:
    - Which demo employees are affected
    - A structured group training campaign (3 phases)
    - Quick wins for this week
    - Success metrics
    """
    data   = get_analytics_data()
    skill  = body.skill
    gap    = next((g for g in data["skill_gaps"] if g["skill"] == skill), None)
    count  = gap["count"] if gap else 0

    # Find employees who DON'T have this skill (used for display in frontend)
    affected = [
        l for l in data["leaderboard"]
        if not any(skill.lower() in s.lower() or s.lower() in skill.lower() for s in l.get("skills", []))
    ]
    # Pad to show at least 3 names
    all_learners = sorted(data["leaderboard"], key=lambda x: x["streak"], reverse=True)
    if len(affected) < 3:
        affected = all_learners[:min(6, len(all_learners))]

    affected_names = [l["username"] for l in affected[:6]]

    prompt = f"""You are an L&D director designing a skill-building campaign.

Skill Gap: {skill}
Employees affected: {count} people
Sample employees who need this skill: {', '.join(affected_names)}

Design a focused group training campaign. Return ONLY valid JSON:
{{
  "skill": "{skill}",
  "campaign_name": "<compelling name for the training campaign>",
  "estimated_weeks": <integer 6-12>,
  "approach": "<one sentence on the overall training strategy>",
  "phases": [
    {{
      "name": "<phase name>",
      "week_range": "<e.g. Week 1-3>",
      "focus": "<what this phase builds>",
      "activities": ["<specific activity>", "<specific activity>", "<specific activity>"],
      "deliverable": "<what employees produce/demonstrate at end of phase>"
    }},
    {{
      "name": "<phase name>",
      "week_range": "<e.g. Week 4-8>",
      "focus": "<what this phase builds>",
      "activities": ["<activity>", "<activity>", "<activity>"],
      "deliverable": "<deliverable>"
    }},
    {{
      "name": "<phase name>",
      "week_range": "<e.g. Week 9-12>",
      "focus": "<mastery and application>",
      "activities": ["<activity>", "<activity>", "<activity>"],
      "deliverable": "<final deliverable>"
    }}
  ],
  "quick_wins": ["<something employees can do THIS WEEK — specific>", "<another quick win>"],
  "success_metrics": ["<measurable metric>", "<measurable metric>", "<measurable metric>"],
  "estimated_cost": "<e.g. Low ($500-2K in course licenses) or Medium ($5-15K for instructor-led)>",
  "roi_note": "<one sentence on the business ROI of closing this gap>"
}}"""

    try:
        text = await asyncio.to_thread(_invoke_llm, prompt, 900)
        result = _parse_json(text)
        result["affected_employees"] = affected_names
        result["total_affected"]     = count
        return result
    except Exception as e:
        logger.error("Gap training plan failed: %s", e)
        return {"error": "Could not generate training plan. Please try again."}
