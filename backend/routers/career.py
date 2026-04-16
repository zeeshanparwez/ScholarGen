"""
Career Tools router — JD Analyzer + Resume Skill Extractor.
Uses Gemini with NIM fallback (same pattern as learningpath.py).
"""

import json
import logging
import os
from datetime import datetime, timezone
from io import BytesIO

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI as _NimSync
from pydantic import BaseModel

from backend.core.database import get_profile, upsert_profile
from backend.core.key_manager import get_api_key
from backend.dependencies import get_current_user

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT, "Config", ".env"))

logger = logging.getLogger(__name__)
router = APIRouter()

_llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    google_api_key=get_api_key(0),
    temperature=0.3,
    max_retries=1,
    model_kwargs={"generation_config": {"thinking_config": {"thinking_budget": 0}}},
)


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
    return _llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end]) if start != -1 else {}


class CareerRequest(BaseModel):
    text: str
    mode: str = "jd"   # 'jd' | 'resume'


class CoverLetterRequest(BaseModel):
    jd_text: str
    notes: str = ""   # optional extra context from user


class PlaylistRequest(BaseModel):
    urls: list[str]


@router.post("/analyze")
def analyze(body: CareerRequest, username: str = Depends(get_current_user)):
    if body.mode == "resume":
        return _analyze_resume(body.text, username)
    return _analyze_jd(body.text, username)


def _analyze_jd(text: str, username: str) -> dict:
    profile = get_profile(username) or {}
    user_skills = profile.get("skills", [])

    prompt = f"""Analyze this job description and extract required skills.
Compare with the candidate's profile below.

Job Description:
{text[:3000]}

Candidate's Current Skills: {', '.join(user_skills) if user_skills else 'None recorded'}

Return ONLY valid JSON:
{{
  "role_title": "inferred job title",
  "required_skills": ["skill1", "skill2"],
  "matched": ["skills candidate already has from the required list"],
  "gaps": ["required skills the candidate is missing"],
  "quick_wins": ["top 3 easiest gaps to close first"]
}}"""

    try:
        result = _parse_json(_invoke_llm(prompt))
        return {
            "mode": "jd",
            "role_title": result.get("role_title", ""),
            "required_skills": result.get("required_skills", []),
            "matched": result.get("matched", []),
            "gaps": result.get("gaps", []),
            "quick_wins": result.get("quick_wins", []),
        }
    except Exception as e:
        logger.error("JD analysis failed: %s", e)
        return {"error": "Analysis failed. Please try again."}


@router.post("/cover-letter")
def generate_cover_letter(body: CoverLetterRequest, username: str = Depends(get_current_user)):
    profile = get_profile(username) or {}
    skills = ", ".join(profile.get("skills", [])[:15]) or "various technical skills"
    current_role = profile.get("current_role", "") or "professional"
    target_role = profile.get("target_role", "")

    notes_section = f"\n\nAdditional context from applicant:\n{body.notes}" if body.notes.strip() else ""

    prompt = f"""Write a professional cover letter for this job application.

Applicant Profile:
- Current role: {current_role}
- Target role: {target_role or 'the advertised position'}
- Key skills: {skills}

Job Description:
{body.jd_text[:2500]}{notes_section}

Write a compelling, specific cover letter (3–4 paragraphs).
- Open with a strong hook, not "I am applying for..."
- Connect specific skills to specific JD requirements
- Show genuine interest in the role/company
- Close with a clear call to action
- Tone: professional but human, confident not arrogant
- Length: ~300 words"""

    try:
        letter = _invoke_llm(prompt)
        return {"cover_letter": letter}
    except Exception as e:
        logger.error("Cover letter generation failed: %s", e)
        return {"error": "Generation failed. Please try again."}


@router.post("/playlist-guide")
def playlist_study_guide(body: PlaylistRequest, username: str = Depends(get_current_user)):
    import re
    from youtube_transcript_api import YouTubeTranscriptApi

    def get_video_id(url: str) -> str | None:
        for pattern in [r'v=([a-zA-Z0-9_-]{11})', r'youtu\.be/([a-zA-Z0-9_-]{11})', r'embed/([a-zA-Z0-9_-]{11})']:
            m = re.search(pattern, url)
            if m:
                return m.group(1)
        return None

    video_data = []
    for url in body.urls[:6]:  # cap at 6 videos
        vid_id = get_video_id(url.strip())
        if not vid_id:
            video_data.append({"url": url, "error": "Could not extract video ID"})
            continue
        try:
            segments = YouTubeTranscriptApi.get_transcript(vid_id)
            text = " ".join(s["text"] for s in segments)[:2500]
            video_data.append({"url": url, "vid_id": vid_id, "text": text})
        except Exception as e:
            video_data.append({"url": url, "vid_id": vid_id, "error": str(e)})

    good = [v for v in video_data if "text" in v]
    if not good:
        return {"error": "Could not fetch transcripts. Check the URLs and try again."}

    combined = "\n\n---\n\n".join(
        f"Video {i+1}: {v['url']}\n{v['text']}"
        for i, v in enumerate(good)
    )

    prompt = f"""You are creating a study guide from {len(good)} YouTube video transcript(s).

{combined}

Generate a structured study guide with:
1. **Overview** — What this playlist collectively teaches (2-3 sentences)
2. **Core Concepts** — Key ideas across all videos (bullet list)
3. **Video Summaries** — 3-4 key points per video, labeled by number
4. **Practical Takeaways** — Actionable things to do/practice after watching
5. **Suggested Study Order** — Recommended viewing sequence with reasoning

Be concise and learning-focused."""

    try:
        guide = _invoke_llm(prompt)
        return {
            "guide": guide,
            "video_count": len(good),
            "failed": [v["url"] for v in video_data if "error" in v],
        }
    except Exception as e:
        logger.error("Playlist guide generation failed: %s", e)
        return {"error": "Guide generation failed. Please try again."}


@router.post("/parse-pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    username: str = Depends(get_current_user),
):
    """Extract plain text from an uploaded PDF file."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF too large. Maximum size is 10 MB.")
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(t for t in pages_text if t.strip())
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text. Make sure the PDF contains selectable text (not a scanned image).",
            )
        return {"text": text.strip(), "pages": len(reader.pages)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF parsing error: %s", e)
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")


def _analyze_resume(text: str, username: str) -> dict:
    prompt = f"""Extract skills and experience summary from this resume/CV.

Resume:
{text[:3000]}

Return ONLY valid JSON:
{{
  "skills": ["skill1", "skill2"],
  "interests": ["area1", "area2"],
  "experience_level": "junior|mid|senior",
  "current_role": "most recent job title or inferred role",
  "suggested_roles": ["role1", "role2", "role3"]
}}"""

    try:
        result = _parse_json(_invoke_llm(prompt))
        skills = result.get("skills", [])
        interests = result.get("interests", [])

        # Merge into profile
        profile = get_profile(username) or {}
        merged_skills = list(dict.fromkeys(profile.get("skills", []) + skills))
        merged_interests = list(dict.fromkeys(profile.get("interests", []) + interests))
        upsert_profile(
            username=username,
            interests=merged_interests,
            skills=merged_skills,
            last_updated=datetime.now(timezone.utc).isoformat(),
            current_role=result.get("current_role", "") or profile.get("current_role", ""),
            target_role=profile.get("target_role", ""),
        )

        return {
            "mode": "resume",
            "skills": skills,
            "interests": interests,
            "experience_level": result.get("experience_level", ""),
            "current_role": result.get("current_role", ""),
            "suggested_roles": result.get("suggested_roles", []),
            "profile_updated": True,
        }
    except Exception as e:
        logger.error("Resume analysis failed: %s", e)
        return {"error": "Analysis failed. Please try again."}
