"""
Analytics router — org-level skill and learning metrics for the CHRO dashboard.
NIM first, Gemini fallback.
"""

import json
import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI as _NimSync

from backend.core.database import get_analytics_data
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
    """NIM first; Gemini fallback."""
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


@router.get("")
def get_dashboard(username: str = Depends(get_current_user)):
    """Return org-level analytics for the CHRO dashboard."""
    data = get_analytics_data()
    return data
