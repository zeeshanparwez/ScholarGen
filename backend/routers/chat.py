import asyncio
import datetime
import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.dependencies import get_current_user
from backend.services.chatbot_service import chatbot_service
from backend.core.database import upsert_profile
from backend.core.collaboration import extract_profile_from_text

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    provider: str = "gemini"   # "gemini" | "groq"


@router.get("/providers")
async def list_providers(username: str = Depends(get_current_user)):
    """Returns which LLM providers are currently available."""
    return {"providers": chatbot_service.available_providers}


def _update_profile_bg(username: str, text: str):
    """Run profile extraction and DB update in a thread (non-blocking)."""
    try:
        profile = extract_profile_from_text(text)
        upsert_profile(
            username=username,
            interests=profile.get("interests", []),
            skills=profile.get("skills", []),
            last_updated=datetime.datetime.now().isoformat(),
        )
    except Exception as exc:
        logger.debug("Profile update skipped for %s: %s", username, exc)


@router.post("/stream")
async def stream_chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    username: str = Depends(get_current_user),
):
    """SSE endpoint — streams tokens as they are generated."""

    async def generate():
        full_response = []
        async for chunk in chatbot_service.stream_chat(username, body.message, body.provider):
            yield chunk
            # Collect tokens to build full response for profile extraction
            if '"type":"token"' in chunk:
                import json as _json
                try:
                    data = _json.loads(chunk[len("data: "):].strip())
                    if data.get("type") == "token":
                        full_response.append(data["content"])
                except Exception:
                    pass

        # Fire-and-forget: profile extraction runs in a thread after stream ends
        # so it never blocks the event loop (avoids freezing if Gemini is down)
        combined = body.message + "\n" + "".join(full_response)
        asyncio.get_event_loop().run_in_executor(
            None, _update_profile_bg, username, combined
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.delete("/clear")
async def clear_chat(username: str = Depends(get_current_user)):
    """Clear conversation memory for the current user."""
    chatbot_service.clear_memory(username)
    return {"message": "Conversation cleared"}
