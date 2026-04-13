import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.dependencies import get_current_user
from backend.services.chatbot_service import chatbot_service
from database import upsert_profile
from collaboration import extract_profile_from_text

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/stream")
async def stream_chat(
    body: ChatRequest,
    username: str = Depends(get_current_user),
):
    """SSE endpoint — streams tokens as they are generated."""

    async def generate():
        full_response = []
        async for chunk in chatbot_service.stream_chat(username, body.message):
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

        # After streaming completes, update user profile in background
        try:
            combined = body.message + "\n" + "".join(full_response)
            profile = extract_profile_from_text(combined)
            upsert_profile(
                username=username,
                interests=profile.get("interests", []),
                skills=profile.get("skills", []),
                last_updated=datetime.datetime.now().isoformat(),
            )
        except Exception:
            pass  # Profile update is non-critical

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.delete("/clear")
async def clear_chat(username: str = Depends(get_current_user)):
    """Clear conversation memory for the current user."""
    chatbot_service.clear_memory(username)
    return {"message": "Conversation cleared"}
