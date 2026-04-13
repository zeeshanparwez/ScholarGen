from fastapi import APIRouter, Depends
from backend.dependencies import get_current_user
from backend.core.collaboration import match_similar_users, suggest_collaboration_topics

router = APIRouter()


@router.get("")
async def get_collaborators(username: str = Depends(get_current_user)):
    """Find similar users and suggest collaboration topics."""
    matched_users = match_similar_users(username, top_n=5)
    topics = suggest_collaboration_topics(matched_users)
    return {
        "matched_users": matched_users,
        "topics": topics,
    }
