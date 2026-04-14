import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List

from backend.dependencies import get_current_user
from backend.core.database import get_profile, upsert_profile

router = APIRouter()


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
async def update_my_profile(
    body: ProfileUpdate,
    username: str = Depends(get_current_user),
):
    upsert_profile(
        username=username,
        interests=body.interests,
        skills=body.skills,
        current_role=body.current_role,
        target_role=body.target_role,
        last_updated=datetime.datetime.now().isoformat(),
    )
    return {"message": "Profile updated"}
