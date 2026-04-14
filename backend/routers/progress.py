from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.dependencies import get_current_user
from backend.core.database import (
    upsert_progress, get_progress, get_progress_by_url,
    update_progress_status, delete_progress,
)

router = APIRouter()

VALID_STATUSES = {"saved", "in_progress", "done"}


class ProgressIn(BaseModel):
    item_type: str   # 'course' | 'paper'
    item_url: str
    title: str
    status: str = "saved"


class StatusUpdate(BaseModel):
    status: str


@router.get("")
def list_progress(username: str = Depends(get_current_user)):
    return {"items": get_progress(username)}


@router.get("/check")
def check_progress(item_url: str = Query(...), username: str = Depends(get_current_user)):
    item = get_progress_by_url(username, item_url)
    return {"item": item}


@router.post("")
def add_progress(body: ProgressIn, username: str = Depends(get_current_user)):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {VALID_STATUSES}")
    ts = datetime.now(timezone.utc).isoformat()
    pid = upsert_progress(username, body.item_type, body.item_url, body.title, body.status, ts)
    return {"id": pid, "item_type": body.item_type, "item_url": body.item_url,
            "title": body.title, "status": body.status, "timestamp": ts}


@router.patch("/{progress_id}")
def update_status(progress_id: int, body: StatusUpdate, username: str = Depends(get_current_user)):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {VALID_STATUSES}")
    ok = update_progress_status(progress_id, username, body.status)
    if not ok:
        raise HTTPException(404, "Progress item not found")
    return {"ok": True, "status": body.status}


@router.delete("/{progress_id}")
def remove_progress(progress_id: int, username: str = Depends(get_current_user)):
    ok = delete_progress(progress_id, username)
    if not ok:
        raise HTTPException(404, "Progress item not found")
    return {"ok": True}
