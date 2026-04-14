from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.dependencies import get_current_user
from backend.core.database import add_bookmark, get_bookmarks, delete_bookmark

router = APIRouter()


class BookmarkIn(BaseModel):
    content: str


@router.get("")
def list_bookmarks(username: str = Depends(get_current_user)):
    return {"bookmarks": get_bookmarks(username)}


@router.post("")
def create_bookmark(body: BookmarkIn, username: str = Depends(get_current_user)):
    ts = datetime.now(timezone.utc).isoformat()
    bid = add_bookmark(username, body.content, ts)
    return {"id": bid, "content": body.content, "timestamp": ts}


@router.delete("/{bookmark_id}")
def remove_bookmark(bookmark_id: int, username: str = Depends(get_current_user)):
    ok = delete_bookmark(bookmark_id, username)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(404, "Bookmark not found")
    return {"ok": True}
