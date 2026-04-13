from fastapi import APIRouter, Depends, Query
from backend.dependencies import get_current_user
from backend.core.course_retriever import CourseRetriever

router = APIRouter()
_retriever = CourseRetriever()


@router.get("/search")
async def search_courses(
    query: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=20),
    username: str = Depends(get_current_user),
):
    """Search NPTEL courses by semantic similarity."""
    raw = _retriever.retrieve(query, top_k=top_k)

    # Parse the markdown+JSON response from CourseRetriever
    import json, re
    results = []
    match = re.search(r"<!-- JSON:(.*?) -->", raw, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            results = payload.get("results", [])
        except Exception:
            pass

    return {"query": query, "results": results}
