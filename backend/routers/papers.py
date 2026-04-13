from fastapi import APIRouter, Depends, Query, HTTPException
from backend.dependencies import get_current_user
import arxiv

router = APIRouter()


@router.get("/search")
async def search_papers(
    topic: str = Query(..., min_length=2),
    max_results: int = Query(5, ge=1, le=20),
    username: str = Depends(get_current_user),
):
    """Search arXiv for recent papers on a topic."""
    try:
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for paper in arxiv.Client().results(search):
            papers.append({
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [a.name for a in paper.authors][:4],
                "summary": paper.summary[:400],
                "pdf_url": paper.pdf_url or "",
                "published": str(paper.published.date()),
            })
        return {"topic": topic, "papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"arXiv search failed: {str(e)}")
