# /// script
# dependencies = ["arxiv", "chromadb", "fastmcp", "mcp"]
# ///
"""
arXiv MCP server — paper search and retrieval backed by ChromaDB in-memory store.

Replaces the per-topic papers/<topic>/papers_info.json file tree.
All papers are stored in a single flat ChromaDB collection that lives for the
lifetime of the MCP subprocess (one chat session).
"""

import json
import os
from typing import List

import arxiv
import chromadb
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research")

# Initialise once at subprocess startup — persists across all tool calls.
_client = chromadb.Client()  # pure in-memory
_papers = _client.get_or_create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"},
)


@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search arXiv for papers on a topic and store them in the in-memory RAG store.

    Args:
        topic: The research topic to search for.
        max_results: How many results to fetch from arXiv (default 5).

    Returns:
        List of arXiv paper IDs found.
    """
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    paper_ids: List[str] = []
    ids_to_add, docs_to_add, metas_to_add = [], [], []

    for paper in arxiv.Client().results(search):
        pid = paper.get_short_id()
        paper_ids.append(pid)

        # Skip papers already in the collection (idempotent)
        if _papers.get(ids=[pid])["ids"]:
            continue

        ids_to_add.append(pid)
        # Combine title + summary as the searchable document text
        docs_to_add.append(f"{paper.title}. {paper.summary}")
        metas_to_add.append(
            {
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors),
                "summary": paper.summary[:500],
                "pdf_url": paper.pdf_url or "",
                "published": str(paper.published.date()),
                "topic": topic,
            }
        )

    if ids_to_add:
        _papers.add(
            ids=ids_to_add,
            documents=docs_to_add,
            metadatas=metas_to_add,
        )

    return paper_ids


@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Retrieve stored metadata for a specific paper by its arXiv ID.

    Args:
        paper_id: The short arXiv ID (e.g. '2301.07041').

    Returns:
        JSON string with paper metadata, or an error message if not found.
    """
    result = _papers.get(ids=[paper_id], include=["metadatas"])
    if result["ids"]:
        return json.dumps(result["metadatas"][0], indent=2)
    return f"No saved information for paper '{paper_id}'."


@mcp.tool()
def search_cached_papers(query: str, n_results: int = 5) -> str:
    """
    Semantically search all previously fetched papers in this session.

    Args:
        query: A natural language query (e.g. 'attention mechanisms in vision').
        n_results: Number of results to return (default 5).

    Returns:
        JSON string with matching papers and their metadata.
    """
    total = _papers.count()
    if total == 0:
        return "No papers cached yet. Use search_papers first."

    results = _papers.query(
        query_texts=[query],
        n_results=min(n_results, total),
        include=["metadatas", "distances"],
    )

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        output.append({"similarity": round(1.0 - dist, 4), **meta})

    return json.dumps(output, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
