"""
NPTEL course retrieval via ChromaDB in-memory vector store.

On first use, pre-computed embeddings are loaded from the Excel seed file into a
ChromaDB in-memory collection. Subsequent queries hit ChromaDB directly — no manual
cosine_similarity needed.
"""

import ast
import json
import os
from functools import lru_cache
from typing import List, Optional

import chromadb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "Config", ".env"))

COURSE_DATA_PATH = os.environ.get(
    "COURSE_DATA_PATH",
    os.path.join(BASE_DIR, "Data", "nptel_courses_with_embeddings.xlsx"),
)
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")


def _to_array(x) -> np.ndarray:
    """Convert various embedding formats (ndarray, list, string) to float32 array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.array(x, dtype=np.float32)
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except Exception:
            cleaned = x.strip().lstrip("[").rstrip("]")
            vals = [float(v) for v in cleaned.split(",") if v.strip()]
            return np.array(vals, dtype=np.float32)
    raise ValueError(f"Unsupported embedding type: {type(x)}")


@lru_cache(maxsize=1)
def _get_collection():
    """
    Build in-memory ChromaDB collection from the Excel seed file.
    lru_cache ensures this runs only once per process.
    """
    client = chromadb.Client()  # pure in-memory; no disk I/O
    collection = client.get_or_create_collection(
        name="nptel_courses",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        return collection  # already populated in this session

    if not os.path.exists(COURSE_DATA_PATH):
        raise FileNotFoundError(f"Course data not found: {COURSE_DATA_PATH}")

    df = pd.read_excel(COURSE_DATA_PATH)
    if "embedding" not in df.columns:
        raise KeyError("'embedding' column missing from course data file")

    ids, embeddings, metadatas, documents = [], [], [], []
    for i, row in df.iterrows():
        try:
            emb = _to_array(row["embedding"]).tolist()
        except Exception:
            continue  # skip rows with malformed embeddings
        ids.append(str(i))
        embeddings.append(emb)
        documents.append(str(row.get("description", "")))
        metadatas.append(
            {
                "course_name": str(row.get("course_name", "Untitled")),
                "url": str(row.get("url", "")),
                "description": str(row.get("description", ""))[:400],
            }
        )

    if ids:
        # Batch insert in chunks of 500 to avoid memory spikes
        batch = 500
        for start in range(0, len(ids), batch):
            collection.add(
                ids=ids[start : start + batch],
                embeddings=embeddings[start : start + batch],
                documents=documents[start : start + batch],
                metadatas=metadatas[start : start + batch],
            )

    return collection


@lru_cache(maxsize=1)
def _get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


class CourseRetriever:
    """
    Wraps ChromaDB collection for NPTEL course semantic search.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.15,
        exclude_urls: Optional[List[str]] = None,
    ) -> str:
        collection = _get_collection()
        model = _get_embed_model()

        query_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

        n_fetch = min(top_k + len(exclude_urls or []) + 5, collection.count() or 1)
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n_fetch,
            include=["metadatas", "distances"],
        )

        lines = [f"### Recommended courses for: **{query}**"]
        payload: dict = {"query": query, "results": []}
        exclude_set = set(exclude_urls or [])

        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            sim = 1.0 - dist  # ChromaDB cosine distance → similarity
            if sim < min_similarity:
                continue
            url = meta.get("url", "")
            if url in exclude_set:
                continue
            name = meta.get("course_name", "Untitled")
            desc = meta.get("description", "")[:250]
            lines.append(
                f"- **{name}**\n  {desc}\n  🔗 {url} _(similarity {sim:.3f})_"
            )
            payload["results"].append(
                {
                    "course_name": name,
                    "url": url,
                    "similarity": sim,
                    "description": meta.get("description", "")[:400],
                }
            )
            if len(payload["results"]) >= top_k:
                break

        if not payload["results"]:
            return f"No strong matches for: **{query}**"

        lines.append("\n<!-- JSON:" + json.dumps(payload) + " -->")
        return "\n".join(lines)


# ── LangChain tool wrapper ────────────────────────────────────────────────────

class CourseQuery(BaseModel):
    query: str = Field(..., description="Study topic, e.g., 'python for data science'")
    top_k: int = Field(3, ge=1, le=20, description="Number of courses to return")
    min_similarity: float = Field(
        0.15, ge=0.0, le=1.0, description="Minimum similarity threshold (0–1)"
    )
    exclude_urls: Optional[List[str]] = Field(
        default=None, description="URLs to exclude from results"
    )


class CourseTool:
    def __init__(self, retriever: CourseRetriever):
        self.retriever = retriever

    def tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            name="find_nptel_courses",
            description=(
                "Recommend NPTEL courses for a given study topic. "
                "Use when the user wants to learn something via NPTEL."
            ),
            func=self._find,
            args_schema=CourseQuery,
            return_direct=False,
        )

    def _find(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.15,
        exclude_urls: Optional[List[str]] = None,
    ) -> str:
        return self.retriever.retrieve(query, top_k, min_similarity, exclude_urls)


if __name__ == "__main__":
    retriever = CourseRetriever()
    print("\nTesting CourseRetriever...")
    res = retriever.retrieve("machine learning", top_k=3)
    print(res)
