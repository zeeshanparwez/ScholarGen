import os
import ast
import json
import math
import numpy as np
import pandas as pd
from typing import List, Optional
from functools import lru_cache
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

load_dotenv("./Config/.env")

class CourseRetriever:
    """
    Loads NPTEL course dataset with precomputed embeddings and retrieves top matches.
    """

    def __init__(self,
                 file_path: str = os.getenv("COURSE_DATA_PATH"),
                 embed_model: str = "BAAI/bge-base-en-v1.5"):
        self.file_path = file_path
        self.embed_model_name = embed_model
        self._model = None
        self._courses = None

    # ---------------------------
    # Properties
    # ---------------------------
    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.embed_model_name)
        return self._model

    @property
    def courses(self) -> pd.DataFrame:
        if self._courses is None:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Course file not found: {self.file_path}")
            df = pd.read_excel(self.file_path)
            if "embedding" not in df.columns:
                raise KeyError("Missing 'embedding' column in Excel")
            df["_emb"] = df["embedding"].apply(self._to_array)
            self._courses = df
        return self._courses

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _to_array(self, x) -> np.ndarray:
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

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # ---------------------------
    # Retrieval
    # ---------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.15,
        exclude_urls: Optional[List[str]] = None,
    ) -> str:
        df = self.courses
        qv = self.embed_query(query)
        embs = np.vstack(df["_emb"].values)
        sims = cosine_similarity(qv.reshape(1, -1), embs)[0]

        if exclude_urls:
            mask = df["url"].astype(str).isin(set(exclude_urls))
            sims = np.where(mask.values, -math.inf, sims)

        sims = np.where(sims >= min_similarity, sims, -math.inf)

        top_idx = np.argsort(sims)[-top_k:][::-1]
        top_idx = [i for i in top_idx if np.isfinite(sims[i])]

        if not top_idx:
            return f"No strong matches for: **{query}**"

        lines = [f"### Recommended courses for: **{query}**"]
        payload = {"query": query, "results": []}

        for i in top_idx:
            r = df.iloc[i]
            name = str(r.get("course_name", "Untitled"))
            url = str(r.get("url", ""))
            desc = str(r.get("description", ""))[:250]
            lines.append(f"- **{name}**\n  {desc}\n  🔗 {url} _(similarity {sims[i]:.3f})_")
            payload["results"].append(
                {
                    "course_name": name,
                    "url": url,
                    "similarity": float(sims[i]),
                    "description": str(r.get("description", ""))[:400],
                }
            )

        lines.append("\n<!-- JSON:" + json.dumps(payload) + " -->")
        return "\n".join(lines)


# ==============================
# LangChain Tool Wrapper
# ==============================
class CourseQuery(BaseModel):
    query: str = Field(..., description="Study topic, e.g., 'python for data science'")
    top_k: int = Field(3, ge=1, le=20, description="Number of courses to return")
    min_similarity: float = Field(0.15, ge=0.0, le=1.0, description="Min similarity (0-1)")
    exclude_urls: Optional[List[str]] = Field(default=None, description="Exclude already shown URLs")


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
        self, query: str, top_k: int = 3, min_similarity: float = 0.15, exclude_urls=None
    ) -> str:
        return self.retriever.retrieve(query, top_k, min_similarity, exclude_urls)


# ==============================
# Test script
# ==============================
if __name__ == "__main__":
    retriever = CourseRetriever()
    print("\n🔎 Testing CourseRetriever...")
    res = retriever.retrieve("machine learning", top_k=3)
    print(res)
