"""
flashcards.py — Skill Assessment quiz generator.
Generates MCQ-style questions for any technical skill/topic.
Used by backend/routers/flashcards.py.
"""

import json
import os
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI as _NimClient

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(_FILE_DIR))
load_dotenv(os.path.join(_BASE_DIR, "Config", ".env"))

# ── Skill Categories (replaces GATE specializations) ──────────────────────────

SKILL_CATEGORIES = {
    "Frontend Development": [
        "HTML & CSS",
        "JavaScript (Core)",
        "React",
        "TypeScript",
        "Vue / Angular",
        "Web Performance & Accessibility",
        "Frontend Testing",
    ],
    "Backend Development": [
        "Python (FastAPI / Django / Flask)",
        "Node.js (Express)",
        "REST API Design",
        "Databases & SQL",
        "Authentication & Security",
        "Caching & Queues",
        "System Design Basics",
    ],
    "Data Science & ML": [
        "Python for Data Science",
        "Statistics & Probability",
        "Machine Learning Fundamentals",
        "Deep Learning & Neural Networks",
        "NLP & LLMs",
        "Data Visualization",
        "MLOps & Model Deployment",
    ],
    "Cloud & DevOps": [
        "AWS Core Services",
        "Azure / GCP",
        "Docker & Containers",
        "Kubernetes",
        "CI/CD Pipelines",
        "Infrastructure as Code (Terraform)",
        "Monitoring & Observability",
    ],
    "Mobile Development": [
        "React Native",
        "Flutter",
        "iOS (Swift)",
        "Android (Kotlin)",
    ],
    "Cybersecurity": [
        "Web Security (OWASP Top 10)",
        "Network Security",
        "Cryptography",
        "Cloud Security",
        "Secure Coding Practices",
    ],
    "Database Engineering": [
        "SQL & Query Optimization",
        "PostgreSQL",
        "MongoDB",
        "Redis",
        "Database Design & Normalization",
        "Data Warehousing",
    ],
    "System Design": [
        "Architecture Patterns",
        "Microservices",
        "Scalability & Load Balancing",
        "API Design & GraphQL",
        "Distributed Systems",
        "Caching Strategies",
    ],
    "General CS Fundamentals": [
        "Algorithms & Data Structures",
        "Design Patterns",
        "Operating Systems",
        "Computer Networks",
        "Compilers & Language Theory",
    ],
}


def get_specializations() -> List[str]:
    return list(SKILL_CATEGORIES.keys())


def get_subjects(skill_area: str) -> List[str]:
    return SKILL_CATEGORIES.get(skill_area, [])


# ── Skill Quiz Generator ───────────────────────────────────────────────────────

class SkillQuizGenerator:
    """Generate skill-assessment MCQs. NIM (Llama 3.3) first, Gemini fallback."""

    def __init__(self):
        self._nim_key  = os.environ.get("NIM_API_KEY", "")
        self._nim_base = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self._nim_model = os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct")
        self._gemini = ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            temperature=0.4,
            max_retries=1,
            model_kwargs={
                "generation_config": {"thinking_config": {"thinking_budget": 0}}
            },
        )

    def _call_nim(self, prompt: str) -> str:
        if not self._nim_key:
            raise RuntimeError("NIM_API_KEY not configured")
        client = _NimClient(base_url=self._nim_base, api_key=self._nim_key)
        r = client.chat.completions.create(
            model=self._nim_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2048,
        )
        return r.choices[0].message.content.strip()

    def _call_gemini(self, prompt: str) -> str:
        response = self._gemini.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def generate_flashcards(
        self,
        skill_area: str,
        sub_area: str,
        topic: str,
        num_questions: int = 5,
    ) -> List[Dict]:
        prompt = self._build_prompt(skill_area, sub_area, topic, num_questions)

        # Try Azure OpenAI first (primary)
        from backend.core.azure_llm import invoke_azure, is_azure_configured
        if is_azure_configured():
            try:
                raw = invoke_azure(prompt, temperature=0.4, max_tokens=2048)
                cards = self._parse_response(raw)
                if cards:
                    return cards
                print("Azure returned no parseable cards, falling back to NIM")
            except Exception as e:
                print(f"Azure failed ({e}), falling back to NIM")

        # Try NIM second
        try:
            raw = self._call_nim(prompt)
            cards = self._parse_response(raw)
            if cards:
                return cards
            print("NIM returned no parseable cards, falling back to Gemini")
        except Exception as e:
            print(f"NIM failed ({e}), falling back to Gemini")

        # Gemini last resort
        try:
            raw = self._call_gemini(prompt)
            cards = self._parse_response(raw)
            if cards:
                return cards
            print("Gemini also returned no parseable cards")
        except Exception as e:
            print(f"Gemini also failed: {e}")

        return []

    def _build_prompt(self, skill_area: str, sub_area: str, topic: str, num: int) -> str:
        return f"""You are a senior technical interviewer and skill assessment expert.

Generate {num} high-quality multiple-choice questions to assess proficiency in:

**Skill Area**: {skill_area}
**Sub-area**: {sub_area}
**Topic**: {topic}

Requirements:
1. Each question must have exactly 4 answer options
2. Mix question types: conceptual understanding, practical/code-based, and scenario/tradeoff questions
3. Mark exactly ONE correct option (index 0–3, zero-based)
4. Provide a clear, educational explanation for the correct answer
5. Questions should match real interview and technical assessment standards
6. Avoid trivial or overly obvious questions — test real understanding

Output ONLY valid JSON, no extra text:

{{
  "flashcards": [
    {{
      "question": "What is the time complexity of Array.prototype.includes() in JavaScript?",
      "options": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
      "correct_index": 2,
      "explanation": "Array.includes() does a linear scan through the array, making it O(n) in the worst case. Use a Set for O(1) lookups."
    }}
  ]
}}

Generate {num} questions now."""

    def _parse_response(self, raw: str) -> List[Dict]:
        try:
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start == -1 or end <= start:
                return []
            data = json.loads(text[start:end])
            result = []
            for fc in data.get("flashcards", []):
                if self._valid(fc):
                    result.append({
                        "question":      fc["question"],
                        "options":       fc["options"],
                        "correct_index": int(fc["correct_index"]),
                        "explanation":   fc.get("explanation", "No explanation provided."),
                    })
            return result
        except Exception as e:
            print(f"Response parse error: {e}")
            return []

    def _valid(self, fc: Dict) -> bool:
        return (
            isinstance(fc, dict)
            and all(k in fc for k in ("question", "options", "correct_index"))
            and isinstance(fc["options"], list) and len(fc["options"]) == 4
            and isinstance(fc["correct_index"], (int, float))
            and 0 <= int(fc["correct_index"]) < 4
        )


# Backward-compat alias used by the router
GATEFlashcardGenerator = SkillQuizGenerator
GATE_SPECIALIZATIONS = SKILL_CATEGORIES
