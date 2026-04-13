"""
flashcards.py — GATE exam flashcard generator.
Used directly by the FastAPI backend router (backend/routers/flashcards.py).
"""

import json
import os
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Project root is 2 levels up from this file (backend/core/ → backend/ → ScholarGen/)
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(_FILE_DIR))
load_dotenv(os.path.join(_BASE_DIR, "Config", ".env"))

# ============== GATE SPECIALIZATIONS CONFIG ==============

GATE_SPECIALIZATIONS = {
    "Computer Science and Information Technology (CS)": [
        "Engineering Mathematics",
        "Digital Logic",
        "Computer Organization and Architecture",
        "Programming and Data Structures",
        "Algorithms",
        "Theory of Computation",
        "Compiler Design",
        "Operating Systems",
        "Databases",
        "Computer Networks"
    ],

    "Electronics and Communication Engineering (EC)": [
        "Engineering Mathematics",
        "Networks",
        "Signals and Systems",
        "Electronic Devices",
        "Analog Circuits",
        "Digital Circuits",
        "Control Systems",
        "Communications",
        "Electromagnetics"
    ],

    "Electrical Engineering (EE)": [
        "Engineering Mathematics",
        "Electric Circuits",
        "Electromagnetic Fields",
        "Signals and Systems",
        "Electrical Machines",
        "Power Systems",
        "Control Systems",
        "Electrical and Electronic Measurements",
        "Power Electronics"
    ],

    "Mechanical Engineering (ME)": [
        "Engineering Mathematics",
        "Applied Mechanics and Design",
        "Fluid Mechanics and Thermal Sciences",
        "Materials, Manufacturing and Industrial Engineering",
        "Strength of Materials",
        "Theory of Machines",
        "Thermodynamics",
        "Heat Transfer",
        "Production Engineering"
    ],

    "Civil Engineering (CE)": [
        "Engineering Mathematics",
        "Structural Engineering",
        "Geotechnical Engineering",
        "Water Resources Engineering",
        "Environmental Engineering",
        "Transportation Engineering",
        "Geomatics Engineering"
    ],

    "Data Science and Artificial Intelligence (DS & AI)": [
        "Linear Algebra",
        "Probability and Statistics",
        "Calculus",
        "Programming",
        "Data Structures and Algorithms",
        "Database Management",
        "Machine Learning",
        "Artificial Intelligence",
        "Data Analytics",
        "Web Technologies"
    ],

    "Chemical Engineering (CH)": [
        "Engineering Mathematics",
        "Process Calculations",
        "Fluid Mechanics",
        "Heat Transfer",
        "Mass Transfer",
        "Chemical Reaction Engineering",
        "Instrumentation and Process Control",
        "Plant Design and Economics"
    ],

    "Instrumentation Engineering (IN)": [
        "Engineering Mathematics",
        "Electrical Circuits",
        "Signals and Systems",
        "Transducers",
        "Process Control",
        "Analog Electronics",
        "Digital Electronics",
        "Measurements"
    ],

    "Aerospace Engineering (AE)": [
        "Engineering Mathematics",
        "Flight Mechanics",
        "Aerodynamics",
        "Structures",
        "Propulsion",
        "Space Dynamics"
    ],

    "Biotechnology (BT)": [
        "Engineering Mathematics",
        "Biochemistry",
        "Microbiology",
        "Cell Biology",
        "Immunology",
        "Genetics",
        "Process Biotechnology",
        "Plant and Animal Biotechnology"
    ]
}

# ============== HELPER FUNCTIONS ==============

def get_specializations() -> List[str]:
    """Get list of available GATE specializations"""
    return list(GATE_SPECIALIZATIONS.keys())


def get_subjects(specialization: str) -> List[str]:
    """Get subjects for a given specialization"""
    return GATE_SPECIALIZATIONS.get(specialization, [])


# ============== FLASHCARD GENERATOR CLASS ==============

class GATEFlashcardGenerator:
    """Generate GATE exam flashcards using Gemini LLM"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            temperature=0.3,
            max_retries=2,
            model_kwargs={
                "generation_config": {
                    "thinking_config": {"thinking_budget": 0}
                }
            },
        )

    def generate_flashcards(
        self,
        specialization: str,
        subject: str,
        topic: str,
        num_questions: int = 5
    ) -> List[Dict]:
        """
        Generate MCQ flashcards for GATE preparation.

        Returns:
            List of flashcard dicts with question, options, correct_index, explanation
        """
        prompt = self._build_prompt(specialization, subject, topic, num_questions)
        max_retries = 2

        for attempt in range(max_retries):
            try:
                import signal

                def _timeout_handler(signum, frame):
                    raise TimeoutError("Flashcard generation timed out")

                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(30)
                try:
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                finally:
                    signal.alarm(0)

                flashcards = self._parse_response(response.content)
                if flashcards:
                    return flashcards
            except TimeoutError:
                print(f"Flashcard generation timed out (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"Flashcard generation error (attempt {attempt + 1}/{max_retries}): {e}")

        return []

    def _build_prompt(self, specialization: str, subject: str, topic: str, num: int) -> str:
        return f"""You are a GATE exam preparation expert.

Generate {num} high-quality multiple-choice questions for GATE examination on:

**Specialization**: {specialization}
**Subject**: {subject}
**Topic**: {topic}

**Requirements**:
1. Each question must have exactly 4 options (A, B, C, D)
2. Questions should match GATE difficulty level (conceptual + numerical/problem-solving)
3. Mark exactly ONE correct option (index 0-3)
4. Provide brief explanation for correct answer
5. Mix question types: conceptual, numerical, and application-based
6. Ensure options are technically sound and non-ambiguous

**Output Format** (JSON only, no extra text):

{{
  "flashcards": [
    {{
      "question": "What is the time complexity of Dijkstra's algorithm using binary heap?",
      "options": ["O(V log V)", "O((V+E) log V)", "O(V²)", "O(E log V)"],
      "correct_index": 1,
      "explanation": "Using binary heap, each decrease-key operation takes O(log V) and occurs E times, giving O((V+E) log V)"
    }}
  ]
}}

Generate {num} questions now in this exact JSON format."""

    def _parse_response(self, raw_response: str) -> List[Dict]:
        try:
            raw = raw_response.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end <= start:
                return []
            data = json.loads(raw[start:end])
            validated = []
            for fc in data.get("flashcards", []):
                if self._validate_flashcard(fc):
                    validated.append({
                        "question": fc["question"],
                        "options": fc["options"],
                        "correct_index": fc["correct_index"],
                        "explanation": fc.get("explanation", "No explanation provided")
                    })
            return validated
        except Exception as e:
            print(f"Response parsing error: {e}")
            return []

    def _validate_flashcard(self, fc: Dict) -> bool:
        if not isinstance(fc, dict):
            return False
        if not all(k in fc for k in ["question", "options", "correct_index"]):
            return False
        if not isinstance(fc["options"], list) or len(fc["options"]) != 4:
            return False
        if not isinstance(fc["correct_index"], int):
            return False
        if not (0 <= fc["correct_index"] < 4):
            return False
        return True
