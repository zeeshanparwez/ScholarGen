"""
flashcards.py - GATE Exam Flashcard Generator using Gemini LLM
Complete implementation with modal dialog support for Streamlit
"""

import json
import streamlit as st
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv("./Config/.env")

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
            model="gemini-2.0-flash",
            temperature=0.3,  # Lower temperature for consistent exam questions
            max_retries=2
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
        
        Args:
            specialization: GATE stream (e.g., "Computer Science & IT")
            subject: Subject within specialization
            topic: Specific topic for questions
            num_questions: Number of questions to generate
            
        Returns:
            List of flashcard dictionaries with question, options, correct_index, explanation
        """
        
        prompt = self._build_prompt(specialization, subject, topic, num_questions)
        
        try:
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            
            flashcards = self._parse_response(response.content)
            return flashcards
            
        except Exception as e:
            print(f"Flashcard generation error: {e}")
            return []
    
    def _build_prompt(self, specialization: str, subject: str, topic: str, num: int) -> str:
        """Build structured prompt for Gemini"""
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
        """Parse and validate LLM response"""
        try:
            # Extract JSON block
            raw = raw_response.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            
            if start == -1 or end <= start:
                return []
            
            json_str = raw[start:end]
            data = json.loads(json_str)
            
            flashcards = data.get("flashcards", [])
            
            # Validate and clean flashcards
            validated = []
            for fc in flashcards:
                if self._validate_flashcard(fc):
                    validated.append({
                        "question": fc["question"],
                        "options": fc["options"],
                        "correct_index": fc["correct_index"],
                        "explanation": fc.get("explanation", "No explanation provided")
                    })
            
            return validated
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"Response parsing error: {e}")
            return []
    
    def _validate_flashcard(self, fc: Dict) -> bool:
        """Validate flashcard structure"""
        if not isinstance(fc, dict):
            return False
        
        required_keys = ["question", "options", "correct_index"]
        if not all(key in fc for key in required_keys):
            return False
        
        if not isinstance(fc["options"], list) or len(fc["options"]) != 4:
            return False
        
        if not isinstance(fc["correct_index"], int):
            return False
        
        if not (0 <= fc["correct_index"] < 4):
            return False
        
        return True


# ============== STREAMLIT MODAL DIALOG ==============

@st.dialog("GATE Flashcard Practice", width="small")
def show_flashcard_modal():
    """Display flashcard generation and practice in a modal dialog"""

    # Reset internal action flag at the START of the modal
    #st.session_state.flashcard_internal_action = False
    
    # Initialize session state
    if "modal_flashcards" not in st.session_state:
        st.session_state.modal_flashcards = []
    if "current_card_index" not in st.session_state:
        st.session_state.current_card_index = 0
    if "modal_answer" not in st.session_state:
        st.session_state.modal_answer = None
    if "show_explanation" not in st.session_state:
        st.session_state.show_explanation = False
    
    # Initialize generator
    generator = GATEFlashcardGenerator()
    
    # Close button
    col_close, col_space = st.columns([1, 5])
    with col_close:
        if st.button("✕ Close"):
            st.session_state.show_flashcard_dialog = False
            st.rerun()
    
    st.markdown("---")
    
    # If no flashcards generated yet, show generation form
    if not st.session_state.modal_flashcards:
        st.markdown("### 📝 Generate Flashcards")
        
        specialization = st.selectbox(
            "Select Specialization",
            options=get_specializations(),
            key="modal_spec"
        )
        
        subjects = get_subjects(specialization)
        subject = st.selectbox(
            "Select Subject",
            options=subjects,
            key="modal_subj"
        )
        
        topic = st.text_input(
            "Enter Topic",
            placeholder="e.g., Binary Search Trees, Dijkstra's Algorithm",
            key="modal_topic"
        )
        
        num_questions = st.selectbox(
            "Number of Questions",
            options=[3, 5, 10],
            index=1,
            key="modal_num"
        )
        
        if st.button("🎯 Generate Flashcards", type="primary", use_container_width=True):
            #st.session_state.flashcard_internal_action = True  # NEW
            if not topic.strip():
                st.warning("⚠️ Please enter a topic")
            else:
                with st.spinner("🤖 Generating flashcards using Gemini..."):
                    flashcards = generator.generate_flashcards(
                        specialization, subject, topic.strip(), num_questions
                    )
                if flashcards:
                    st.session_state.modal_flashcards = flashcards
                    st.session_state.current_card_index = 0
                    st.session_state.modal_answer = None
                    st.session_state.show_explanation = False
                    st.success(f"✅ Generated {len(flashcards)} flashcards!")
                    st.rerun()
                else:
                    st.error("❌ Failed to generate flashcards. Check API key or try simpler topic.")
    
    # Display current flashcard
    else:
        cards = st.session_state.modal_flashcards
        idx = st.session_state.current_card_index
        card = cards[idx]
        
        # Progress indicator
        st.progress((idx + 1) / len(cards))
        st.caption(f"Question {idx + 1} of {len(cards)}")
        
        st.markdown("---")
        st.markdown(f"**{card['question']}**")
        st.markdown("")
        
        # Show options as buttons
        for i, option in enumerate(card['options']):
            button_type = "primary" if st.session_state.modal_answer == i else "secondary"
            if st.button(
                f"{chr(65 + i)}. {option}",
                key=f"opt_{idx}_{i}",
                use_container_width=True,
                type=button_type
            ):
                #st.session_state.flashcard_internal_action = True  # NEW
                st.session_state.modal_answer = i
                st.session_state.show_explanation = True
                st.rerun()
        
        # Show result if answered
        if st.session_state.show_explanation:
            st.markdown("---")
            correct_idx = card['correct_index']
            
            if st.session_state.modal_answer == correct_idx:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Wrong! Correct answer: **{chr(65 + correct_idx)}**. {card['options'][correct_idx]}")
            
            with st.expander("📖 View Explanation", expanded=True):
                st.info(card['explanation'])
        
        # Navigation buttons
        # Navigation buttons (inside dialog)
        col1, col2, col3 = st.columns(3)

        with col1:
            if idx > 0:
                if st.button("⬅️ Previous", use_container_width=True, key=f"prev_{idx}"):
                    #st.session_state.flashcard_internal_action = True  # NEW
                    st.session_state.current_card_index -= 1
                    st.session_state.modal_answer = None
                    st.session_state.show_explanation = False
                    #st.session_state.flashcard_internal_action = True
                    st.rerun()   # dialog will reopen automatically because flag is True

        with col2:
            if st.button("🔄 New Set", use_container_width=True, key=f"new_{idx}"):
                #st.session_state.flashcard_internal_action = True  # NEW
                st.session_state.modal_flashcards = []
                st.session_state.current_card_index = 0
                st.session_state.modal_answer = None
                st.session_state.show_explanation = False
                #st.session_state.flashcard_internal_action = True
                st.rerun()

        with col3:
            if idx < len(cards) - 1:
                if st.button("Next ➡️", use_container_width=True, key=f"next_{idx}"):
                    #st.session_state.flashcard_internal_action = True  # NEW
                    st.session_state.current_card_index += 1
                    st.session_state.modal_answer = None
                    st.session_state.show_explanation = False
                    #st.session_state.flashcard_internal_action = True
                    st.rerun()
            else:
                if st.button("✅ Finish", use_container_width=True, type="primary", key=f"finish_{idx}"):
                    #st.session_state.flashcard_internal_action = True  # NEW
                    st.session_state.modal_flashcards = []
                    st.session_state.current_card_index = 0
                    st.session_state.modal_answer = None
                    st.session_state.show_explanation = False
                    st.session_state.show_flashcard_dialog = False  # ONLY here we close
                    st.rerun()