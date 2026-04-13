from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from backend.dependencies import get_current_user
from flashcards import GATEFlashcardGenerator, GATE_SPECIALIZATIONS, get_subjects

router = APIRouter()
_generator = GATEFlashcardGenerator()


class FlashcardRequest(BaseModel):
    specialization: str
    subject: str
    topic: str = Field(..., min_length=2)
    num_questions: int = Field(5, ge=3, le=10)


@router.get("/specializations")
async def get_specializations(username: str = Depends(get_current_user)):
    """Return all GATE specializations and their subjects."""
    return {"specializations": GATE_SPECIALIZATIONS}


@router.post("/generate")
async def generate_flashcards(
    body: FlashcardRequest,
    username: str = Depends(get_current_user),
):
    """Generate GATE MCQ flashcards for a given topic."""
    valid_subjects = get_subjects(body.specialization)
    if not valid_subjects:
        raise HTTPException(status_code=400, detail="Invalid specialization")
    if body.subject not in valid_subjects:
        raise HTTPException(status_code=400, detail="Invalid subject for this specialization")

    flashcards = _generator.generate_flashcards(
        body.specialization, body.subject, body.topic, body.num_questions
    )
    if not flashcards:
        raise HTTPException(status_code=500, detail="Failed to generate flashcards — try again")

    return {"flashcards": flashcards}
