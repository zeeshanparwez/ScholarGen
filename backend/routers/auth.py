from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.database import create_user, verify_user
from backend.jwt_utils import create_token

router = APIRouter()


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6)


class AuthResponse(BaseModel):
    token: str
    username: str


@router.post("/signup", status_code=201)
async def signup(body: AuthRequest):
    success, message = create_user(body.username.strip(), body.password)
    if not success:
        raise HTTPException(status_code=409, detail=message)
    token = create_token(body.username.strip())
    return AuthResponse(token=token, username=body.username.strip())


@router.post("/login")
async def login(body: AuthRequest):
    success, message = verify_user(body.username.strip(), body.password)
    if not success:
        raise HTTPException(status_code=401, detail=message)
    token = create_token(body.username.strip())
    return AuthResponse(token=token, username=body.username.strip())
