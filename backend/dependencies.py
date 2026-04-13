from fastapi import Depends, HTTPException, Header
from backend.jwt_utils import decode_token


async def get_current_user(authorization: str = Header(...)) -> str:
    """FastAPI dependency — extracts and validates the Bearer JWT token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must start with 'Bearer '")
    token = authorization[len("Bearer "):]
    return decode_token(token)
