"""
ScholarGen — FastAPI backend
Run from the project root:  uvicorn backend.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Project root is the CWD when uvicorn is invoked from ScholarGen/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, "Config", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from backend.routers import auth, chat, courses, papers, flashcards, collaborate, learningpath, profile, bookmarks, progress, career, analytics
from backend.services.chatbot_service import chatbot_service
@asynccontextmanager
async def lifespan(app: FastAPI):
    await chatbot_service.initialize()
    yield
    await chatbot_service.cleanup()


app = FastAPI(
    title="ScholarGen API",
    version="1.0.0",
    description="AI-powered educational assistant backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,        prefix="/api/auth",        tags=["auth"])
app.include_router(chat.router,        prefix="/api/chat",        tags=["chat"])
app.include_router(courses.router,     prefix="/api/courses",     tags=["courses"])
app.include_router(papers.router,      prefix="/api/papers",      tags=["papers"])
app.include_router(flashcards.router,  prefix="/api/flashcards",  tags=["flashcards"])
app.include_router(collaborate.router,   prefix="/api/collaborate",   tags=["collaborate"])
app.include_router(learningpath.router,  prefix="/api/learningpath",  tags=["learningpath"])
app.include_router(profile.router,       prefix="/api/profile",       tags=["profile"])
app.include_router(bookmarks.router,     prefix="/api/bookmarks",     tags=["bookmarks"])
app.include_router(progress.router,      prefix="/api/progress",      tags=["progress"])
app.include_router(career.router,        prefix="/api/career",        tags=["career"])
app.include_router(analytics.router,     prefix="/api/analytics",     tags=["analytics"])

# Serve React build — only mounted if the build exists.
# Run `npm run build` in frontend/ first.
FRONTEND_DIST = os.path.join(_ROOT, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
    logging.getLogger(__name__).info("Serving frontend from %s", FRONTEND_DIST)
else:
    logging.getLogger(__name__).info(
        "No frontend/dist found — run `npm run build` inside frontend/ to serve the UI"
    )
