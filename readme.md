# ScholarGen — AI-Powered Student Learning Assistant

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.7-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-3.1%20Flash%20Lite-4285F4.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ScholarGen (EduAssist)** is an AI-powered educational assistant that helps students find courses, discover research papers, generate flashcards, and collaborate with peers — all through a streaming chat interface powered by Google Gemini and LangGraph.

## Features

- **Streaming Chat** — token-by-token SSE streaming with real-time tool-use indicators
- **NPTEL Course Search** — semantic search across 3,200+ NPTEL courses using ChromaDB
- **Research Paper Discovery** — arXiv search, full paper extraction, and in-session semantic cache
- **YouTube Transcript** — extract and summarize transcripts from educational videos
- **Web Fetch** — retrieve and analyze any webpage or article
- **GATE Flashcards** — AI-generated MCQ flashcards for GATE exam preparation
- **Study Partner Matching** — match with similar users based on interests and skills
- **Per-User Memory** — isolated conversation history per account via LangGraph thread IDs
- **JWT Authentication** — secure login/signup with bcrypt password hashing

## Architecture

```
Browser (React + Vite + Tailwind)
  └── /api/* → FastAPI (uvicorn :8000)
        ├── /api/auth        — JWT login / signup
        ├── /api/chat/stream — SSE streaming chat (LangGraph ReAct agent)
        ├── /api/courses     — NPTEL semantic search (ChromaDB)
        ├── /api/papers      — arXiv paper search
        ├── /api/flashcards  — GATE MCQ generation (Gemini)
        └── /api/collaborate — User similarity matching

LangGraph ReAct Agent
  ├── LLM: Gemini 3.1 Flash Lite (langchain-google-genai)
  ├── Memory: MemorySaver (per-user UUID thread_id)
  └── Tools (via MCP + LangChain):
        ├── find_nptel_courses    — ChromaDB semantic search
        ├── search_papers         — arXiv search (research_mcp.py)
        ├── extract_info          — Full paper metadata (research_mcp.py)
        ├── search_cached_papers  — In-session ChromaDB paper cache
        ├── get_transcript        — YouTube transcript (youtube_mcp.py)
        └── fetch                 — Web content (mcp-server-fetch via uvx)
```

## Project Structure

```
ScholarGen/
├── backend/
│   ├── main.py                      # FastAPI app, lifespan, router registration
│   ├── jwt_utils.py                 # JWT create/decode (48h expiry)
│   ├── dependencies.py              # get_current_user FastAPI dependency
│   ├── agent_orchestrator.py        # MCPSessionManager — launches MCP subprocesses
│   ├── services/
│   │   └── chatbot_service.py       # Singleton LangGraph agent (init at startup)
│   ├── routers/
│   │   ├── auth.py                  # POST /register, POST /login
│   │   ├── chat.py                  # POST /stream, DELETE /clear
│   │   ├── courses.py               # GET /search
│   │   ├── papers.py                # GET /search
│   │   ├── flashcards.py            # GET /subjects, POST /generate
│   │   └── collaborate.py           # GET / (user matching)
│   ├── core/
│   │   ├── database.py              # SQLite — users + user_profiles (WAL mode)
│   │   ├── course_retriever.py      # ChromaDB loader + LangChain tool wrapper
│   │   ├── collaboration.py         # Profile extraction + cosine similarity matching
│   │   └── flashcards.py            # GATE MCQ generation via Gemini
│   └── mcp/
│       ├── research_mcp.py          # FastMCP — arXiv search + ChromaDB paper cache
│       └── youtube_mcp.py           # FastMCP — YouTube transcript extraction
├── frontend/
│   ├── src/
│   │   ├── api.js                   # API client with SSE parser
│   │   ├── pages/
│   │   │   ├── LoginPage.jsx
│   │   │   └── ChatPage.jsx
│   │   └── components/
│   │       ├── Sidebar.jsx          # Tool activity + panel navigation
│   │       ├── ChatMessage.jsx      # Markdown rendering
│   │       ├── ChatInput.jsx
│   │       ├── CoursesPanel.jsx
│   │       ├── PapersPanel.jsx
│   │       ├── CollaboratePanel.jsx
│   │       └── FlashcardModal.jsx
│   ├── dist/                        # Built output — served by FastAPI at /
│   └── package.json
├── Config/
│   └── .env                         # API keys and secrets (not committed)
├── Data/
│   ├── nptel_courses_with_embeddings.xlsx   # Pre-embedded NPTEL catalog
│   └── scholargen.db                        # SQLite DB (auto-created on first run)
├── scripts/
│   └── generate_embeddings.py       # Re-embed NPTEL catalog with Gemini API
├── requirements.txt
└── readme.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [`uv`](https://docs.astral.sh/uv/) — required for MCP server management

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone and install

```bash
git clone https://github.com/zeeshanparwez/ScholarGen.git
cd ScholarGen
pip install -r requirements.txt
```

### 2. Configure environment

Create `Config/.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
JWT_SECRET_KEY=any_random_secret_string

# Optional — defaults shown
GEMINI_MODEL=gemini-3.1-flash-lite-preview
COURSE_DATA_PATH=/absolute/path/to/Data/nptel_courses_with_embeddings.xlsx
```

Get a free Google API key at [aistudio.google.com](https://aistudio.google.com).

### 3. Build the frontend

```bash
cd frontend
npm install
npm run build   # outputs to frontend/dist/ — served automatically by FastAPI
cd ..
```

### 4. Run

```bash
uvicorn backend.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000), create an account, and start chatting.

### Development mode (hot-reload on both sides)

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend (Vite dev server, proxies /api to :8000)
cd frontend && npm run dev
# Open http://localhost:5173
```

## Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI Studio key (Gemini + Embeddings) | Yes |
| `JWT_SECRET_KEY` | Any random secret string for JWT signing | Yes |
| `GEMINI_MODEL` | Gemini model name | No (default: `gemini-3.1-flash-lite-preview`) |
| `COURSE_DATA_PATH` | Absolute path to NPTEL Excel file | No (auto-detected) |

## SSE Streaming Protocol

`POST /api/chat/stream` returns `text/event-stream`:

```
data: {"type": "token",     "content": "Hello"}
data: {"type": "tool_call", "tool": "find_nptel_courses", "status": "start"}
data: {"type": "tool_call", "tool": "find_nptel_courses", "status": "end"}
data: {"type": "done"}
data: {"type": "error",     "content": "..."}
```

## Example Queries

| Goal | Query |
|------|-------|
| Find courses | "Recommend NPTEL courses for deep learning" |
| Research papers | "Find recent papers on transformer architectures" |
| Video summary | "Get the transcript from this YouTube lecture: [URL]" |
| Concept help | "Explain attention mechanism step by step" |
| Flashcards | Use the Flashcards panel → select GATE branch → generate |
| Study partners | Use the Collaborate panel to find similar users |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI 0.115, uvicorn, Python 3.11 |
| Frontend | React 18, Vite 5, Tailwind CSS 3 |
| Agent | LangGraph 0.6.7, LangChain 0.3.27 |
| LLM | Google Gemini 3.1 Flash Lite |
| Vector store | ChromaDB 0.6.3 (in-memory) |
| MCP tools | FastMCP, mcp-server-fetch (via uvx) |
| Auth | PyJWT 2.9.0, bcrypt |
| Database | SQLite (stdlib) — WAL mode |
| Embeddings | Gemini Embedding API (`gemini-embedding-2-preview`, 768d) |

## Deployment (Free Tier)

Recommended stack: **Render** (backend) + **Vercel** (frontend) + **Supabase** (optional DB)

**Render (backend):**
- Build command: `pip install -r requirements.txt && cd frontend && npm install && npm run build`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Set env vars: `GOOGLE_API_KEY`, `JWT_SECRET_KEY`

**Vercel (frontend, optional separate deploy):**
- Build: `npm run build` from `frontend/`
- Output: `frontend/dist`
- Set `VITE_API_URL` to your Render backend URL

## License

MIT License — see [LICENSE](LICENSE) for details.
