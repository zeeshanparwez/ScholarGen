# ScholarGen (EduAssist)

AI-powered educational research and learning assistant — FastAPI backend + React frontend.

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
```bash
# Config/.env must contain:
GOOGLE_API_KEY=<your_google_genai_key>
JWT_SECRET_KEY=<any_random_secret_string>
COURSE_DATA_PATH=/absolute/path/to/Data/nptel_courses_with_embeddings.xlsx
# Optional:
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

`uv` is required for MCP server management:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install and build frontend
```bash
cd frontend
npm install
npm run build   # produces frontend/dist/ — served by FastAPI
cd ..
```

### 4. Run the app
```bash
# From ScholarGen/ root:
uvicorn backend.main:app --reload --port 8000
# Open http://localhost:8000
```

### Dev mode (hot-reload on both sides)
```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend (Vite dev server with /api proxy)
cd frontend && npm run dev
# Open http://localhost:5173
```

## Architecture

```
Browser (React + Tailwind)
  └── /api/* → FastAPI (uvicorn :8000)
        ├── /api/auth        — JWT login/signup (bcrypt + SQLite)
        ├── /api/chat/stream — SSE token-by-token streaming
        ├── /api/courses     — NPTEL semantic search (ChromaDB)
        ├── /api/papers      — arXiv search
        ├── /api/flashcards  — GATE MCQ generation (Gemini)
        └── /api/collaborate — User matching (embeddings)

ChatbotService (singleton)
  └── LangGraph ReAct agent (create_react_agent)
        ├── Gemini 2.0-flash (LLM)
        ├── MemorySaver (per-user thread isolation via UUID thread_id)
        └── Tools: find_nptel_courses, search_papers, extract_info,
                   search_cached_papers, get_transcript, fetch
```

## Key Files

| Path | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app, lifespan, routers, static mount |
| `backend/jwt_utils.py` | JWT create/decode (48h expiry) |
| `backend/dependencies.py` | `get_current_user` FastAPI dependency |
| `backend/routers/` | auth, chat, courses, papers, flashcards, collaborate |
| `backend/services/chatbot_service.py` | Singleton LangGraph agent service |
| `database.py` | SQLite layer (users + profiles, bcrypt, WAL mode) |
| `agent_orchestrator.py` | MCPSessionManager, LangGraph agent builder |
| `course_retriever.py` | ChromaDB in-memory NPTEL semantic search |
| `research_mcp.py` | FastMCP server — arXiv search + ChromaDB paper cache |
| `collaboration.py` | User profile matching + Gemini profile extraction |
| `flashcards.py` | GATE MCQ generation via Gemini |
| `frontend/src/` | React app (Vite + Tailwind) |
| `frontend/src/api.js` | API client with SSE stream parser |
| `frontend/src/pages/` | LoginPage, ChatPage |
| `frontend/src/components/` | Sidebar, ChatMessage, ChatInput, panels |

## Data

| Path | Contents |
|------|---------|
| `Config/.env` | API keys and secrets |
| `Data/nptel_courses_with_embeddings.xlsx` | Pre-embedded NPTEL course catalog (loaded into ChromaDB at first use) |
| `Data/scholargen.db` | SQLite — users + user_profiles (auto-created on first run) |

## SSE Streaming Protocol

`POST /api/chat/stream` responds with `text/event-stream`:

```
data: {"type": "token",     "content": "Hello"}
data: {"type": "tool_call", "tool": "find_nptel_courses", "status": "start"}
data: {"type": "tool_call", "tool": "find_nptel_courses", "status": "end"}
data: {"type": "done"}
data: {"type": "error",     "content": "..."}
```

## Deployment (Render free tier)

1. Set env vars in Render dashboard
2. Build command: `pip install -r requirements.txt && cd frontend && npm install && npm run build`
3. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## Tech Stack

- **FastAPI** 0.115 + **uvicorn** — backend
- **LangGraph** 0.6.7 + **LangChain** 0.3.27 — agent orchestration
- **Google Gemini 2.0-flash** (`langchain-google-genai`) — LLM
- **ChromaDB** 0.6.3 — in-memory vector store (courses + papers)
- **SQLite** (stdlib) — user storage with bcrypt + WAL mode
- **PyJWT** 2.9.0 — authentication tokens
- **MCP** 1.13.1 + **FastMCP** — tool server protocol (arXiv, YouTube, fetch)
- **SentenceTransformers** (`BAAI/bge-base-en-v1.5`) — embeddings
- **React 18** + **Vite 5** + **Tailwind CSS 3** — frontend
