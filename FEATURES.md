# ScholarGen — Feature Backlog

Sorted by effort. Everything here fits naturally into the existing architecture — no new dependencies or major rewrites needed.

---

## Quick Wins (< 1 hour each)

### 1. Chat Export as Markdown
**What:** A download button in the chat header that saves the full conversation as a `.md` file.  
**Why:** Students want to keep notes from a research session.  
**Files:** `frontend/src/pages/ChatPage.jsx` only — messages are already in state.  
**How:** Map `messages` array → markdown string → `Blob` → `URL.createObjectURL` → `<a download>` click.  
No backend needed.

---

### 2. Flashcard Score Summary Screen
**What:** After finishing all flashcards, show a results screen: "You got 4 / 5 correct" with a breakdown.  
**Why:** The quiz already tracks correct/incorrect in state — just add a summary panel at the end.  
**Files:** `frontend/src/components/FlashcardModal.jsx` only.  
**How:** When `currentIndex === flashcards.length - 1` and user answers, show a summary screen instead of "Next". Count `results.filter(r => r.correct).length`.

---

### 3. Copy Button on Chat Messages
**What:** A small copy icon on each assistant message that copies the text to clipboard.  
**Why:** Students copy explanations into notes constantly.  
**Files:** `frontend/src/components/ChatMessage.jsx` only.  
**How:** `navigator.clipboard.writeText(content)` on button click. Show a brief "Copied!" tooltip.

---

### 4. Recent Searches (localStorage)
**What:** Show last 5 searches as clickable chips above the search input in CoursesPanel and PapersPanel.  
**Why:** Students search for the same topic multiple times across sessions.  
**Files:** `frontend/src/components/CoursesPanel.jsx`, `frontend/src/components/PapersPanel.jsx`.  
**How:** On each search, push query to `localStorage.getItem('sg_recent_courses')` (JSON array, max 5). Render as chips. Click chip → fills input and triggers search.

---

### 5. Dark Mode Toggle
**What:** A sun/moon icon in the sidebar that switches between light and dark theme.  
**Why:** Studying at night — this is table stakes for any student app.  
**Files:** `frontend/tailwind.config.js` (add `darkMode: 'class'`), `frontend/src/components/Sidebar.jsx`, add `dark:` classes to components.  
**How:** Toggle `dark` class on `document.documentElement`. Store preference in localStorage. Tailwind's `dark:` variants handle the rest.

---

### 6. Clear Chat Confirmation Modal
**What:** Instead of instantly clearing chat, show a small modal "Clear conversation? This cannot be undone."  
**Why:** Users accidentally click Clear and lose context.  
**Files:** `frontend/src/pages/ChatPage.jsx` only.  
**How:** Add a `showConfirm` state. Render a simple modal with Confirm/Cancel buttons. Only call `chat.clear()` on confirm.

---

## Easy Features (1–3 hours each)

### 7. User Profile Page
**What:** A new page showing the user's auto-extracted interests and skills (from conversations), with the ability to edit them manually.  
**Why:** The profile data is already being extracted and saved to SQLite after every chat — it's just never shown to the user.  
**Files:**
- `backend/routers/profile.py` (new) — `GET /api/profile`, `PUT /api/profile`
- `backend/main.py` — register new router
- `frontend/src/api.js` — add `profile.get()`, `profile.update()`
- `frontend/src/pages/ProfilePage.jsx` (new)
- `frontend/src/components/Sidebar.jsx` — add Profile nav item

**Backend:**
```python
@router.get("")
async def get_profile(username: str = Depends(get_current_user)):
    return get_profile(username) or {"interests": [], "skills": []}

@router.put("")
async def update_profile(body: ProfileUpdate, username: str = Depends(get_current_user)):
    upsert_profile(username, body.interests, body.skills, datetime.now().isoformat())
```

**Frontend:** Show interests and skills as editable tag chips. Add/remove tags. Save button calls `PUT /api/profile`.

---

### 8. Course Bookmarks
**What:** A heart icon on each course card in CoursesPanel that saves the course. A "Saved" tab shows all bookmarked courses.  
**Why:** Students find courses they want to revisit later.  
**Files:**
- `backend/core/database.py` — add `bookmarks` table
- `backend/routers/bookmarks.py` (new) — `GET`, `POST`, `DELETE /api/bookmarks/courses`
- `frontend/src/components/CoursesPanel.jsx` — heart icon + saved tab
- `frontend/src/api.js` — bookmark API methods

**Schema addition:**
```sql
CREATE TABLE IF NOT EXISTS course_bookmarks (
    username TEXT, course_name TEXT, url TEXT, description TEXT,
    saved_at TEXT, PRIMARY KEY (username, url)
)
```

---

### 9. Paper Reading List
**What:** A bookmark icon on each paper card in PapersPanel. A "Reading List" tab shows saved papers with read/unread status.  
**Why:** Students collect papers to read later and need to track which they've read.  
**Files:** Same pattern as Course Bookmarks above but for papers.  
**Schema addition:**
```sql
CREATE TABLE IF NOT EXISTS paper_bookmarks (
    username TEXT, paper_id TEXT, title TEXT, summary TEXT,
    pdf_url TEXT, authors TEXT, is_read INTEGER DEFAULT 0,
    saved_at TEXT, PRIMARY KEY (username, paper_id)
)
```
Add a "Mark as Read" toggle on each saved paper card.

---

### 10. Quiz from Any Paper
**What:** A "Generate Quiz" button on each paper card in PapersPanel that sends the paper's title + summary to the flashcard generator.  
**Why:** The flashcard generator already works. Papers already have summaries. This just connects them.  
**Files:**
- `backend/routers/flashcards.py` — add `POST /api/flashcards/from-paper` endpoint
- `frontend/src/components/PapersPanel.jsx` — "Quiz me on this" button per paper card
- `frontend/src/components/FlashcardModal.jsx` — accept pre-generated flashcards as prop

**Backend change:** Accept free-form `context` text instead of requiring specialization/subject. Build a different prompt: "Generate {n} MCQs based on this paper: {title}\n\n{summary}".

---

### 11. Recommended Courses on Chat Home
**What:** When the user opens the chat and has no messages yet, show 3 recommended NPTEL courses based on their profile interests.  
**Why:** The profile interests are already stored — surfacing them creates a personalized experience from login.  
**Files:**
- `frontend/src/pages/ChatPage.jsx` — on mount, fetch profile interests → call `courses.search()`
- `frontend/src/api.js` — already has `courses.search()`

**How:** In `useEffect` on mount, call `profile.get()`. If interests exist, call `courses.search(interests[0])`. Show results as soft cards above the example queries. Zero backend changes needed.

---

### 12. Study Streak Counter
**What:** A fire icon in the sidebar showing "🔥 3 day streak" — increments every day the user chats.  
**Why:** Streaks are a proven engagement mechanic. Easy to add.  
**Files:**
- `backend/core/database.py` — add `last_active TEXT, streak_days INTEGER DEFAULT 0` to `user_profiles`
- `backend/routers/auth.py` or `backend/routers/chat.py` — update streak on each login/chat
- `backend/routers/profile.py` — return streak in `GET /api/profile`
- `frontend/src/components/Sidebar.jsx` — show streak badge

**Logic:** On each chat message, compare `last_active` date to today. If yesterday → increment streak. If today → no change. If older → reset to 1. Update `last_active = today`.

---

## Moderate (half day each)

### 13. Flashcard History & Stats
**What:** Track all flashcard sessions — which topic, how many correct, when. Show a history table and overall accuracy per subject.  
**Why:** Students want to know which GATE subjects they're weak in.  
**Files:**
- `backend/core/database.py` — add `flashcard_sessions` table
- `backend/routers/flashcards.py` — save session result on `POST /api/flashcards/result`
- New `StatsPanel.jsx` in frontend showing a table of sessions + accuracy per subject

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS flashcard_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT, specialization TEXT, subject TEXT, topic TEXT,
    total INTEGER, correct INTEGER, taken_at TEXT
)
```

---

### 14. Share Chat Link (Read-Only)
**What:** A "Share" button that generates a public read-only link to the current conversation.  
**Why:** Students want to share a research session with friends or professors.  
**Files:**
- `backend/core/database.py` — add `shared_chats` table (id, username, messages_json, created_at)
- `backend/routers/share.py` (new) — `POST /api/share` → returns share_id; `GET /api/share/{id}` → returns messages (no auth)
- `frontend/src/pages/ChatPage.jsx` — Share button
- `frontend/src/pages/SharedChatPage.jsx` (new) — read-only view

---

### 15. Markdown Notes Pad
**What:** A simple persistent notes panel in the sidebar where students can jot down notes in Markdown. Notes are saved to the backend per user.  
**Why:** Students want a scratch pad alongside the AI — "remember this" moment captured instantly.  
**Files:**
- `backend/core/database.py` — add `notes TEXT DEFAULT ''` column to `user_profiles`
- `backend/routers/profile.py` — `GET/PUT /api/notes`
- `frontend/src/components/NotesPanel.jsx` (new) — textarea with live Markdown preview, auto-save on blur

---

## Implementation Priority Order

If doing these one by one, recommended order:

```
Day 1 Morning  → 1. Chat Export  +  2. Flashcard Score Summary  +  3. Copy Button
Day 1 Afternoon → 5. Dark Mode  +  4. Recent Searches  +  6. Clear Confirmation
Day 2 Morning  → 7. Profile Page  (backend + frontend)
Day 2 Afternoon → 8. Course Bookmarks  +  9. Paper Reading List
Day 3          → 10. Quiz from Paper  +  11. Recommended Courses
Day 4          → 12. Study Streak  +  13. Flashcard Stats
Day 5          → 14. Share Link  +  15. Notes Pad
```

---

## No External Dependencies Needed

All features above use only what's already installed:
- SQLite (stdlib) for new tables
- React state / localStorage for client-side features
- Existing Gemini API for quiz-from-paper
- Existing FastAPI patterns for new endpoints
- Tailwind CSS `dark:` variants for dark mode
