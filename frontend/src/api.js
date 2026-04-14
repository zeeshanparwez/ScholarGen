const BASE = '/api'

const headers = (extra = {}) => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${localStorage.getItem('sg_token') || ''}`,
  ...extra,
})

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: headers(),
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

// ── Auth ─────────────────────────────────────────────────────────────────────

export const auth = {
  signup: (username, password) =>
    fetch(`${BASE}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }).then(async r => {
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Signup failed')
      return data
    }),

  login: (username, password) =>
    fetch(`${BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }).then(async r => {
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Login failed')
      return data
    }),
}

// ── Chat (SSE streaming) ──────────────────────────────────────────────────────

export const chat = {
  /**
   * Stream a chat message.
   * Calls onToken(str), onToolCall(tool, status), onDone(), onError(str).
   */
  providers: () => request('/chat/providers'),

  stream: async (message, { onToken, onToolCall, onDone, onError, provider = 'gemini' }) => {
    let res
    try {
      res = await fetch(`${BASE}/chat/stream`, {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify({ message, provider }),
      })
    } catch (e) {
      onError(e.message)
      return
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Stream failed' }))
      onError(err.detail || 'Stream failed')
      return
    }

    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const data = JSON.parse(line.slice(6))
          if (data.type === 'token')     onToken(data.content)
          if (data.type === 'tool_call') onToolCall(data.tool, data.status)
          if (data.type === 'done')      onDone()
          if (data.type === 'error')     onError(data.content)
        } catch { /* malformed line — skip */ }
      }
    }
  },

  clear: () => request('/chat/clear', { method: 'DELETE' }),
}

// ── Courses ───────────────────────────────────────────────────────────────────

export const courses = {
  search: (query, topK = 5) =>
    request(`/courses/search?query=${encodeURIComponent(query)}&top_k=${topK}`),
}

// ── Papers ────────────────────────────────────────────────────────────────────

export const papers = {
  search: (topic, maxResults = 5) =>
    request(`/papers/search?topic=${encodeURIComponent(topic)}&max_results=${maxResults}`),
}

// ── Flashcards ────────────────────────────────────────────────────────────────

export const flashcards = {
  specializations: () => request('/flashcards/specializations'),
  generate: (specialization, subject, topic, num_questions = 5) =>
    request('/flashcards/generate', {
      method: 'POST',
      body: JSON.stringify({ specialization, subject, topic, num_questions }),
    }),
}

// ── Collaborate ───────────────────────────────────────────────────────────────

export const collaborate = {
  match: () => request('/collaborate'),
}

// ── Profile ───────────────────────────────────────────────────────────────────

export const profile = {
  get: () => request('/profile'),
  update: (data) => request('/profile', { method: 'PUT', body: JSON.stringify(data) }),
}

// ── Learning Path ─────────────────────────────────────────────────────────────

export const learningpath = {
  generate: (current_role, target_role, job_description = null) =>
    request('/learningpath/generate', {
      method: 'POST',
      body: JSON.stringify({ current_role, target_role, job_description }),
    }),
}
