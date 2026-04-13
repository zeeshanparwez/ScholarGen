import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, Wrench, Search, Globe, Youtube, Database, BookOpen, BookMarked, Zap } from 'lucide-react'

const TOOL_META = {
  find_nptel_courses:   { label: 'Searching NPTEL courses',  icon: BookOpen  },
  search_papers:        { label: 'Searching research papers', icon: Search    },
  extract_info:         { label: 'Fetching paper details',    icon: BookMarked },
  search_cached_papers: { label: 'Searching session papers',  icon: Database  },
  get_transcript:       { label: 'Reading YouTube video',     icon: Youtube   },
  fetch:                { label: 'Fetching web page',         icon: Globe     },
}
import Sidebar from '../components/Sidebar'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'
import CoursesPanel from '../components/CoursesPanel'
import PapersPanel from '../components/PapersPanel'
import FlashcardsPanel from '../components/FlashcardModal'
import CollaboratePanel from '../components/CollaboratePanel'
import { chat } from '../api'

const WELCOME = `**Hi! I'm EduAssist** — your AI learning companion.

Here's what I can do for you:
- **Find NPTEL courses** on any topic
- **Search research papers** from arXiv
- **Summarize articles and videos** — just paste a URL
- **Answer academic questions** with step-by-step explanations
- **Generate quizzes** on any concept

Try asking: *"Find me NPTEL courses on machine learning"* or *"Explain transformer architecture"*`

const EXAMPLE_QUERIES = [
  'Find NPTEL courses on deep learning',
  'Summarize recent papers on LLMs',
  'Explain the PageRank algorithm',
  'What is the difference between TCP and UDP?',
]

export default function ChatPage() {
  const [panel, setPanel] = useState('chat')
  const [messages, setMessages] = useState([{ role: 'assistant', content: WELCOME, id: 'welcome' }])
  const [streaming, setStreaming] = useState(false)
  const [activeTools, setActiveTools] = useState([])
  const [runningTools, setRunningTools] = useState(new Set())
  const bottomRef = useRef(null)
  const navigate = useNavigate()

  const username = localStorage.getItem('sg_user') || 'User'

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(scrollToBottom, [messages, scrollToBottom])

  const logout = () => {
    localStorage.removeItem('sg_token')
    localStorage.removeItem('sg_user')
    navigate('/login', { replace: true })
  }

  const clearChat = async () => {
    try { await chat.clear() } catch { /* non-critical */ }
    setMessages([{ role: 'assistant', content: WELCOME, id: 'welcome' }])
  }

  const sendMessage = useCallback(async (text) => {
    if (streaming) return
    const userMsg = { role: 'user', content: text, id: Date.now().toString() }
    const assistantId = (Date.now() + 1).toString()
    const assistantMsg = { role: 'assistant', content: '', id: assistantId, streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setStreaming(true)
    setRunningTools(new Set())

    await chat.stream(text, {
      onToken: (token) => {
        setMessages(prev =>
          prev.map(m => m.id === assistantId ? { ...m, content: m.content + token } : m)
        )
      },
      onToolCall: (tool, status) => {
        setRunningTools(prev => {
          const next = new Set(prev)
          if (status === 'start') next.add(tool)
          else next.delete(tool)
          return next
        })
        if (status === 'start' && !activeTools.includes(tool)) {
          setActiveTools(prev => [...new Set([...prev, tool])])
        }
      },
      onDone: () => {
        setMessages(prev =>
          prev.map(m => m.id === assistantId ? { ...m, streaming: false } : m)
        )
        setStreaming(false)
        setRunningTools(new Set())
      },
      onError: (err) => {
        setMessages(prev =>
          prev.map(m => m.id === assistantId
            ? { ...m, content: `*Error: ${err}*`, streaming: false }
            : m
          )
        )
        setStreaming(false)
        setRunningTools(new Set())
      },
    })
  }, [streaming, activeTools])

  const handleExample = (q) => {
    setPanel('chat')
    sendMessage(q)
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      <Sidebar
        activePanel={panel}
        onNavigate={setPanel}
        onLogout={logout}
        username={username}
        toolsActive={activeTools}
      />

      <main className="flex-1 flex flex-col min-w-0">
        {/* ── CHAT PANEL ── */}
        {panel === 'chat' && (
          <>
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-3.5 border-b border-gray-100 shrink-0">
              <div>
                <h1 className="font-semibold text-gray-900 text-sm">EduAssist Chat</h1>
                <p className="text-xs text-gray-400">
                  {streaming
                    ? runningTools.size > 0
                      ? `${TOOL_META[[...runningTools][0]]?.label ?? [...runningTools][0]}…`
                      : 'Generating response…'
                    : 'Ready'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                {runningTools.size > 0 && (() => {
                  const toolName = [...runningTools][0]
                  const meta = TOOL_META[toolName] ?? { label: toolName, icon: Wrench }
                  const Icon = meta.icon
                  return (
                    <div className="flex items-center gap-1.5 text-xs text-accent-600 bg-accent-50 px-2.5 py-1 rounded-full">
                      <Loader2 size={11} className="animate-spin" />
                      <Icon size={11} />
                      {meta.label}
                    </div>
                  )
                })()}
                <button
                  onClick={clearChat}
                  className="btn-ghost text-xs py-1.5 px-3 text-gray-500"
                  disabled={streaming}
                >
                  Clear chat
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto">
              <div className="max-w-3xl mx-auto py-4">
                {messages.map((msg) => (
                  <ChatMessage key={msg.id} {...msg} />
                ))}
                <div ref={bottomRef} />
              </div>
            </div>

            {/* Example queries — shown only with just the welcome message */}
            {messages.length === 1 && (
              <div className="max-w-3xl mx-auto w-full px-4 pb-2">
                <div className="grid grid-cols-2 gap-2">
                  {EXAMPLE_QUERIES.map(q => (
                    <button
                      key={q}
                      onClick={() => handleExample(q)}
                      className="text-left text-xs text-gray-600 bg-gray-50 hover:bg-accent-50 hover:text-accent-700 border border-gray-100 hover:border-accent-200 px-3 py-2 rounded-xl transition-all"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <ChatInput onSend={sendMessage} disabled={streaming} streaming={streaming} />
          </>
        )}

        {/* ── OTHER PANELS ── */}
        {panel === 'courses'     && <CoursesPanel />}
        {panel === 'papers'      && <PapersPanel />}
        {panel === 'flashcards'  && <FlashcardsPanel />}
        {panel === 'collaborate' && <CollaboratePanel />}
      </main>
    </div>
  )
}
