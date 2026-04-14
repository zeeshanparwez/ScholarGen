import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, Wrench, Search, Globe, Youtube, Database, BookOpen, BookMarked, Zap, ChevronDown, Timer } from 'lucide-react'

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
import LearningPathPanel from '../components/LearningPathPanel'
import ProfilePanel from '../components/ProfilePanel'
import OnboardingModal from '../components/OnboardingModal'
import CareerPanel from '../components/CareerPanel'
import BookmarksPanel from '../components/BookmarksPanel'
import ProgressPanel from '../components/ProgressPanel'
import PomodoroTimer from '../components/PomodoroTimer'
import { chat, profile as profileApi } from '../api'

const PROVIDERS = [
  { id: 'gemini',        label: 'Gemini',    sub: 'gemini-3.1-flash-lite-preview',  color: 'text-blue-600',   tools: true  },
  { id: 'groq',          label: 'Groq',      sub: 'GPT-OSS 120B',                   color: 'text-emerald-600', tools: false },
  { id: 'nim',           label: 'NIM',       sub: 'Llama 3.3 70B',                  color: 'text-green-700',  tools: true  },
  { id: 'nim_llama4',    label: 'NIM',       sub: 'Llama 4 Maverick 17B',           color: 'text-teal-600',   tools: true  },
  { id: 'nim_deepseek',  label: 'DeepSeek',  sub: 'R1 Distill 32B · reasoning',    color: 'text-purple-600', tools: false },
  { id: 'nim_qwq',       label: 'QwQ',       sub: 'QwQ-32B · reasoning',            color: 'text-orange-600', tools: false },
]

const WELCOME = `**Hi! I'm UpskillOS** — your AI-powered upskilling assistant.

Here's what I can do for you:
- **Build your learning path** — tell me your current role and where you want to go
- **Find relevant courses** on any skill or technology
- **Surface industry insights** — latest research and trends from arXiv
- **Assess your skills** — generate quizzes on any topic
- **Summarize articles and videos** — just paste a URL

Try asking: *"I'm a backend developer, what do I need to learn to become an ML engineer?"*`

const EXAMPLE_QUERIES = [
  'What skills do I need to become a cloud architect?',
  'Find learning resources for product management',
  'What are the latest trends in generative AI?',
  'Explain microservices architecture with examples',
]

export default function ChatPage() {
  const [panel, setPanel] = useState('chat')
  const [messages, setMessages] = useState([{ role: 'assistant', content: WELCOME, id: 'welcome' }])
  const [streaming, setStreaming] = useState(false)
  const [activeTools, setActiveTools] = useState([])
  const [runningTools, setRunningTools] = useState(new Set())
  const [provider, setProvider] = useState(
    () => localStorage.getItem('upskill_provider') || 'gemini'
  )
  const [providerOpen, setProviderOpen] = useState(false)
  const [showPomodoro, setShowPomodoro] = useState(false)
  const [streak, setStreak] = useState(0)
  const bottomRef = useRef(null)
  const providerRef = useRef(null)
  const navigate = useNavigate()

  const username = localStorage.getItem('sg_user') || 'User'
  const [showOnboarding, setShowOnboarding] = useState(
    !localStorage.getItem('upskill_onboarded')
  )

  // Record daily activity and fetch streak on mount
  useEffect(() => {
    profileApi.streak()
      .then(d => setStreak(d.streak ?? 0))
      .catch(() => {})
  }, [])

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(scrollToBottom, [messages, scrollToBottom])

  // Close provider dropdown on outside click (ref-based so inside clicks still work)
  useEffect(() => {
    if (!providerOpen) return
    const handler = (e) => {
      if (providerRef.current && !providerRef.current.contains(e.target)) {
        setProviderOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [providerOpen])

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
      provider,
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
  }, [streaming, activeTools, provider])

  const handleExample = (q) => {
    setPanel('chat')
    sendMessage(q)
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {showOnboarding && <OnboardingModal onDone={() => setShowOnboarding(false)} />}
      <Sidebar
        activePanel={panel}
        onNavigate={setPanel}
        onLogout={logout}
        username={username}
        toolsActive={activeTools}
        streak={streak}
      />

      <main className="flex-1 flex flex-col min-w-0 relative">
        {/* ── CHAT PANEL ── */}
        {panel === 'chat' && (
          <>
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-3.5 border-b border-gray-100 shrink-0">
              <div>
                <h1 className="font-semibold text-gray-900 text-sm">UpskillOS Chat</h1>
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

                {/* ── Model selector ── */}
                <div className="relative" ref={providerRef}>
                  <button
                    onClick={() => setProviderOpen(o => !o)}
                    disabled={streaming}
                    className="flex items-center gap-1.5 text-xs border border-gray-200 rounded-lg px-2.5 py-1.5 hover:bg-gray-50 transition-colors"
                  >
                    <span className={`font-medium ${PROVIDERS.find(p => p.id === provider)?.color}`}>
                      {PROVIDERS.find(p => p.id === provider)?.label}
                    </span>
                    <span className="text-gray-400 hidden sm:inline">
                      · {PROVIDERS.find(p => p.id === provider)?.sub}
                    </span>
                    <ChevronDown size={12} className="text-gray-400" />
                  </button>
                  {providerOpen && (
                    <div className="absolute right-0 top-full mt-1 w-56 bg-white border border-gray-200 rounded-xl shadow-lg z-50 py-1 overflow-hidden">
                      {PROVIDERS.map(p => (
                        <button
                          key={p.id}
                          onClick={() => {
                            setProvider(p.id)
                            localStorage.setItem('upskill_provider', p.id)
                            setProviderOpen(false)
                          }}
                          className={`w-full text-left px-3 py-2.5 hover:bg-gray-50 transition-colors ${
                            provider === p.id ? 'bg-gray-50' : ''
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className={`text-sm font-medium ${p.color}`}>{p.label}</span>
                            {p.tools
                              ? <span className="text-[10px] bg-green-50 text-green-700 border border-green-200 px-1.5 py-0.5 rounded-full leading-none">tools</span>
                              : <span className="text-[10px] bg-gray-50 text-gray-400 border border-gray-200 px-1.5 py-0.5 rounded-full leading-none">chat only</span>
                            }
                          </div>
                          <div className="text-xs text-gray-400 mt-0.5">{p.sub}</div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => setShowPomodoro(p => !p)}
                  className={`p-1.5 rounded-lg transition-colors ${showPomodoro ? 'bg-accent-50 text-accent-600' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'}`}
                  title="Pomodoro timer"
                >
                  <Timer size={15} />
                </button>

                <button
                  onClick={clearChat}
                  className="btn-ghost text-xs py-1.5 px-3 text-gray-500"
                  disabled={streaming}
                >
                  Clear chat
                </button>
              </div>
            </div>

            {/* Floating Pomodoro Timer */}
            {showPomodoro && (
              <div className="absolute top-16 right-4 z-40">
                <PomodoroTimer onClose={() => setShowPomodoro(false)} />
              </div>
            )}

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
        {panel === 'learningpath' && <LearningPathPanel />}
        {panel === 'profile'      && <ProfilePanel />}
        {panel === 'courses'      && <CoursesPanel />}
        {panel === 'papers'       && <PapersPanel />}
        {panel === 'flashcards'   && <FlashcardsPanel />}
        {panel === 'collaborate'  && <CollaboratePanel />}
        {panel === 'career'       && <CareerPanel onNavigate={setPanel} />}
        {panel === 'bookmarks'    && <BookmarksPanel />}
        {panel === 'progress'     && <ProgressPanel />}
      </main>
    </div>
  )
}
