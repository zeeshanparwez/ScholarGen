import { Zap, MessageSquare, BookOpen, FileText, Users, LogOut, ChevronRight, Search, Globe, Youtube, Database, BookMarked, Map, User } from 'lucide-react'
import clsx from 'clsx'

const TOOL_META = {
  find_nptel_courses:   { label: 'NPTEL Course Search',    icon: BookOpen,    desc: 'Finding relevant courses for you' },
  search_papers:        { label: 'arXiv Paper Search',     icon: Search,      desc: 'Searching research papers' },
  extract_info:         { label: 'Paper Details',          icon: BookMarked,  desc: 'Fetching full paper info' },
  search_cached_papers: { label: 'Session Paper Search',   icon: Database,    desc: 'Searching papers from this session' },
  get_transcript:       { label: 'YouTube Transcript',     icon: Youtube,     desc: 'Extracting video transcript' },
  fetch:                { label: 'Web Fetch',              icon: Globe,       desc: 'Reading web content' },
}

const NAV = [
  { id: 'chat',         label: 'AI Assistant',      icon: MessageSquare },
  { id: 'learningpath', label: 'Learning Path',      icon: Map },
  { id: 'courses',      label: 'Course Library',     icon: BookOpen },
  { id: 'papers',       label: 'Industry Insights',  icon: FileText },
  { id: 'flashcards',   label: 'Skill Assessment',   icon: Zap },
  { id: 'collaborate',  label: 'Talent Network',     icon: Users },
  { id: 'profile',      label: 'My Profile',         icon: User },
]

export default function Sidebar({ activePanel, onNavigate, onLogout, username, toolsActive }) {
  return (
    <aside className="w-60 shrink-0 flex flex-col bg-gray-50 border-r border-gray-200 h-full">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 py-4 border-b border-gray-200">
        <div className="w-8 h-8 bg-accent-600 rounded-lg flex items-center justify-center">
          <Zap size={18} className="text-white" strokeWidth={2} />
        </div>
        <span className="font-semibold text-gray-900 tracking-tight">UpskillOS</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        <p className="px-2 mb-2 text-xs font-medium text-gray-400 uppercase tracking-wider">Menu</p>
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            className={clsx(
              'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all text-left group',
              activePanel === id
                ? 'bg-accent-50 text-accent-700'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            )}
          >
            <Icon
              size={16}
              strokeWidth={activePanel === id ? 2.5 : 2}
              className={activePanel === id ? 'text-accent-600' : 'text-gray-400 group-hover:text-gray-600'}
            />
            {label}
            {activePanel === id && <ChevronRight size={14} className="ml-auto text-accent-400" />}
          </button>
        ))}

        {/* Active tools */}
        {toolsActive?.length > 0 && (
          <div className="mt-6">
            <p className="px-2 mb-2 text-xs font-medium text-gray-400 uppercase tracking-wider">Tools Used</p>
            {toolsActive.map(tool => {
              const meta = TOOL_META[tool] ?? { label: tool, icon: Zap, desc: '' }
              const Icon = meta.icon
              return (
                <div key={tool} className="flex items-start gap-2 px-3 py-2 rounded-lg">
                  <div className="w-6 h-6 rounded-md bg-emerald-50 flex items-center justify-center shrink-0 mt-0.5">
                    <Icon size={12} className="text-emerald-600" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-gray-700 leading-tight">{meta.label}</p>
                    <p className="text-xs text-gray-400 leading-tight mt-0.5">{meta.desc}</p>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </nav>

      {/* User + logout */}
      <div className="px-3 py-3 border-t border-gray-200">
        <div className="flex items-center gap-3 px-3 py-2">
          <div className="w-7 h-7 rounded-full bg-accent-100 text-accent-700 flex items-center justify-center text-xs font-semibold shrink-0">
            {username?.[0]?.toUpperCase() ?? 'U'}
          </div>
          <span className="text-sm font-medium text-gray-900 truncate flex-1">{username}</span>
          <button
            onClick={onLogout}
            className="text-gray-400 hover:text-gray-700 p-1 rounded transition-colors"
            title="Sign out"
          >
            <LogOut size={15} />
          </button>
        </div>
      </div>
    </aside>
  )
}
