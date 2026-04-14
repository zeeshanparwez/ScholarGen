import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Zap, BookOpen, Brain, Users } from 'lucide-react'
import { auth } from '../api'

export default function LoginPage() {
  const [tab, setTab] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const handle = async (e) => {
    e.preventDefault()
    setError('')
    if (!username.trim() || !password) return
    setLoading(true)
    try {
      const fn = tab === 'login' ? auth.login : auth.signup
      const data = await fn(username.trim(), password)
      localStorage.setItem('sg_token', data.token)
      localStorage.setItem('sg_user', data.username)
      navigate('/', { replace: true })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* Left panel — branding */}
      <div className="hidden lg:flex w-1/2 bg-gradient-to-br from-accent-600 to-accent-700 text-white flex-col justify-between p-12">
        <div className="flex items-center gap-3">
          <Zap size={32} strokeWidth={1.5} />
          <span className="text-xl font-semibold tracking-tight">UpskillOS</span>
        </div>
        <div>
          <h1 className="text-4xl font-bold leading-tight mb-4">
            AI-powered upskilling<br />for modern professionals
          </h1>
          <p className="text-accent-100 text-lg leading-relaxed mb-10">
            Build personalised learning paths, discover industry insights,
            assess your skills, and connect with your talent network — all in one place.
          </p>
          <div className="grid grid-cols-2 gap-4">
            {[
              { icon: BookOpen, label: 'Learning Paths',    desc: 'Role-based roadmaps' },
              { icon: Brain,    label: 'Industry Insights', desc: 'Research & trends' },
              { icon: Zap,      label: 'Skill Assessment',  desc: 'AI-generated quizzes' },
              { icon: Users,    label: 'Talent Network',    desc: 'Find peers & mentors' },
            ].map(({ icon: Icon, label, desc }) => (
              <div key={label} className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                <Icon size={20} strokeWidth={1.5} className="mb-2 text-accent-100" />
                <div className="font-medium text-sm">{label}</div>
                <div className="text-accent-200 text-xs mt-0.5">{desc}</div>
              </div>
            ))}
          </div>
        </div>
        <p className="text-accent-200 text-sm">Powered by Google Gemini · LangGraph · ChromaDB</p>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-gray-50">
        <div className="w-full max-w-sm">
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center gap-2 mb-8 justify-center">
            <Zap size={28} className="text-accent-600" />
            <span className="text-xl font-semibold text-gray-900">UpskillOS</span>
          </div>

          <div className="card p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-1">
              {tab === 'login' ? 'Welcome back' : 'Create account'}
            </h2>
            <p className="text-gray-500 text-sm mb-6">
              {tab === 'login' ? 'Sign in to continue upskilling' : 'Start your upskilling journey'}
            </p>

            {/* Tabs */}
            <div className="flex bg-gray-100 rounded-lg p-1 mb-6 text-sm font-medium">
              {['login', 'signup'].map(t => (
                <button
                  key={t}
                  onClick={() => { setTab(t); setError('') }}
                  className={`flex-1 py-1.5 rounded-md transition-all ${
                    tab === t ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {t === 'login' ? 'Sign in' : 'Sign up'}
                </button>
              ))}
            </div>

            <form onSubmit={handle} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1.5">Username</label>
                <input
                  className="input-field"
                  placeholder="Enter your username"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  required
                  autoFocus
                  minLength={3}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1.5">Password</label>
                <input
                  type="password"
                  className="input-field"
                  placeholder="Enter your password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  minLength={6}
                />
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 rounded-lg">
                  {error}
                </div>
              )}

              <button type="submit" className="btn-primary w-full py-2.5 mt-2" disabled={loading}>
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                    </svg>
                    {tab === 'login' ? 'Signing in...' : 'Creating account...'}
                  </span>
                ) : tab === 'login' ? 'Sign in' : 'Create account'}
              </button>
            </form>

            {tab === 'signup' && (
              <p className="text-xs text-gray-400 mt-4 text-center">
                Username: 3+ chars · Password: 6+ chars
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
