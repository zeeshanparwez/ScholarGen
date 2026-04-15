import { useState } from 'react'
import { Users, Sparkles, Loader2, UserCircle2, Lightbulb, RefreshCw, Flame } from 'lucide-react'
import { collaborate } from '../api'

export default function CollaboratePanel() {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState('')

  const find = async () => {
    setLoading(true); setError('')
    try {
      const res = await collaborate.match()
      setData(res)
    } catch (err) {
      setError(err.message)
    } finally { setLoading(false) }
  }

  const scoreColor = (score) => {
    if (score >= 60) return 'bg-emerald-100 text-emerald-700'
    if (score >= 30) return 'bg-amber-100 text-amber-700'
    return 'bg-gray-100 text-gray-500'
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <Users size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">Talent Network</h2>
        </div>
        <p className="text-sm text-gray-500">Find colleagues with similar skills and interests</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        {!data && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            <div className="w-16 h-16 bg-accent-50 rounded-2xl flex items-center justify-center mb-4">
              <Users size={32} className="text-accent-400" strokeWidth={1.5} />
            </div>
            <h3 className="font-semibold text-gray-900 mb-1">Find Your Learning Circle</h3>
            <p className="text-sm text-gray-500 max-w-xs mb-6">
              We'll match you with colleagues who share your skills and career goals.
            </p>
            <button className="btn-primary" onClick={find}>
              <Sparkles size={16} /> Find Matches
            </button>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center gap-3 py-16 text-gray-500 text-sm">
            <Loader2 size={20} className="animate-spin text-accent-500" />
            Finding your best matches…
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg mb-4">{error}</div>
        )}

        {data && !loading && (
          <div className="space-y-6">
            {/* Matched users */}
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <UserCircle2 size={16} className="text-accent-500" />
                Top Matches
              </h3>
              {data.matched_users?.length > 0 ? (
                <div className="space-y-2">
                  {data.matched_users.map((match) => {
                    const name = typeof match === 'string' ? match : match.username
                    const role = match.current_role || ''
                    const target = match.target_role || ''
                    const shared = match.shared_skills || []
                    const score = match.score ?? null

                    return (
                      <div key={name} className="card px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div className="w-9 h-9 rounded-full bg-accent-100 text-accent-700 flex items-center justify-center text-sm font-bold shrink-0">
                            {name[0]?.toUpperCase()}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <p className="text-sm font-semibold text-gray-900">{name}</p>
                              {score !== null && (
                                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${scoreColor(score)}`}>
                                  {score}% match
                                </span>
                              )}
                            </div>
                            {role && <p className="text-xs text-gray-400 truncate">{role}{target ? ` → ${target}` : ''}</p>}
                          </div>
                        </div>
                        {shared.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1 pl-12">
                            {shared.map(s => (
                              <span key={s} className="text-[10px] bg-accent-50 text-accent-700 border border-accent-100 rounded-full px-2 py-0.5">
                                {s}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              ) : (
                <p className="text-sm text-gray-500 bg-gray-50 rounded-xl px-4 py-3">
                  No matches found. Update your profile with skills and interests first.
                </p>
              )}
            </div>

            {/* Collaboration topics */}
            {data.topics?.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                  <Lightbulb size={16} className="text-amber-500" /> Collaboration Ideas
                </h3>
                <div className="space-y-2">
                  {data.topics.map((topic, i) => (
                    <div key={i} className="flex items-start gap-3 px-4 py-3 bg-amber-50 border border-amber-100 rounded-xl">
                      <Flame size={13} className="text-amber-500 shrink-0 mt-0.5" />
                      <p className="text-xs text-amber-800 leading-relaxed">{topic}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button className="btn-ghost text-sm w-full border border-gray-200" onClick={find}>
              <RefreshCw size={14} /> Refresh matches
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
