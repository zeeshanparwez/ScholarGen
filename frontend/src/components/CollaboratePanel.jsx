import { useState } from 'react'
import { Users, Sparkles, Loader2, UserCircle2, Lightbulb } from 'lucide-react'
import { collaborate } from '../api'

export default function CollaboratePanel() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const find = async () => {
    setLoading(true); setError('')
    try {
      const res = await collaborate.match()
      setData(res)
    } catch (err) {
      setError(err.message)
    } finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <Users size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">Collaborate</h2>
        </div>
        <p className="text-sm text-gray-500">Find students with similar interests and skills</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        {!data && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            <div className="w-16 h-16 bg-accent-50 rounded-2xl flex items-center justify-center mb-4">
              <Users size={32} className="text-accent-400" strokeWidth={1.5} />
            </div>
            <h3 className="font-semibold text-gray-900 mb-1">Find Study Partners</h3>
            <p className="text-sm text-gray-500 max-w-xs mb-6">
              We'll match you with students who share your interests and skills based on your conversations.
            </p>
            <button className="btn-primary" onClick={find}>
              <Sparkles size={16} /> Find Matches
            </button>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center gap-3 py-16 text-gray-500 text-sm">
            <Loader2 size={20} className="animate-spin text-accent-500" />
            Finding similar learners…
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg mb-4">{error}</div>
        )}

        {data && !loading && (
          <div className="space-y-6 animate-fade-in">
            {/* Matched users */}
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <UserCircle2 size={16} className="text-accent-500" /> Matched Learners
              </h3>
              {data.matched_users?.length > 0 ? (
                <div className="grid grid-cols-2 gap-2">
                  {data.matched_users.map((user) => (
                    <div key={user} className="card px-4 py-3 flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-accent-100 text-accent-700 flex items-center justify-center text-sm font-semibold shrink-0">
                        {user[0]?.toUpperCase()}
                      </div>
                      <span className="text-sm font-medium text-gray-900 truncate">{user}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500 bg-gray-50 rounded-xl px-4 py-3">
                  No matches found yet. Keep chatting to build your profile!
                </p>
              )}
            </div>

            {/* Suggested topics */}
            {data.topics?.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                  <Lightbulb size={16} className="text-amber-500" /> Collaboration Topics
                </h3>
                <div className="space-y-2">
                  {data.topics.map((topic, i) => (
                    <div key={i} className="card px-4 py-3 text-sm text-gray-700 leading-relaxed">
                      {topic}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button className="btn-ghost text-sm w-full border border-gray-200" onClick={find}>
              <Loader2 size={14} /> Refresh matches
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
