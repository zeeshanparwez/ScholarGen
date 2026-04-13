import { useState } from 'react'
import { Search, ExternalLink, BookOpen, Loader2 } from 'lucide-react'
import { courses } from '../api'

export default function CoursesPanel() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [searched, setSearched] = useState(false)

  const search = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true); setError(''); setSearched(true)
    try {
      const data = await courses.search(query.trim(), 8)
      setResults(data.results || [])
    } catch (err) {
      setError(err.message)
    } finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <BookOpen size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">NPTEL Courses</h2>
        </div>
        <p className="text-sm text-gray-500">Semantic search over the NPTEL course catalog</p>
      </div>

      <div className="px-6 py-4 border-b border-gray-100">
        <form onSubmit={search} className="flex gap-2">
          <input
            className="input-field flex-1"
            placeholder="e.g. machine learning, data structures, thermodynamics…"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
          <button type="submit" className="btn-primary px-4" disabled={loading || !query.trim()}>
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
            {loading ? 'Searching…' : 'Search'}
          </button>
        </form>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg mb-4">{error}</div>
        )}

        {loading && (
          <div className="flex items-center gap-3 text-gray-500 text-sm py-8 justify-center">
            <Loader2 size={20} className="animate-spin text-accent-500" />
            Searching courses…
          </div>
        )}

        {!loading && searched && results.length === 0 && !error && (
          <div className="text-center text-gray-500 text-sm py-12">No courses found. Try a different query.</div>
        )}

        {!loading && results.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs text-gray-400 mb-3">{results.length} courses found for "{query}"</p>
            {results.map((course, i) => (
              <div key={i} className="card p-4 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-gray-900 text-sm leading-snug mb-1">
                      {course.course_name}
                    </h3>
                    <p className="text-xs text-gray-500 leading-relaxed line-clamp-2">
                      {course.description}
                    </p>
                    <div className="flex items-center gap-3 mt-2">
                      <span className="text-xs text-accent-600 font-medium">
                        {(course.similarity * 100).toFixed(0)}% match
                      </span>
                      {course.url && (
                        <a
                          href={course.url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-accent-600 transition-colors"
                        >
                          <ExternalLink size={11} /> View on NPTEL
                        </a>
                      )}
                    </div>
                  </div>
                  <div className="w-10 h-10 rounded-lg bg-accent-50 flex items-center justify-center shrink-0">
                    <BookOpen size={18} className="text-accent-500" strokeWidth={1.5} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {!searched && (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 py-16">
            <BookOpen size={40} strokeWidth={1} className="mb-3 text-gray-200" />
            <p className="text-sm font-medium text-gray-500">Search NPTEL courses</p>
            <p className="text-xs mt-1">Type a topic above to find relevant courses</p>
          </div>
        )}
      </div>
    </div>
  )
}
