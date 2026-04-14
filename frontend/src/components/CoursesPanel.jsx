import { useState, useEffect } from 'react'
import { Search, ExternalLink, BookOpen, Loader2, BookmarkPlus, BookmarkCheck } from 'lucide-react'
import clsx from 'clsx'
import { courses, progress as progressApi } from '../api'

const STATUS_META = {
  saved:       { label: 'Saved',       color: 'bg-amber-50 text-amber-700 border-amber-200' },
  in_progress: { label: 'In Progress', color: 'bg-blue-50 text-blue-700 border-blue-200'    },
  done:        { label: 'Done',        color: 'bg-green-50 text-green-700 border-green-200'  },
}
const STATUS_CYCLE = ['saved', 'in_progress', 'done']

export default function CoursesPanel() {
  const [query, setQuery]         = useState('')
  const [results, setResults]     = useState([])
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState('')
  const [searched, setSearched]   = useState(false)
  const [progressMap, setProgressMap] = useState({})  // url → { id, status }
  const [tracking, setTracking]   = useState(new Set())

  // Load existing progress on mount
  useEffect(() => {
    progressApi.list().then(d => {
      const map = {}
      for (const item of d.items || []) {
        if (item.item_type === 'course') map[item.item_url] = item
      }
      setProgressMap(map)
    }).catch(() => {})
  }, [])

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

  const trackCourse = async (course) => {
    if (!course.url) return
    const url = course.url
    setTracking(s => new Set([...s, url]))
    try {
      const existing = progressMap[url]
      if (existing) {
        const nextStatus = STATUS_CYCLE[(STATUS_CYCLE.indexOf(existing.status) + 1) % STATUS_CYCLE.length]
        await progressApi.update(existing.id, nextStatus)
        setProgressMap(prev => ({ ...prev, [url]: { ...existing, status: nextStatus } }))
      } else {
        const item = await progressApi.add('course', url, course.course_name)
        setProgressMap(prev => ({ ...prev, [url]: item }))
      }
    } catch { /* ignore */ }
    finally {
      setTracking(s => { const n = new Set(s); n.delete(url); return n })
    }
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
            {results.map((course, i) => {
              const tracked = course.url ? progressMap[course.url] : null
              const isTracking = course.url && tracking.has(course.url)
              const statusMeta = tracked ? STATUS_META[tracked.status] : null
              return (
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

                    <div className="flex flex-col items-end gap-2 shrink-0">
                      <div className="w-10 h-10 rounded-lg bg-accent-50 flex items-center justify-center">
                        <BookOpen size={18} className="text-accent-500" strokeWidth={1.5} />
                      </div>
                      {course.url && (
                        <button
                          onClick={() => trackCourse(course)}
                          disabled={isTracking}
                          title={tracked ? `Status: ${tracked.status} — click to advance` : 'Track this course'}
                          className={clsx(
                            'flex items-center gap-1 text-xs px-2 py-1 rounded-lg border transition-all',
                            tracked
                              ? statusMeta?.color
                              : 'text-gray-400 border-gray-200 hover:border-amber-300 hover:text-amber-600 hover:bg-amber-50'
                          )}
                        >
                          {isTracking
                            ? <Loader2 size={10} className="animate-spin" />
                            : tracked
                              ? <BookmarkCheck size={10} />
                              : <BookmarkPlus size={10} />
                          }
                          {isTracking ? '…' : tracked ? statusMeta?.label : 'Track'}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
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
