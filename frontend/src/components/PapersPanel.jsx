import { useState, useEffect } from 'react'
import { Search, ExternalLink, FileText, Loader2, Users, BookmarkPlus, BookmarkCheck } from 'lucide-react'
import clsx from 'clsx'
import { papers, progress as progressApi } from '../api'

const STATUS_META = {
  saved:       { label: 'Saved',       color: 'bg-amber-50 text-amber-700 border-amber-200' },
  in_progress: { label: 'In Progress', color: 'bg-blue-50 text-blue-700 border-blue-200'    },
  done:        { label: 'Done',        color: 'bg-green-50 text-green-700 border-green-200'  },
}
const STATUS_CYCLE = ['saved', 'in_progress', 'done']

export default function PapersPanel() {
  const [topic, setTopic]         = useState('')
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
        if (item.item_type === 'paper') map[item.item_url] = item
      }
      setProgressMap(map)
    }).catch(() => {})
  }, [])

  const search = async (e) => {
    e.preventDefault()
    if (!topic.trim()) return
    setLoading(true); setError(''); setSearched(true)
    try {
      const data = await papers.search(topic.trim(), 8)
      setResults(data.papers || [])
    } catch (err) {
      setError(err.message)
    } finally { setLoading(false) }
  }

  const paperUrl = (paper) => paper.pdf_url || (paper.id ? `https://arxiv.org/abs/${paper.id}` : null)

  const trackPaper = async (paper) => {
    const url = paperUrl(paper)
    if (!url) return
    setTracking(s => new Set([...s, url]))
    try {
      const existing = progressMap[url]
      if (existing) {
        const nextStatus = STATUS_CYCLE[(STATUS_CYCLE.indexOf(existing.status) + 1) % STATUS_CYCLE.length]
        await progressApi.update(existing.id, nextStatus)
        setProgressMap(prev => ({ ...prev, [url]: { ...existing, status: nextStatus } }))
      } else {
        const item = await progressApi.add('paper', url, paper.title)
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
          <FileText size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">Research Papers</h2>
        </div>
        <p className="text-sm text-gray-500">Search arXiv for the latest research</p>
      </div>

      <div className="px-6 py-4 border-b border-gray-100">
        <form onSubmit={search} className="flex gap-2">
          <input
            className="input-field flex-1"
            placeholder="e.g. attention mechanisms, federated learning, BERT…"
            value={topic}
            onChange={e => setTopic(e.target.value)}
          />
          <button type="submit" className="btn-primary px-4" disabled={loading || !topic.trim()}>
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
            Fetching from arXiv…
          </div>
        )}

        {!loading && searched && results.length === 0 && !error && (
          <div className="text-center text-gray-500 text-sm py-12">No papers found. Try a different topic.</div>
        )}

        {!loading && results.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs text-gray-400 mb-3">{results.length} papers found for "{topic}"</p>
            {results.map((paper) => {
              const url = paperUrl(paper)
              const tracked = url ? progressMap[url] : null
              const isTracking = url && tracking.has(url)
              const statusMeta = tracked ? STATUS_META[tracked.status] : null
              return (
                <div key={paper.id} className="card p-4 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-gray-900 text-sm leading-snug mb-1">
                        {paper.title}
                      </h3>
                      <div className="flex items-center gap-1.5 text-xs text-gray-400 mb-2">
                        <Users size={11} />
                        <span className="truncate">{paper.authors?.join(', ')}{paper.authors?.length > 4 ? ' et al.' : ''}</span>
                        <span className="shrink-0 ml-auto">{paper.published}</span>
                      </div>
                      <p className="text-xs text-gray-500 leading-relaxed line-clamp-3">
                        {paper.summary}
                      </p>
                      <div className="mt-2 flex items-center gap-3">
                        <span className="text-xs font-mono text-gray-400">{paper.id}</span>
                        {paper.pdf_url && (
                          <a
                            href={paper.pdf_url}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 text-xs text-accent-600 hover:text-accent-700 font-medium"
                          >
                            <ExternalLink size={11} /> Read PDF
                          </a>
                        )}
                      </div>
                    </div>

                    {url && (
                      <button
                        onClick={() => trackPaper(paper)}
                        disabled={isTracking}
                        title={tracked ? `Status: ${tracked.status} — click to advance` : 'Track this paper'}
                        className={clsx(
                          'flex items-center gap-1 text-xs px-2 py-1 rounded-lg border transition-all shrink-0',
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
              )
            })}
          </div>
        )}

        {!searched && (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 py-16">
            <FileText size={40} strokeWidth={1} className="mb-3 text-gray-200" />
            <p className="text-sm font-medium text-gray-500">Search research papers</p>
            <p className="text-xs mt-1">Powered by the arXiv API</p>
          </div>
        )}
      </div>
    </div>
  )
}
