import { useState, useEffect } from 'react'
import { Bookmark, Trash2, Loader2, BookmarkX } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { bookmarks as bookmarksApi } from '../api'

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

export default function BookmarksPanel() {
  const [items, setItems]     = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState('')

  useEffect(() => {
    bookmarksApi.list()
      .then(d => setItems(d.bookmarks || []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const remove = async (id) => {
    try {
      await bookmarksApi.remove(id)
      setItems(prev => prev.filter(b => b.id !== id))
    } catch { /* ignore */ }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <Bookmark size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">Saved Responses</h2>
        </div>
        <p className="text-sm text-gray-500">
          {items.length > 0 ? `${items.length} saved response${items.length !== 1 ? 's' : ''}` : 'Bookmark AI responses to revisit them here'}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {loading && (
          <div className="flex items-center gap-2 text-gray-400 text-sm py-12 justify-center">
            <Loader2 size={18} className="animate-spin" /> Loading…
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">{error}</div>
        )}

        {!loading && items.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 py-16">
            <BookmarkX size={40} strokeWidth={1} className="mb-3 text-gray-200" />
            <p className="text-sm font-medium text-gray-500">No bookmarks yet</p>
            <p className="text-xs mt-1">Hover over any AI message and click Save to bookmark it</p>
          </div>
        )}

        {!loading && items.length > 0 && (
          <div className="space-y-3">
            {items.map(item => (
              <div key={item.id} className="card p-4 group">
                <div className="flex items-start justify-between gap-3 mb-2">
                  <span className="text-xs text-gray-400">{formatDate(item.timestamp)}</span>
                  <button
                    onClick={() => remove(item.id)}
                    className="text-gray-300 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100 shrink-0"
                    title="Delete bookmark"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
                <div className="text-sm text-gray-700 prose-chat line-clamp-6 overflow-hidden">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.content}</ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
