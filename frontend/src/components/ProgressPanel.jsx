import { useState, useEffect } from 'react'
import { CheckSquare, Loader2, BookOpen, FileText, Trash2, ExternalLink, BarChart3 } from 'lucide-react'
import clsx from 'clsx'
import { progress as progressApi } from '../api'

const STATUS_CYCLE = ['saved', 'in_progress', 'done']

const STATUS_META = {
  saved:       { label: 'Saved',       color: 'bg-amber-50 text-amber-700 border-amber-200'  },
  in_progress: { label: 'In Progress', color: 'bg-blue-50 text-blue-700 border-blue-200'     },
  done:        { label: 'Done',        color: 'bg-green-50 text-green-700 border-green-200'  },
}

function StatusBadge({ status, onClick, loading }) {
  const meta = STATUS_META[status] || STATUS_META.saved
  return (
    <button
      onClick={onClick}
      disabled={loading}
      title="Click to change status"
      className={clsx(
        'text-xs px-2 py-0.5 rounded-full border font-medium transition-all hover:opacity-80',
        meta.color, loading && 'opacity-50 cursor-not-allowed'
      )}
    >
      {loading ? '…' : meta.label}
    </button>
  )
}

export default function ProgressPanel() {
  const [items, setItems]       = useState([])
  const [loading, setLoading]   = useState(true)
  const [filter, setFilter]     = useState('all')
  const [updating, setUpdating] = useState(new Set())

  useEffect(() => {
    progressApi.list()
      .then(d => setItems(d.items || []))
      .finally(() => setLoading(false))
  }, [])

  const cycleStatus = async (item) => {
    const next = STATUS_CYCLE[(STATUS_CYCLE.indexOf(item.status) + 1) % STATUS_CYCLE.length]
    setUpdating(s => new Set([...s, item.id]))
    try {
      await progressApi.update(item.id, next)
      setItems(prev => prev.map(i => i.id === item.id ? { ...i, status: next } : i))
    } catch { /* ignore */ }
    finally {
      setUpdating(s => { const n = new Set(s); n.delete(item.id); return n })
    }
  }

  const remove = async (id) => {
    try {
      await progressApi.remove(id)
      setItems(prev => prev.filter(i => i.id !== id))
    } catch { /* ignore */ }
  }

  const visible = items.filter(i => filter === 'all' || i.item_type === filter)
  const counts = {
    saved:       items.filter(i => i.status === 'saved').length,
    in_progress: items.filter(i => i.status === 'in_progress').length,
    done:        items.filter(i => i.status === 'done').length,
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <CheckSquare size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">Learning Tracker</h2>
        </div>
        <p className="text-sm text-gray-500">Track your courses and papers progress</p>
      </div>

      {/* Stats */}
      {!loading && items.length > 0 && (
        <div className="px-6 py-3 border-b border-gray-100 flex gap-4">
          {[
            { label: 'Saved',       key: 'saved',       color: 'text-amber-600' },
            { label: 'In Progress', key: 'in_progress', color: 'text-blue-600'  },
            { label: 'Done',        key: 'done',        color: 'text-green-600' },
          ].map(s => (
            <div key={s.key} className="flex items-center gap-1.5">
              <span className={clsx('text-lg font-bold', s.color)}>{counts[s.key]}</span>
              <span className="text-xs text-gray-400">{s.label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Filter tabs */}
      <div className="flex gap-1 px-6 py-2 border-b border-gray-100">
        {[
          { id: 'all',    label: 'All',     count: items.length    },
          { id: 'course', label: 'Courses', count: items.filter(i => i.item_type === 'course').length },
          { id: 'paper',  label: 'Papers',  count: items.filter(i => i.item_type === 'paper').length  },
        ].map(f => (
          <button
            key={f.id}
            onClick={() => setFilter(f.id)}
            className={clsx(
              'text-xs px-3 py-1.5 rounded-lg transition-colors',
              filter === f.id
                ? 'bg-accent-50 text-accent-700 font-medium'
                : 'text-gray-500 hover:bg-gray-100'
            )}
          >
            {f.label} {f.count > 0 && <span className="ml-1 text-gray-400">({f.count})</span>}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {loading && (
          <div className="flex items-center gap-2 text-gray-400 text-sm py-12 justify-center">
            <Loader2 size={18} className="animate-spin" /> Loading…
          </div>
        )}

        {!loading && visible.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 py-16">
            <BarChart3 size={40} strokeWidth={1} className="mb-3 text-gray-200" />
            <p className="text-sm font-medium text-gray-500">Nothing tracked yet</p>
            <p className="text-xs mt-1">Click the bookmark icon on any course or paper to start tracking</p>
          </div>
        )}

        {!loading && visible.length > 0 && (
          <div className="space-y-2">
            {visible.map(item => {
              const Icon = item.item_type === 'course' ? BookOpen : FileText
              return (
                <div key={item.id} className="card p-3.5 flex items-center gap-3 group">
                  <div className={clsx(
                    'w-7 h-7 rounded-md flex items-center justify-center shrink-0',
                    item.item_type === 'course' ? 'bg-accent-50' : 'bg-purple-50'
                  )}>
                    <Icon size={13} className={item.item_type === 'course' ? 'text-accent-600' : 'text-purple-600'} />
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800 truncate">{item.title}</p>
                    <span className="text-xs text-gray-400 capitalize">{item.item_type}</span>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <StatusBadge
                      status={item.status}
                      onClick={() => cycleStatus(item)}
                      loading={updating.has(item.id)}
                    />
                    {item.item_url && (
                      <a
                        href={item.item_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-gray-300 hover:text-accent-600 transition-colors"
                        title="Open"
                      >
                        <ExternalLink size={13} />
                      </a>
                    )}
                    <button
                      onClick={() => remove(item.id)}
                      className="text-gray-200 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
                      title="Remove"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
